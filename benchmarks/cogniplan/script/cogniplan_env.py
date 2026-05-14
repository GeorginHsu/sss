"""CogniPlan Exploration Environment.

Faithful to the official CogniPlan evaluation:
  - Raycasting sensor (720 rays, 0.5° step) that respects walls
  - Sparse node graph (NODE_RESOLUTION=3.6 m), edges only through explored free space
  - Travel distance accumulated in metres (CELL_SIZE=0.4 m/cell)
  - Start position from marker pixel (value 208) in map
  - Episode ends when explored_rate > 0.9999 or all frontier-utility exhausted
  - Maximum 128 planning steps

Map convention: FREE=255, OCCUPIED=1, UNKNOWN=127

Observation dict:
    global_map    uint8 (H, W) — belief map (FREE=255, OCCUPIED=1, UNKNOWN=127)
    robot_pos     float32 (2,) — (x, y) in metres
    frontiers     list[(float, float)] — frontier coords in metres
    nodes         list[dict] — {coords, utility, visited, neighbors}
    neighbors     list[(float, float)] — reachable neighbour coords in metres
    explored_rate float

env.step(target) — target (x, y) in metres; snapped to nearest neighbour node.
env.get_metrics() — returns explored_rate, travel_distance, n_steps, success.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label as sk_label

# ── Official CogniPlan constants (parameter.py) ───────────────────────────────
CELL_SIZE         = 0.4                          # metres per grid cell
NODE_RESOLUTION   = 3.6                          # metres between graph nodes
SENSOR_RANGE      = 16.0                         # metres
UTILITY_RANGE     = 0.8 * SENSOR_RANGE           # 12.8 m
MIN_UTILITY       = 2
FRONTIER_CELL_SIZE = 2 * CELL_SIZE               # 0.8 m
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # 78.4 m
MAX_EPISODE_STEP  = 128
SUCCESS_THRESHOLD = 0.9999

FREE     = 255
OCCUPIED = 1
UNKNOWN  = 127


# ── Sensor (sensor.py) ────────────────────────────────────────────────────────

def _ray_cast(
    x0: float, y0: float, x1: float, y1: float,
    ground_truth: np.ndarray, robot_belief: np.ndarray,
) -> np.ndarray:
    """Bresenham ray: reveal cells, stop after 10 consecutive wall hits."""
    x0, y0, x1, y1 = round(x0), round(y0), round(x1), round(y1)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2
    collision_flag = 0
    max_collision = 10

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = int(ground_truth[y, x])
        if k == OCCUPIED:
            collision_flag += 1
            if collision_flag >= max_collision:
                break
        elif collision_flag > 0:
            break
        if x == x1 and y == y1:
            break
        robot_belief[y, x] = k
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return robot_belief


def _sensor_work(
    robot_cell: np.ndarray,
    sensor_range_cells: float,
    robot_belief: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """360° raycasting sensor, 720 rays at 0.5° increments."""
    angle_inc = 0.5 / 180 * math.pi
    x0, y0 = float(robot_cell[0]), float(robot_cell[1])
    angle = 0.0
    while angle < 2 * math.pi:
        x1 = x0 + math.cos(angle) * sensor_range_cells
        y1 = y0 + math.sin(angle) * sensor_range_cells
        robot_belief = _ray_cast(x0, y0, x1, y1, ground_truth, robot_belief)
        angle += angle_inc
    return robot_belief


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _coords_to_cell(
    coords: np.ndarray, origin_x: float, origin_y: float
) -> np.ndarray:
    col = int(round((coords[0] - origin_x) / CELL_SIZE))
    row = int(round((coords[1] - origin_y) / CELL_SIZE))
    return np.array([col, row], dtype=int)


def _check_collision(
    start: np.ndarray, end: np.ndarray,
    belief: np.ndarray, origin_x: float, origin_y: float,
) -> bool:
    """Bresenham collision check on belief map; OCCUPIED or UNKNOWN = blocked."""
    sc = _coords_to_cell(start, origin_x, origin_y)
    ec = _coords_to_cell(end, origin_x, origin_y)
    x0, y0 = int(sc[0]), int(sc[1])
    x1, y1 = int(ec[0]), int(ec[1])
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2
    while 0 <= x < belief.shape[1] and 0 <= y < belief.shape[0]:
        k = int(belief[y, x])
        if k == OCCUPIED or k == UNKNOWN:
            return True
        if x == x1 and y == y1:
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False


# ── Node ──────────────────────────────────────────────────────────────────────

class _Node:
    """Exploration graph node (matches official CogniPlan Node)."""

    __slots__ = (
        "coords", "observable_frontiers", "utility", "visited",
        "neighbor_set", "neighbor_matrix", "need_update_neighbor",
    )

    def __init__(
        self, coords: np.ndarray, frontiers: set, env: "ExplorationEnv"
    ) -> None:
        self.coords = coords
        self.observable_frontiers: set = set()
        self.utility: int = 0
        self.visited: bool = False
        # 5x5 neighbour matrix: -1=unchecked, 0=blocked, 1=reachable
        self.neighbor_matrix = -np.ones((5, 5), dtype=np.int8)
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set: set = {(float(coords[0]), float(coords[1]))}
        self.need_update_neighbor: bool = True
        self._init_frontiers(frontiers, env)

    def _init_frontiers(self, frontiers: set, env: "ExplorationEnv") -> None:
        if not frontiers:
            return
        farr = np.array(list(frontiers))
        dists = np.linalg.norm(farr - self.coords, axis=1)
        for pt in farr[dists < UTILITY_RANGE]:
            if not _check_collision(
                self.coords, pt, env.robot_belief,
                env.belief_origin_x, env.belief_origin_y,
            ):
                self.observable_frontiers.add((float(pt[0]), float(pt[1])))
        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def update_frontiers(
        self, new_frontiers: set, global_frontiers: set, env: "ExplorationEnv"
    ) -> None:
        self.observable_frontiers -= self.observable_frontiers - global_frontiers
        if new_frontiers:
            farr = np.array(list(new_frontiers))
            dists = np.linalg.norm(farr - self.coords, axis=1)
            for pt in farr[dists < UTILITY_RANGE]:
                if not _check_collision(
                    self.coords, pt, env.robot_belief,
                    env.belief_origin_x, env.belief_origin_y,
                ):
                    self.observable_frontiers.add((float(pt[0]), float(pt[1])))
        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def update_neighbors(
        self, nodes: dict, env: "ExplorationEnv"
    ) -> None:
        ci = 2
        for i in range(5):
            for j in range(5):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                nb_coords = np.round(
                    np.array([
                        self.coords[0] + (i - ci) * NODE_RESOLUTION,
                        self.coords[1] + (j - ci) * NODE_RESOLUTION,
                    ]), 1,
                )
                nb_key = (float(nb_coords[0]), float(nb_coords[1]))
                if nb_key not in nodes:
                    continue
                blocked = _check_collision(
                    self.coords, nb_coords, env.robot_belief,
                    env.belief_origin_x, env.belief_origin_y,
                )
                val = 0 if blocked else 1
                self.neighbor_matrix[i, j] = val
                if not blocked:
                    self.neighbor_set.add(nb_key)
                    nb_node = nodes[nb_key]
                    nb_node.neighbor_matrix[ci + (ci - i), ci + (ci - j)] = 1
                    nb_node.neighbor_set.add(
                        (float(self.coords[0]), float(self.coords[1]))
                    )
        if self.utility == 0:
            self.need_update_neighbor = False

    def set_visited(self) -> None:
        self.visited = True
        self.observable_frontiers = set()
        self.utility = 0


# ── Main Environment ──────────────────────────────────────────────────────────

class ExplorationEnv:
    """2D occupancy-grid exploration environment, faithful to official CogniPlan."""

    def __init__(
        self,
        map_path: str,
        sensor_range: float = SENSOR_RANGE,
        max_steps: int = MAX_EPISODE_STEP,
        success_threshold: float = SUCCESS_THRESHOLD,
    ) -> None:
        self.map_path = map_path
        self.sensor_range = sensor_range
        self.max_steps = max_steps
        self.success_threshold = success_threshold

        # ── Load and binarise map (matches official import_ground_truth) ──────
        raw = np.array(Image.open(map_path).convert("L"))

        # Start position: pixel 208, take 11th occurrence (index 10)
        positions = np.argwhere(raw == 208)  # (N,2) as (row, col)
        if len(positions) == 0:
            raise ValueError(f"No start marker (pixel 208) in {map_path}")
        row10, col10 = positions[min(10, len(positions) - 1)]

        # free = >150 OR 50–80 (matches official binarisation)
        gt_bool = (raw > 150) | ((raw <= 80) & (raw >= 50))
        self.gt_map: np.ndarray = (gt_bool.astype(np.uint8) * 254 + 1)
        self.height, self.width = self.gt_map.shape

        # Robot starts at metric origin (0, 0)
        self.belief_origin_x: float = round(-col10 * CELL_SIZE, 1)
        self.belief_origin_y: float = round(-row10 * CELL_SIZE, 1)
        self._start_cell = np.array([col10, row10], dtype=int)

        self.total_free: int = int(np.sum(self.gt_map == FREE))
        if self.total_free == 0:
            raise ValueError(f"Map {map_path} has no free cells")

        # Episode state (initialised in reset)
        self.robot_belief: np.ndarray = np.full(self.gt_map.shape, UNKNOWN, dtype=np.uint8)
        self.robot_location: np.ndarray = np.array([0.0, 0.0])
        self.robot_cell: np.ndarray = self._start_cell.copy()
        self.travel_dist: float = 0.0
        self.explored_rate: float = 0.0
        self._done: bool = False
        self.n_steps: int = 0
        self._nodes: dict = {}
        self._prev_frontiers: set = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self) -> dict[str, Any]:
        self.robot_belief = np.full(self.gt_map.shape, UNKNOWN, dtype=np.uint8)
        self.robot_location = np.array([0.0, 0.0])
        self.robot_cell = self._start_cell.copy()
        self.travel_dist = 0.0
        self.explored_rate = 0.0
        self._done = False
        self.n_steps = 0
        self._nodes = {}
        self._prev_frontiers = set()

        sensor_cells = round(self.sensor_range / CELL_SIZE)
        self.robot_belief = _sensor_work(
            self.robot_cell, sensor_cells, self.robot_belief, self.gt_map
        )
        self._eval_explored_rate()
        self._update_graph()
        return self._get_obs()

    def step(
        self, target: tuple[float, float]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._done:
            return self._get_obs(), self._get_info()

        next_loc = self._resolve_target(
            np.array([float(target[0]), float(target[1])])
        )

        dist = float(np.linalg.norm(self.robot_location - next_loc))
        self.robot_location = next_loc
        self.robot_cell = _coords_to_cell(
            next_loc, self.belief_origin_x, self.belief_origin_y
        )
        self.travel_dist += dist

        sensor_cells = round(self.sensor_range / CELL_SIZE)
        self.robot_belief = _sensor_work(
            self.robot_cell, sensor_cells, self.robot_belief, self.gt_map
        )

        key = (round(float(next_loc[0]), 1), round(float(next_loc[1]), 1))
        if key in self._nodes:
            self._nodes[key].set_visited()

        self._eval_explored_rate()
        self._update_graph()
        self.n_steps += 1

        has_utility = any(n.utility > 0 for n in self._nodes.values())
        if (
            self.explored_rate > self.success_threshold
            or not has_utility
            or self.n_steps >= self.max_steps
        ):
            self._done = True

        return self._get_obs(), self._get_info()

    def get_metrics(self) -> dict[str, Any]:
        return {
            "explored_rate": self.explored_rate,
            "travel_distance": self.travel_dist,
            "n_steps": self.n_steps,
            "success": self.explored_rate > self.success_threshold,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eval_explored_rate(self) -> None:
        self.explored_rate = float(
            np.sum(self.robot_belief == FREE) / max(self.total_free, 1)
        )

    def _get_frontiers(self) -> set:
        h, w = self.height, self.width
        belief = self.robot_belief
        unk = (belief == UNKNOWN).astype(np.int32)
        up = np.pad(unk, 1, constant_values=0)
        nb = (
            up[2:, 1:w+1] + up[:h, 1:w+1]
            + up[1:h+1, 2:] + up[1:h+1, :w]
            + up[:h, 2:] + up[2:, :w]
            + up[2:, 2:] + up[:h, :w]
        )
        mask = (belief == FREE) & (nb > 1) & (nb < 8)
        cells = np.argwhere(mask)
        if cells.shape[0] == 0:
            return set()
        coords = np.round(
            np.stack([
                cells[:, 1] * CELL_SIZE + self.belief_origin_x,
                cells[:, 0] * CELL_SIZE + self.belief_origin_y,
            ], axis=1),
            1,
        )
        if FRONTIER_CELL_SIZE != CELL_SIZE:
            return self._downsample_frontiers(coords)
        return set(map(tuple, coords.tolist()))

    def _downsample_frontiers(self, data: np.ndarray) -> set:
        vidx = (data / FRONTIER_CELL_SIZE).astype(int)
        vdict: dict = {}
        for i, pt in enumerate(data):
            k = (int(vidx[i, 0]), int(vidx[i, 1]))
            if k not in vdict:
                vdict[k] = pt
            else:
                center = np.array(k) * FRONTIER_CELL_SIZE
                if np.linalg.norm(pt - center) < np.linalg.norm(vdict[k] - center):
                    vdict[k] = pt
        return {(float(v[0]), float(v[1])) for v in vdict.values()}

    def _connected_free_map(self) -> np.ndarray:
        free = self.robot_belief == FREE
        # 8-connectivity: structure of ones
        labeled, _ = sk_label(free, structure=np.ones((3, 3), dtype=int))
        cx, cy = int(self.robot_cell[0]), int(self.robot_cell[1])
        if not (0 <= cy < self.height and 0 <= cx < self.width):
            return np.zeros_like(free)
        lbl = labeled[cy, cx]
        if lbl == 0:
            return np.zeros_like(free)
        return labeled == lbl

    def _update_graph(self) -> None:
        loc = self.robot_location
        frontiers = self._get_frontiers()

        new_frontiers = frontiers - self._prev_frontiers
        if new_frontiers:
            nf = np.array(list(new_frontiers))
            keep = np.linalg.norm(nf - loc, axis=1) <= self.sensor_range + FRONTIER_CELL_SIZE
            new_frontiers = set(map(tuple, nf[keep].tolist()))
        self._prev_frontiers = frontiers

        half = UPDATING_MAP_SIZE / 2
        ox, oy = self.belief_origin_x, self.belief_origin_y
        map_x_max = ox + (self.width - 1) * CELL_SIZE
        map_y_max = oy + (self.height - 1) * CELL_SIZE

        x_min = max(loc[0] - half, ox)
        x_max = min(loc[0] + half, map_x_max)
        y_min = max(loc[1] - half, oy)
        y_max = min(loc[1] + half, map_y_max)

        x_min = round(math.ceil(x_min / NODE_RESOLUTION) * NODE_RESOLUTION, 1)
        x_max = round(math.floor(x_max / NODE_RESOLUTION) * NODE_RESOLUTION, 1)
        y_min = round(math.ceil(y_min / NODE_RESOLUTION) * NODE_RESOLUTION, 1)
        y_max = round(math.floor(y_max / NODE_RESOLUTION) * NODE_RESOLUTION, 1)

        if x_min > x_max or y_min > y_max:
            return

        connected = self._connected_free_map()
        x_vals = np.round(np.arange(x_min, x_max + 1e-3, NODE_RESOLUTION), 1)
        y_vals = np.round(np.arange(y_min, y_max + 1e-3, NODE_RESOLUTION), 1)

        for xv in x_vals:
            for yv in y_vals:
                coords = np.array([xv, yv])
                cell = _coords_to_cell(coords, ox, oy)
                cx, cy = int(cell[0]), int(cell[1])
                if not (0 <= cy < self.height and 0 <= cx < self.width):
                    continue
                if not connected[cy, cx]:
                    continue

                key = (float(xv), float(yv))
                if key not in self._nodes:
                    self._nodes[key] = _Node(coords, frontiers, self)
                else:
                    node = self._nodes[key]
                    if not node.visited and (
                        node.utility > 0
                        or np.linalg.norm(coords - loc) <= 2 * self.sensor_range
                    ):
                        node.update_frontiers(new_frontiers, frontiers, self)

                node = self._nodes[key]
                if node.need_update_neighbor and (
                    np.linalg.norm(coords - loc) < self.sensor_range + NODE_RESOLUTION
                ):
                    node.update_neighbors(self._nodes, self)

    def _resolve_target(self, target: np.ndarray) -> np.ndarray:
        loc_key = (round(float(self.robot_location[0]), 1), round(float(self.robot_location[1]), 1))
        if loc_key in self._nodes:
            neighbors = self._nodes[loc_key].neighbor_set - {loc_key}
            if neighbors:
                nb_arr = np.array(list(neighbors))
                best = nb_arr[np.argmin(np.linalg.norm(nb_arr - target, axis=1))]
                return np.array([best[0], best[1]])
        # Fallback: nearest node overall
        if not self._nodes:
            return self.robot_location.copy()
        keys = np.array(list(self._nodes.keys()))
        best = keys[np.argmin(np.linalg.norm(keys - target, axis=1))]
        return np.array([best[0], best[1]])

    def _get_obs(self) -> dict[str, Any]:
        loc_key = (round(float(self.robot_location[0]), 1), round(float(self.robot_location[1]), 1))
        neighbors: list = []
        if loc_key in self._nodes:
            neighbors = [
                nb for nb in self._nodes[loc_key].neighbor_set if nb != loc_key
            ]
        nodes_info = [
            {
                "coords": k,
                "utility": n.utility,
                "visited": n.visited,
                "neighbors": [nb for nb in n.neighbor_set if nb != k],
            }
            for k, n in self._nodes.items()
        ]
        return {
            "global_map": self.robot_belief.copy(),
            "robot_pos": self.robot_location.astype(np.float32),
            "frontiers": list(self._get_frontiers()),
            "nodes": nodes_info,
            "neighbors": neighbors,
            "explored_rate": self.explored_rate,
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "done": self._done,
            "explored_rate": self.explored_rate,
            "travel_distance": self.travel_dist,
            "n_steps": self.n_steps,
            "success": self.explored_rate > self.success_threshold,
        }
