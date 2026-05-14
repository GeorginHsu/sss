"""Shared constants for prompt construction and history truncation.

Two-stage truncation flow:
  Raw output → Orchestrator storage (STDOUT_HISTORY_TAIL / STDERR_HISTORY_TAIL)
             → Prompt construction  (STDOUT_PROMPT_TAIL  / STDERR_PROMPT_TAIL)
Full command output is always saved to disk (command_output.json).
"""

# ── History storage (orchestrator.py) ────────────────────────────
# How much stdout/stderr is kept in the in-memory IterationRecord.
STDOUT_HISTORY_TAIL: int = 500       # chars of stdout kept in IterationRecord
STDERR_HISTORY_TAIL: int = 300       # chars of stderr kept on command failure
STDERR_ERROR_TAIL: int = 300         # chars of stderr in error messages

# ── Prompt construction (agent_prompt.py) ────────────────────────
# How much of each field appears in the LLM prompt.
HISTORY_COMPACT_MAX: int = 20        # max history entries sent to agent
HISTORY_RECENT_COUNT: int = 3        # entries with full detail (stdout, stderr)
DETAIL_OUTPUT_TAIL: int = 300        # stdout chars in history (recent entries)
DETAIL_EVAL_ERROR_TAIL: int = 500    # eval_error chars in history (recent entries)
STDOUT_PROMPT_TAIL: int = 2000       # stdout chars in current command_output
STDERR_PROMPT_TAIL: int = 1000       # stderr chars in current command_output
MAX_WORKSPACE_CONTENT_SIZE: int = 50_000  # total workspace content chars before truncation

# ── Timeouts ─────────────────────────────────────────────────────
PIP_INSTALL_TIMEOUT: int = 300       # seconds for pip install commands
