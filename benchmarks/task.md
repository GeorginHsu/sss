# FARBench Task Table

This table summarizes the 29 paper-release tasks, including metric direction, score range, GPU class, GPU count, and time budget.

| Task ID | Domain | Task Type | Metric | Direction | Score Range | GPU Class | Max GPUs | Budget (h) | Description |
|---|---|---|---|---|---|---|---:|---:|---|
| `screenspot_pro` | Computer Vision | Visual Grounding | Grounding score | higher | [0, 1] | 5090 | 2 | 10 | Locate target UI elements from natural-language instructions. |
| `ade20k` | Computer Vision | Semantic Segmentation | mIoU | higher | [0, 1] | 5090 | 4 | 10 | Predict dense semantic segmentation masks for scene images. |
| `cifar100lt` | Computer Vision | Long-Tailed Learning | Balanced acc. | higher | [0, 1] | 5090 | 1 | 3 | Learn image classification under a long-tailed class distribution. |
| `cifar100n` | Computer Vision | Noisy-Label Learning | Accuracy | higher | [0, 1] | 5090 | 1 | 5 | Learn image classification from noisy human annotations. |
| `crohme_hmer` | Computer Vision | Optical Character Recognition | ExpRate | higher | [0, 1] | 5090 | 1 | 6 | Convert handwritten formulas into structured LaTeX. |
| `domainnet_quickdraw` | Computer Vision | Domain Generalization | Accuracy | higher | [0, 1] | 5090 | 4 | 10 | Generalize recognition from source domains to sketch-like target images. |
| `objaverse_3dgen` | Computer Vision | 3D Generation | LPIPS | lower | [0, 1] | 6000 | 2 | 8 | Generate 3D-consistent object views from limited visual input. |
| `split_cifar100` | Computer Vision | Continual Learning | Avg. acc. | higher | [0, 1] | 5090 | 1 | 5 | Learn classes sequentially while retaining performance on earlier classes. |
| `terra_incognita` | Computer Vision | OOD Classification | Balanced acc. | higher | [0, 1] | 5090 | 1 | 3 | Generalize animal classification across unseen camera-trap domains. |
| `habitat3` | Robotics | Reinforcement Learning | Nav success | higher | [0, 1] | 6000 | 4 | 10 | Navigate embodied environments to find or follow target agents. |
| `humanoidbench` | Robotics | Humanoid Control | Success rate | higher | [0, 1] | 6000 | 1 | 10 | Train whole-body humanoid policies for locomotion and manipulation. |
| `vlabench_manipulation` | Robotics | Manipulation Learning | Success rate | higher | [0, 1] | 5090 | 4 | 10 | Learn vision-language-action policies for robotic manipulation. |
| `cogniplan` | Robotics | Autonomous Exploration | Exploration score | higher | [0, 62] | 6000 | 1 | 4 | Explore unknown environments and maximize spatial coverage. |
| `minigrid` | Robotics | Navigation | Success rate | higher | [0, 1] | 5090 | 1 | 10 | Learn navigation and puzzle-solving policies in grid-world environments. |
| `climsim_lowres` | AI for Science | Climate Modeling | R2 | higher | [-1, 1] | 5090 | 1 | 4 | Emulate atmospheric physical tendencies from low-resolution climate states. |
| `etth1_forecasting` | AI for Science | Time-Series Forecasting | MSE | lower | [0, 2] | 5090 | 1 | 1 | Forecast long-horizon multivariate time-series dynamics. |
| `flip_aav` | AI for Science | Protein Prediction | Spearman rho | higher | [-1, 1] | 5090 | 1 | 5 | Predict protein variant fitness under extrapolative mutation settings. |
| `metrla_traffic` | AI for Science | Traffic Forecasting | MAE | lower | [0, 8] | 5090 | 1 | 3 | Forecast future traffic speed over a road-sensor graph. |
| `ogbg_molpcba` | AI for Science | Molecular Classification | Avg. precision | higher | [0, 1] | 5090 | 1 | 6 | Predict molecular activity labels from graph-structured molecules. |
| `qm9` | AI for Science | Molecular Property Prediction | MAE | lower | [0, 2] | 5090 | 1 | 2 | Predict molecular properties from molecular structure. |
| `weatherbench_z500t850` | AI for Science | Weather Forecasting | RMSE | lower | [0, 3133] | 5090 | 1 | 5 | Forecast atmospheric variables over global weather fields. |
| `wilds_fmow` | AI for Science | Remote Sensing | Worst-region acc. | higher | [0, 1] | 6000 | 4 | 10 | Classify satellite images across regions and time. |
| `iwildcam_wilds` | AI for Science | Biodiversity Monitoring | Macro-F1 | higher | [0, 1] | 6000 | 4 | 8 | Classify camera-trap wildlife under location shift. |
| `asvspoof2021_la` | Audio/Speech | Deepfake Detection | EER | lower | [0, 1] | 5090 | 1 | 8 | Detect spoofed speech under challenging acoustic conditions. |
| `voicebank_demand` | Audio/Speech | Speech Enhancement | PESQ | higher | [-0.5, 4.5] | 5090 | 1 | 6 | Denoise noisy speech and recover clean speech waveforms. |
| `aime_math_rl` | NLP | Math Reasoning | Exact match | higher | [0, 1] | 6000 | 4 | 20 | Improve mathematical reasoning through reinforcement learning. |
| `assist2009_kt` | NLP | Knowledge Tracing | AUC-ROC | higher | [0, 1] | 5090 | 1 | 0.5 | Predict response correctness from historical learning interactions. |
| `bigcodebench_codegen` | NLP | Code Generation | Pass@1 | higher | [0, 1] | 6000 | 4 | 10 | Generate executable code solutions for programming tasks. |
| `qlib_stock` | NLP | Financial Forecasting | IC mean | higher | [-1, 1] | 6000 | 1 | 4 | Predict future stock returns from historical market features. |
