# Interpretable Autonomous Driving via Reinforcement Learning and Policy Distillation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange)](https://stable-baselines3.readthedocs.io/)
[![highway-env](https://img.shields.io/badge/Environment-highway--v0-green)](https://highway-env.farama.org/)
[![scikit-learn](https://img.shields.io/badge/XAI-Decision%20Tree-red?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yuvashakthiraj/rl-xai-highway/blob/main/rl_xai_highway.py)

---

A research-oriented project that trains a **Proximal Policy Optimization (PPO)** agent to navigate a multi-lane highway simulation and then applies **Explainable AI (XAI)** techniques to distill the learned neural policy into a human-readable Decision Tree — bridging the gap between black-box deep reinforcement learning and interpretable autonomous driving.

---

## Motivation

Deep RL agents can outperform hand-crafted controllers at complex driving tasks, but their decision-making is inherently opaque. In safety-critical systems like autonomous vehicles, *why* an agent takes an action is as important as *what* action it takes. This project demonstrates **post-hoc policy distillation** as a practical XAI technique: a surrogate Decision Tree is trained to mimic the PPO policy, producing auditable if-then-else rules that domain experts can inspect, validate, and challenge.

---

## Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Pipeline Overview                        │
│                                                                 │
│  highway-v0 ──► PPO Training ──► Rollout Collection            │
│  (gymnasium)    (SB3, 30K steps)  (20 episodes, 777 steps)     │
│                                         │                       │
│                                         ▼                       │
│                              Decision Tree Distillation         │
│                              (scikit-learn, max_depth=4)        │
│                                         │                       │
│                          ┌──────────────┼──────────────┐        │
│                          ▼              ▼              ▼        │
│                     Text Rules    Tree Diagram   Feature        │
│                    (export_text)  (plot_tree)   Importance      │
└─────────────────────────────────────────────────────────────────┘
```

### Observation Space — Kinematics (5 × 5 = 25 features)

Each observation is a normalised kinematic snapshot of the ego vehicle and its 4 nearest neighbours:

| Vehicle | Features |
|---------|----------|
| `ego` | `presence, x, y, vx, vy` |
| `v1` (nearest) | `presence, x, y, vx, vy` |
| `v2` | `presence, x, y, vx, vy` |
| `v3` | `presence, x, y, vx, vy` |
| `v4` | `presence, x, y, vx, vy` |

All positions and velocities are expressed **relative to the ego vehicle** and **normalised to \[-1, 1\]**.

### Action Space — DiscreteMetaAction

| ID | Action | Description |
|----|--------|-------------|
| 0 | `LANE_LEFT` | Change to left lane |
| 1 | `IDLE` | Maintain lane and speed |
| 2 | `LANE_RIGHT` | Change to right lane |
| 3 | `FASTER` | Increase target speed |
| 4 | `SLOWER` | Decrease target speed |

---

## Results

| Metric | Value |
|--------|-------|
| PPO Training Timesteps | 30,000 |
| Mean Evaluation Reward (10 eps) | **26.38 ± 8.09** |
| Rollout Transitions Collected | 777 (20 episodes) |
| Decision Tree Depth | 4 |
| Decision Tree Leaves | 8 |
| **Fidelity to PPO Policy** | **99.23%** |

### Key XAI Finding — Feature Importance

The surrogate Decision Tree identified **`ego_y` (lateral lane position)** as overwhelmingly the most influential feature (97.6% of total Gini reduction), confirming that the PPO agent's decisions are dominated by its current lane position — a semantically meaningful and behaviourally valid insight.

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `ego_y` (ego lateral position) | 0.9760 |
| 2 | `v2_vx` (2nd vehicle relative speed) | 0.0143 |
| 3 | `v1_y` (nearest vehicle lateral position) | 0.0055 |
| 4 | `v3_vx` | 0.0021 |
| 5 | `v4_vx` | 0.0011 |

---

## Repository Structure

```
rl-xai-highway/
├── rl_xai_highway.py        # Main script — all 12 Colab cells
├── requirements.txt         # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quickstart

### Run on Google Colab (recommended)

Click the badge at the top of this README, or upload `rl_xai_highway.py` to a new Colab notebook and run cells sequentially.

### Run locally

```bash
git clone https://github.com/Yuvashakthiraj/rl-xai-highway.git
cd rl-xai-highway

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python rl_xai_highway.py
```

> **Note:** A GPU is not required. The MLP policy and Decision Tree are CPU-bound. Expected runtime on Colab CPU: ~8–12 minutes.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `highway-env` | Multi-lane highway driving simulation (Farama Gymnasium) |
| `stable-baselines3[extra]` | PPO implementation with vectorised environments |
| `gymnasium` | RL environment interface |
| `scikit-learn` | Decision Tree training, evaluation, and visualisation |
| `matplotlib` | Tree diagram and feature importance plots |
| `imageio` + `imageio-ffmpeg` | Frame-by-frame video encoding |
| `moviepy` | Video post-processing |

---

## Methodology

### 1. Reinforcement Learning — PPO

PPO is an on-policy actor-critic algorithm that optimises the clipped surrogate objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1 - \varepsilon, 1 + \varepsilon)\hat{A}_t \right) \right]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ and $\varepsilon = 0.2$. The clipping prevents catastrophically large policy updates, making PPO stable across a wide range of hyperparameters.

### 2. XAI — Policy Distillation

After training, the PPO agent is rolled out for 20 episodes. Each `(observation, action)` pair is collected into a supervised dataset. A `DecisionTreeClassifier(max_depth=4)` is then fitted on this dataset — a **surrogate model** (also called a student or mimic model) that approximates the neural policy with human-readable if-else logic.

**Fidelity** measures the fraction of timesteps where the tree's predicted action matches the PPO agent's action. At 99.23%, the depth-4 tree almost perfectly captures the PPO decision surface over the observed state distribution.

---

## Outputs

Running the full script produces:

- **`ppo_highway_agent.zip`** — Saved PPO weights (reload with `PPO.load()`)
- **`ppo_highway_agent.mp4`** — Video of the trained agent navigating traffic
- **`decision_tree_policy.png`** — Visual decision tree coloured by action class
- **`feature_importance.png`** — Gini-reduction feature importance bar chart

---

## Limitations and Future Work

- **Distribution shift:** The Decision Tree is only faithful on states the PPO agent actually visited. Out-of-distribution states may be misrepresented.
- **Depth trade-off:** `max_depth=4` prioritises interpretability over completeness. A deeper tree would improve fidelity but reduce readability.
- **Global vs. local explanations:** Future work could apply LIME or SHAP for per-decision local explanations rather than a single global surrogate.
- **Richer environments:** Extending to `intersection-v0` or `roundabout-v0` would test whether the XAI findings generalise across traffic topologies.
- **Reward shaping:** Incorporating collision-avoidance rewards and comfort penalties would make the agent — and its explanation — more realistic.

---

## References

1. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
2. Leurent, E. (2018). *An Environment for Autonomous Driving Decision-Making.* [highway-env](https://github.com/Farama-Foundation/HighwayEnv)
3. Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations.* JMLR 22(268):1−8
4. Bastani, O. et al. (2018). *Verifiable Reinforcement Learning via Policy Extraction.* NeurIPS 2018

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
