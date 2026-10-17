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
┌────────────────────────────────────────────────────────────────────────┐
│                        Full Research Pipeline                          │
│                                                                        │
│  highway-v0 ──► PPO Training ──► Rollout Collection (40 episodes)     │
│  (gymnasium)    (SB3, 30K steps)        │                              │
│                                ┌────────┴────────┐                    │
│                                ▼                 ▼                    │
│                           Train split        Test split               │
│                           (80%)              (20%, held-out)          │
│                                │                 │                    │
│                                ▼                 │                    │
│                       Decision Tree Fit          │                    │
│                       (max_depth ∈ {1…8})        │                    │
│                                └────────┬────────┘                    │
│                                         ▼                              │
│                          ┌──────────────┼──────────────┐              │
│                          ▼              ▼              ▼              │
│                     Depth Sweep    ego_y          Driving             │
│                     (fidelity vs   Feature        Metrics +           │
│                      complexity)   Ablation       Dist. Shift        │
└────────────────────────────────────────────────────────────────────────┘
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

### Results

### Phase 1 — Baseline

| Metric | Value |
|--------|-------|
| PPO Training Timesteps | 30,000 |
| Mean Evaluation Reward (10 eps) | **26.38 ± 8.09** |
| Decision Tree Depth | 4 |
| Decision Tree Leaves | 8 |
| Fidelity (same-data, Phase 1) | 99.23% |

> **Note:** The Phase 1 tree was fitted on a 20-episode rollout (~1,572 transitions) and measured on the **same** data used for training (in-sample). Phase 2 uses a separate 40-episode rollout split 80/20 — a different dataset — which produces a slightly different tree structure (12 leaves at depth 4 instead of 8). The Phase 2 train fidelity (98.81%) is therefore not directly comparable to the Phase 1 figure (99.23%); both are correct for their respective datasets.

### Phase 2 — Research Experiments

**Experiment 1: Honest train/test fidelity**

| Split | Transitions | Fidelity |
|-------|------------|---------|
| Training (80%) | 1,257 | 98.81% |
| **Test (20%, held-out)** | **315** | **97.78%** |
| Generalisation gap | — | **1.03 pp** ✅ minimal overfitting |

**Experiment 2: Depth vs. fidelity sweep**

| Depth | Leaves | Train % | Test % |
|-------|--------|---------|--------|
| 1 | 2 | 96.66 | 95.87 |
| **2** | **4** | **98.25** | **97.78** |
| 3 | 8 | 98.73 | 98.10 |
| 4 ◄ | 12 | 98.81 | 97.78 |
| 5 | 14 | 98.81 | 97.78 |
| 6 | 16 | 98.81 | 98.10 |
| 8 | 17 | 98.81 | 98.10 |

> **Key insight:** Depth 2 (4 leaves = 4 rules) achieves identical test fidelity (97.78%) to depth 4, 5, and 6. The PPO policy on this task is representable by **4 decision rules**. Adding complexity beyond depth 2 yields no measurable improvement.

**Experiment 3: `ego_y` feature ablation**

| Condition | Features | Test Fidelity | Change |
|-----------|---------|--------------|--------|
| Baseline | All 25 | 97.78% | — |
| Without `ego_y` | 24 | 97.78% | **0.00 pp** |
| `ego_y` only | 1 | 95.87% | −1.91 pp |

> **Revised finding:** The Phase 1 result reporting 97.6% Gini importance for `ego_y` was a **tree artefact**. Removing `ego_y` entirely causes zero fidelity drop, meaning the policy boundary is equally representable without it. Simultaneously, `ego_y` alone achieves 95.87%, suggesting multiple features span a similar decision boundary. The policy is not uniquely dependent on any single kinematic feature.

**Experiment 4: Driving behaviour metrics (30 episodes)**

| Metric | Value |
|--------|-------|
| Collision rate | 26.7% |
| **Success rate** | **73.3%** |
| Mean speed | 21.08 ± 1.71 m/s |
| Mean episode length | 33.4 ± 11.4 steps |
| Mean lane changes / episode | 8.63 |

> The agent adopted an aggressive speed-maximising strategy (21.08 m/s, 8.63 lane changes/ep), consistent with the `FASTER`-heavy action distribution observed in Phase 1. The 26.7% collision rate reflects the 30,000-step training budget; further training would be expected to reduce this.

**Experiment 5: Generalisation under distribution shift**

| Traffic Condition | Vehicles | Samples | Surrogate Fidelity |
|------------------|----------|---------|-------------------|
| Sparse | 10 | 584 | 97.26% |
| Medium (training density) | 20 | 568 | 98.59% |
| **Dense** | **30** | **500** | **94.60%** |

> The surrogate retains high fidelity under distribution shift, with only a 4pp drop in dense traffic. This suggests the Decision Tree explanation is robust across the tested range of traffic densities.

### Key XAI Finding — Corrected Interpretation

The Phase 1 analysis identified `ego_y` as overwhelmingly dominant (97.6% Gini importance). The Phase 2 ablation **revises this**: `ego_y` was an arbitrary choice by the tree, not a causally necessary feature. The deeper finding is that the PPO policy on `highway-v0` at this training budget is sufficiently simple that **a depth-2 tree with 4 leaves captures it with 97.78% fidelity** — regardless of which specific kinematic features are used as splits.

| Rank | Feature | Gini Importance | Ablation Verdict |
|------|---------|----------------|-----------------|
| 1 | `ego_y` | 0.9760 | **Artefact** — removable at 0.00 pp cost |
| 2 | `v2_vx` | 0.0143 | Minor contributor |
| 3 | `v1_y` | 0.0055 | Minor contributor |
| 4 | `v3_vx` | 0.0021 | Negligible |
| 5 | `v4_vx` | 0.0011 | Negligible |

---

## Repository Structure

```
rl-xai-highway/
├── rl_xai_highway.py        # Main script — all 18 Colab cells (Phase 1: 1–12, Phase 2: 13–18)
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

After training, the PPO agent is rolled out for **20 episodes (Phase 1)**. Each `(observation, action)` pair is collected into a supervised dataset. A `DecisionTreeClassifier(max_depth=4)` is then fitted on this dataset — a **surrogate model** (also called a student or mimic model) that approximates the neural policy with human-readable if-else logic.

**Phase 2** extends this to **40 rollout episodes**, split 80/20 into training and held-out test sets, enabling an honest out-of-sample fidelity measurement (see Experiment 1).

**Fidelity** measures the fraction of timesteps where the tree's predicted action matches the PPO agent's action. At 99.23%, the Phase 1 depth-4 tree almost perfectly captures the PPO decision surface over the observed state distribution (measured on the same data used for training). The honest out-of-sample test fidelity in Phase 2 is **97.78%**.

---

## Outputs

Running the full script (Cells 1–18) produces:

**Phase 1:**
- **`ppo_highway_agent.zip`** — Saved PPO weights (reload with `PPO.load()`)
- **`ppo_highway_agent.mp4`** — Video of the trained agent navigating traffic
- **`decision_tree_policy.png`** — Visual decision tree coloured by action class
- **`feature_importance.png`** — Gini-reduction feature importance bar chart

**Phase 2:**
- **`depth_fidelity_tradeoff.png`** — Fidelity vs. tree complexity curve
- **`ego_y_ablation.png`** — Ablation bar chart (baseline / no-ego_y / ego_y-only)
- **`driving_behaviour_metrics.png`** — Distributions of speed, length, lane changes
- **`generalisation_distribution_shift.png`** — Fidelity across 3 traffic densities

---

## Limitations and Future Work

- **Distribution shift:** The Decision Tree is only faithful on states the PPO agent actually visited. Out-of-distribution states may be misrepresented.
- **Depth trade-off:** The depth sweep (Experiment 2) shows that `max_depth=2` is the optimal point — it achieves the same test fidelity (97.78%) as deeper trees while using only 4 rules. The Phase 1 baseline used `max_depth=4`; this was revised in Phase 2.
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
