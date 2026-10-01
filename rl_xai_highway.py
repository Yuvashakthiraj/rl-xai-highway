# RL + XAI Project: PPO on highway-env + Decision Tree Policy Distillation
# ─────────────────────────────────────────────────────────────────────────────
# Google Colab notebook — run cells top-to-bottom.

# %% [markdown]
"""
# 🚗 RL + XAI: Interpretable Autonomous Driving via Policy Distillation

**Pipeline Overview:**
1. Install dependencies
2. Train a PPO agent on `highway-v0`
3. Collect rollout data from the trained agent
4. Distill the PPO policy into a shallow Decision Tree
5. Visualise the tree rules (text + graph + feature importance)
6. Render a video of the PPO agent navigating traffic
"""


# %% [markdown]
"""
## Cell 1 — Install Dependencies
"""

# %%
# Run this once. Restart runtime if prompted.
!pip install highway-env stable-baselines3[extra] gymnasium scikit-learn \
             moviepy imageio imageio-ffmpeg --quiet


# %% [markdown]
"""
## Cell 2 — Imports & Reproducibility Seed
"""

# %%
import os
import warnings
import numpy as np
import gymnasium as gym
import highway_env                          # registers highway-v0 with gymnasium
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ── Video rendering ──────────────────────────────────────────────────────────
import imageio
from IPython.display import HTML, display
from base64 import b64encode

warnings.filterwarnings("ignore")

# Global seed — guarantees reproducibility across NumPy, SB3, and sklearn
SEED = 42
np.random.seed(SEED)

print("✅ All imports successful.")
print(f"   gymnasium   : {gym.__version__}")
print(f"   highway_env : {highway_env.__version__}")


# %% [markdown]
"""
## Cell 3 — Environment Configuration

### Observation: Kinematics (5 vehicles × 5 features)

| Index | Vehicle | Features |
|-------|---------|----------|
| 0 | Ego (agent) | presence, x, y, vx, vy |
| 1–4 | Nearest neighbours | presence, x, y, vx, vy |

All positions/velocities are **relative to the ego vehicle** and **normalised** to [-1, 1].

### Actions: DiscreteMetaAction
| ID | Name | Description |
|----|------|-------------|
| 0 | LANE_LEFT | Move one lane to the left |
| 1 | IDLE | Maintain current lane and speed |
| 2 | LANE_RIGHT | Move one lane to the right |
| 3 | FASTER | Increase target speed |
| 4 | SLOWER | Decrease target speed |
"""

# %%
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,          # ego + 4 surrounding vehicles
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,            # positions relative to ego vehicle
        "normalize": True,            # normalise to [-1, 1]
        "order": "sorted",            # nearest vehicles first
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,                 # 4-lane highway
    "vehicles_count": 20,             # 20 surrounding NPCs
    "duration": 40,                   # episode length in seconds
    "initial_spacing": 2,
    "collision_reward": -1.0,         # penalty for collisions
    "reward_speed_range": [20, 30],   # reward band for maintaining speed (m/s)
    "simulation_frequency": 15,       # Hz
    "policy_frequency": 1,            # decisions per second
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

# Human-readable feature names for the decision tree (25 total)
# Format: {vehicle_label}_{feature} where v0 = ego, v1–v4 = neighbours
FEATURE_NAMES = []
for v_idx in range(5):
    prefix = "ego" if v_idx == 0 else f"v{v_idx}"
    for feat in ["presence", "x", "y", "vx", "vy"]:
        FEATURE_NAMES.append(f"{prefix}_{feat}")

# Map integer action IDs to human-readable strings
ACTION_LABELS = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

print("✅ Environment configuration set.")
print(f"   Observation : (5 vehicles × 5 features) → flattened to 25-dim vector")
print(f"   Feature names : {FEATURE_NAMES}")
print(f"   Actions       : {ACTION_LABELS}")


# %% [markdown]
"""
## Cell 4 — Train the PPO Agent

**PPO (Proximal Policy Optimization)** alternates between:
1. Collecting on-policy rollouts from the current policy π
2. Optimising the clipped surrogate objective to update π without catastrophic policy shifts

$$L^{CLIP}(\\theta) = \\mathbb{E}_t\\left[\\min\\left(r_t(\\theta)\\hat{A}_t,\\; \\text{clip}(r_t(\\theta), 1-\\varepsilon, 1+\\varepsilon)\\hat{A}_t\\right)\\right]$$

where $r_t(\\theta) = \\frac{\\pi_\\theta(a_t|s_t)}{\\pi_{\\theta_{old}}(a_t|s_t)}$ and $\\varepsilon=0.2$.
"""

# %%
TOTAL_TIMESTEPS = 30_000   # sufficient for a competent highway policy

# Vectorised environment — 4 parallel copies speeds up data collection
n_envs = 4
vec_env = make_vec_env(
    "highway-v0",
    n_envs=n_envs,
    seed=SEED,
    env_kwargs={"config": ENV_CONFIG},
)

# ── PPO Model ──────────────────────────────────────────────────────────────
# MlpPolicy: fully-connected MLP shared between actor (policy) and critic (value fn)
# Default architecture: Linear(25→64) → Tanh → Linear(64→64) → Tanh →
#                       actor Linear(64→5) + critic Linear(64→1)
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    learning_rate=3e-4,           # Adam optimiser step size
    n_steps=256,                  # rollout steps per env before each update
    batch_size=64,                # minibatch size for stochastic gradient updates
    n_epochs=10,                  # optimisation epochs per rollout collection
    gamma=0.99,                   # discount factor γ
    gae_lambda=0.95,              # GAE λ — bias-variance trade-off in advantage
    clip_range=0.2,               # PPO clip ε
    ent_coef=0.01,                # entropy bonus — prevents premature convergence
    vf_coef=0.5,                  # value function loss weight
    max_grad_norm=0.5,            # gradient clipping norm
    verbose=1,
    seed=SEED,
    tensorboard_log="./ppo_highway_tb/",
)

print(f"🚀 Training PPO for {TOTAL_TIMESTEPS:,} timesteps across {n_envs} parallel envs...")
model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)

# Persist trained weights
model.save("ppo_highway_agent")
print("✅ Training complete. Model saved as 'ppo_highway_agent.zip'.")


# %% [markdown]
"""
## Cell 5 — Evaluate the PPO Agent
"""

# %%
# Run 10 deterministic evaluation episodes (no exploration noise)
eval_env = gym.make("highway-v0", config=ENV_CONFIG)

mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

print(f"📊 PPO Evaluation Results (10 episodes):")
print(f"   Mean reward : {mean_reward:.2f} ± {std_reward:.2f}")
eval_env.close()


# %% [markdown]
"""
## Cell 6 — Collect Rollout Dataset for Policy Distillation

We run the **deterministic** PPO policy for 20 full episodes and record every
`(flattened_observation, action)` pair.  This supervised dataset is then used
to train a decision tree that *imitates* the PPO policy.

This is **knowledge distillation**: the neural network acts as the *teacher*;
the decision tree is the *student*.
"""

# %%
N_EVAL_EPISODES = 20   # number of PPO episodes to record

rollout_observations = []  # list of 25-d numpy arrays
rollout_actions = []       # list of integer actions

record_env = gym.make("highway-v0", config=ENV_CONFIG)
obs, info = record_env.reset(seed=SEED)

episodes_done = 0
total_steps = 0

print(f"🎲 Collecting rollout data over {N_EVAL_EPISODES} PPO episodes...")

while episodes_done < N_EVAL_EPISODES:
    # Flatten the (5, 5) kinematics observation into a 25-d vector
    flat_obs = obs.flatten()

    # Deterministic action from PPO — argmax over policy logits
    action, _ = model.predict(obs, deterministic=True)

    rollout_observations.append(flat_obs)
    rollout_actions.append(int(action))

    obs, reward, terminated, truncated, info = record_env.step(action)
    total_steps += 1

    if terminated or truncated:
        episodes_done += 1
        obs, info = record_env.reset()

record_env.close()

# Materialise as numpy arrays
X_rollout = np.array(rollout_observations)   # shape: (N_steps, 25)
y_rollout = np.array(rollout_actions)         # shape: (N_steps,)

print(f"✅ Dataset collected.")
print(f"   Total timesteps : {total_steps:,}")
print(f"   X shape         : {X_rollout.shape}  (timesteps × features)")
print(f"   y shape         : {y_rollout.shape}")
print(f"\n   Action distribution in collected data:")
unique, counts = np.unique(y_rollout, return_counts=True)
for a, c in zip(unique, counts):
    bar = "█" * int(50 * c / len(y_rollout))
    print(f"     {ACTION_LABELS[a]:12s} ({a}): {c:4d} ({100*c/len(y_rollout):5.1f}%)  {bar}")


# %% [markdown]
"""
## Cell 7 — Train the Decision Tree (Policy Distillation)

A `DecisionTreeClassifier(max_depth=4)` is fitted on the rollout dataset.
`max_depth=4` limits the tree to **16 leaves** maximum — shallow enough
that a human can read every branch, yet expressive enough to capture the
dominant structure of the PPO policy.

**Fidelity** = fraction of steps where the tree's action matches PPO's action.
"""

# %%
dt_model = DecisionTreeClassifier(
    max_depth=4,          # at most 4 decision levels — keeps the tree legible
    criterion="gini",     # Gini impurity as the split criterion
    min_samples_split=10, # require ≥10 samples to split a node
    min_samples_leaf=5,   # require ≥5 samples in each leaf
    random_state=SEED,
)

dt_model.fit(X_rollout, y_rollout)

# Measure fidelity: how often does the tree agree with PPO?
y_pred = dt_model.predict(X_rollout)
fidelity = accuracy_score(y_rollout, y_pred)

target_names = [ACTION_LABELS[i] for i in sorted(ACTION_LABELS.keys())]

print(f"🌳 Decision Tree trained.")
print(f"   Tree depth (actual) : {dt_model.get_depth()}")
print(f"   Number of leaves    : {dt_model.get_n_leaves()}")
print(f"   Fidelity to PPO     : {fidelity * 100:.2f}%\n")

print("📋 Per-class fidelity report:")
# Pass `labels` explicitly so classification_report always evaluates all 5
# action classes — even if the PPO rollout only used a subset of them.
print(classification_report(
    y_rollout, y_pred,
    labels=list(ACTION_LABELS.keys()),   # force evaluation over all 5 action IDs
    target_names=target_names,
    zero_division=0,                      # suppress warnings for zero-sample classes
))


# %% [markdown]
"""
## Cell 8 — Export Decision Tree Rules as Text

`export_text()` walks the fitted tree and prints every split as a human-readable
**if-then-else** condition using the semantic feature names we defined
(`ego_x`, `v1_vx`, etc.).

Each leaf shows the **class weights** — the distribution of PPO actions in that region of state space.
"""

# %%
print("=" * 72)
print("🌳 DECISION TREE POLICY RULES — Plain-text Export")
print("=" * 72)
print("Feature legend:")
print("  ego_*       = ego vehicle (the controlled agent)")
print("  v1, v2, ... = surrounding vehicles sorted by proximity (nearest = v1)")
print("  x, y        = relative longitudinal/lateral position (normalised)")
print("  vx, vy      = relative longitudinal/lateral velocity (normalised)")
print("  presence    = 1.0 if vehicle is detected; 0.0 otherwise")
print("=" * 72, "\n")

tree_rules = export_text(
    dt_model,
    feature_names=FEATURE_NAMES,
    max_depth=4,
    spacing=3,
    decimals=3,
    show_weights=True,
)
print(tree_rules)


# %% [markdown]
"""
## Cell 9 — Visualise the Decision Tree Graphically

Each internal node shows the **split condition**, **Gini impurity**, **sample count**,
and **majority class**.  Nodes are colour-coded by majority predicted action.
"""

# %%
fig, ax = plt.subplots(figsize=(28, 10))

plot_tree(
    dt_model,
    feature_names=FEATURE_NAMES,
    class_names=target_names,
    filled=True,          # colour nodes by majority class
    rounded=True,         # aesthetic rounded rectangles
    fontsize=9,
    ax=ax,
    proportion=False,     # show raw sample counts (not fractions)
    impurity=True,        # display Gini impurity in each node
    precision=3,
)

ax.set_title(
    "Distilled PPO Policy — Decision Tree Surrogate (max_depth = 4)\n"
    f"Trained on {N_EVAL_EPISODES} PPO evaluation episodes | highway-v0",
    fontsize=14, fontweight="bold", pad=16,
)

# Legend patches (approximate colours matched to matplotlib's filled tree palette)
from matplotlib import colormaps
cmap = colormaps.get_cmap("tab10")
legend_patches = [
    mpatches.Patch(color=cmap(i / 5), label=f"{i}: {ACTION_LABELS[i]}")
    for i in range(5)
]
ax.legend(handles=legend_patches, loc="upper right", fontsize=9,
          title="Action Classes", framealpha=0.9)

plt.tight_layout()
plt.savefig("decision_tree_policy.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved 'decision_tree_policy.png'.")


# %% [markdown]
"""
## Cell 10 — Feature Importance Analysis

Decision Tree feature importance = total **Gini reduction** attributed to
splits on each feature across all nodes, weighted by sample count.

A high importance means the tree (and by extension, the PPO policy) uses that
kinematic feature heavily when choosing actions.
"""

# %%
importances = dt_model.feature_importances_       # shape: (25,)
sorted_idx = np.argsort(importances)[::-1]         # descending

# Keep only features that appear in at least one split
nonzero_mask = importances[sorted_idx] > 0
top_features  = np.array(FEATURE_NAMES)[sorted_idx][nonzero_mask]
top_importances = importances[sorted_idx][nonzero_mask]

# ── Bar chart ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, max(4, len(top_features) * 0.45)))

norm = plt.Normalize(vmin=0, vmax=top_importances.max())
colors = plt.cm.RdYlGn(norm(top_importances[::-1]))

bars = ax.barh(top_features[::-1], top_importances[::-1], color=colors, edgecolor="grey", linewidth=0.5)

ax.set_xlabel("Feature Importance (Gini Reduction)", fontsize=11)
ax.set_title(
    "Decision Tree Feature Importances\n(Surrogate of PPO Policy | highway-v0)",
    fontsize=12, fontweight="bold",
)

for bar, val in zip(bars, top_importances[::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=8)

ax.set_xlim(0, top_importances.max() * 1.18)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved 'feature_importance.png'.")

print("\n🔍 Top features used by the surrogate policy:")
for rank, (feat, imp) in enumerate(zip(top_features[:8], top_importances[:8]), 1):
    bar = "█" * int(imp / top_importances.max() * 30)
    print(f"   {rank}. {feat:20s}  {imp:.4f}  {bar}")


# %% [markdown]
"""
## Cell 11 — Render a Video of the PPO Agent

We record one full episode (highway rendering at 15 fps) and embed
the `.mp4` directly in Colab for immediate playback.
"""

# %%
VIDEO_PATH = "ppo_highway_agent.mp4"
VIDEO_FPS  = 15   # matches simulation_frequency

# Environment with rgb_array rendering enabled
video_env = gym.make("highway-v0", config=ENV_CONFIG, render_mode="rgb_array")
obs, info = video_env.reset(seed=SEED + 99)

frames = []
episode_reward = 0.0
step_count = 0

print("🎬 Recording one PPO episode...")

while True:
    frame = video_env.render()    # returns H×W×3 uint8 numpy array
    frames.append(frame)

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = video_env.step(action)
    episode_reward += reward
    step_count += 1

    if terminated or truncated:
        frames.append(video_env.render())  # capture terminal frame
        break

video_env.close()
print(f"   Captured {len(frames)} frames | {step_count} steps | reward = {episode_reward:.2f}")

# ── Encode to .mp4 ─────────────────────────────────────────────────────────
writer = imageio.get_writer(VIDEO_PATH, fps=VIDEO_FPS, codec="libx264", quality=8)
for frame in frames:
    writer.append_data(frame)
writer.close()
print(f"✅ Video saved as '{VIDEO_PATH}'.")

# ── Inline playback in Colab ───────────────────────────────────────────────
def show_video_colab(path: str) -> None:
    """Embed and display an .mp4 file inline in Google Colab."""
    with open(path, "rb") as f:
        video_b64 = b64encode(f.read()).decode("ascii")
    display(HTML(f"""
    <video width="720" height="400" controls autoplay loop muted
           style="border:1px solid #ccc; border-radius:8px; margin:8px 0;">
      <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """))

show_video_colab(VIDEO_PATH)


# %% [markdown]
"""
## Cell 12 — Summary Dashboard
"""

# %%
print("=" * 72)
print("📊 EXPERIMENT SUMMARY — RL + XAI | highway-v0")
print("   PPO Agent (Stable-Baselines3) + Decision Tree (scikit-learn)")
print("=" * 72)

# ── Agent performance ───────────────────────────────────────────────────────
print(f"\n🤖  PPO Agent")
print(f"    Policy          : MlpPolicy (MLP actor-critic)")
print(f"    Training steps  : {TOTAL_TIMESTEPS:,}")
print(f"    Parallel envs   : {n_envs}")
print(f"    Eval reward     : {mean_reward:.2f} ± {std_reward:.2f} (10 episodes)")

# ── Dataset ─────────────────────────────────────────────────────────────────
print(f"\n📦  Rollout Dataset")
print(f"    Episodes        : {N_EVAL_EPISODES}")
print(f"    Transitions     : {len(X_rollout):,}")
print(f"    Feature dims    : {X_rollout.shape[1]} (5 vehicles × 5 features)")

# ── Surrogate tree ──────────────────────────────────────────────────────────
print(f"\n🌳  Surrogate Decision Tree")
print(f"    Max depth       : {dt_model.max_depth}  (actual: {dt_model.get_depth()})")
print(f"    Leaves          : {dt_model.get_n_leaves()}")
print(f"    Fidelity to PPO : {fidelity * 100:.2f}%")

# ── Top features ────────────────────────────────────────────────────────────
print(f"\n🔬  Top-5 Most Influential Kinematic Features:")
for rank, (feat, imp) in enumerate(zip(top_features[:5], top_importances[:5]), 1):
    bar = "█" * int(imp / top_importances.max() * 25)
    print(f"    {rank}. {feat:20s} {imp:.4f}  {bar}")

# ── Saved artefacts ─────────────────────────────────────────────────────────
print(f"\n💾  Saved Files:")
print(f"    ppo_highway_agent.zip    — PPO weights (reload with PPO.load())")
print(f"    ppo_highway_agent.mp4    — agent video (inline or download)")
print(f"    decision_tree_policy.png — visual tree diagram")
print(f"    feature_importance.png   — feature importance bar chart")

print("\n" + "=" * 72)
print("✅  Pipeline complete. All 12 cells executed successfully.")
print("=" * 72)

# =============================================================================
# PHASE 2 — Research Experiments (Cells 13–18)
# Append these cells to rl_xai_highway.py and run after Cell 12.
# All variables from Phase 1 (model, X_rollout, y_rollout, FEATURE_NAMES,
# ACTION_LABELS, ENV_CONFIG, SEED) must already be in memory.
# =============================================================================


# %% [markdown]
"""
---
# 🔬 Phase 2 — Research Experiments

**Research Question:**
> *How accurately can a highway-driving reinforcement-learning policy be
> represented by a compact interpretable surrogate model — and how does
> this fidelity change with model complexity, feature ablation, and
> distribution shift?*

## Experiments
| # | Experiment | Key Output |
|---|-----------|-----------|
| 13 | Train / Test split | Honest test fidelity (no data leakage) |
| 14 | Depth vs. fidelity sweep | Complexity–interpretability trade-off curve |
| 15 | `ego_y` feature ablation | Causal role of lane position |
| 16 | Driving behaviour metrics | Policy quality beyond reward |
| 17 | Generalisation under distribution shift | Surrogate robustness |
| 18 | Research summary report | All results consolidated |
"""


# %% [markdown]
"""
## Cell 13 — Train / Test Split: Honest Fidelity Measurement

**Why this matters:**
The 99.23% fidelity from Phase 1 was measured on the *same* data used to
train the Decision Tree. That is in-sample accuracy — the tree may have
simply memorised the trajectories. An honest evaluation requires the tree
to be tested on *unseen* PPO decisions.

We fix this by:
1. Collecting a fresh, larger rollout dataset (40 episodes total).
2. Splitting 80 / 20 into training and held-out test sets.
3. Fitting the Decision Tree on the training split only.
4. Reporting fidelity separately on train and test.
"""

# %%
from sklearn.model_selection import train_test_split

# ── Step 1: Collect a larger rollout dataset ──────────────────────────────
# We double the episodes (40) so both train and test sets are meaningful.
N_TOTAL_EPISODES = 40

obs_all, act_all = [], []
env_collect = __import__('gymnasium').make("highway-v0", config=ENV_CONFIG)
obs_c, _ = env_collect.reset(seed=SEED + 100)
eps_done = 0

print(f"📦 Collecting {N_TOTAL_EPISODES} episodes for train/test split...")
while eps_done < N_TOTAL_EPISODES:
    obs_all.append(obs_c.flatten())
    action_c, _ = model.predict(obs_c, deterministic=True)
    act_all.append(int(action_c))
    obs_c, _, terminated_c, truncated_c, _ = env_collect.step(action_c)
    if terminated_c or truncated_c:
        eps_done += 1
        obs_c, _ = env_collect.reset()

env_collect.close()

X_all = __import__('numpy').array(obs_all)
y_all = __import__('numpy').array(act_all)
print(f"   Total transitions : {len(X_all):,}")

# ── Step 2: 80/20 stratified split ───────────────────────────────────────
# stratify=y_all ensures action class proportions are preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.20,
    random_state=SEED,
    stratify=y_all,
)
print(f"   Train samples     : {len(X_train):,}  ({100*len(X_train)/len(X_all):.0f}%)")
print(f"   Test  samples     : {len(X_test):,}  ({100*len(X_test)/len(X_all):.0f}%)")

# ── Step 3: Fit the tree on training data only ────────────────────────────
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

dt_split = DecisionTreeClassifier(
    max_depth=4, criterion="gini",
    min_samples_split=10, min_samples_leaf=5,
    random_state=SEED,
)
dt_split.fit(X_train, y_train)

# ── Step 4: Evaluate on both splits ──────────────────────────────────────
train_fidelity = accuracy_score(y_train, dt_split.predict(X_train))
test_fidelity  = accuracy_score(y_test,  dt_split.predict(X_test))

target_names_c13 = [ACTION_LABELS[i] for i in sorted(ACTION_LABELS.keys())]

print(f"\n📊 Train/Test Fidelity Results:")
print(f"   Training fidelity : {train_fidelity * 100:.2f}%  (in-sample  — expected high)")
print(f"   Test fidelity     : {test_fidelity  * 100:.2f}%  (out-of-sample — honest score)")

gap = (train_fidelity - test_fidelity) * 100
status = "✅ minimal overfitting" if gap < 3 else "⚠️  overfitting detected"
print(f"   Generalisation gap: {gap:.2f} pp  →  {status}")

print(f"\n📋 Test-set per-class report:")
print(classification_report(
    y_test, dt_split.predict(X_test),
    labels=list(ACTION_LABELS.keys()),
    target_names=target_names_c13,
    zero_division=0,
))


# %% [markdown]
"""
## Cell 14 — Depth vs. Fidelity: Interpretability–Complexity Trade-off

We sweep `max_depth` ∈ {1, 2, 3, 4, 5, 6, 8} and, for each depth, record:
- **Train fidelity** (how well the tree fits the training data)
- **Test fidelity** (generalisation to unseen PPO decisions)
- **Number of leaves** (proxy for rule complexity)

The resulting curve directly answers: *how many decision rules does a
surrogate model need before it faithfully represents the PPO policy?*
A steep early rise followed by a plateau indicates that a shallow tree
captures most of the policy structure.
"""

# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DEPTHS = [1, 2, 3, 4, 5, 6, 8]

results_depth = []
for depth in DEPTHS:
    dt_d = DecisionTreeClassifier(
        max_depth=depth, criterion="gini",
        min_samples_split=10, min_samples_leaf=5,
        random_state=SEED,
    )
    dt_d.fit(X_train, y_train)
    results_depth.append({
        "depth":          depth,
        "actual_depth":   dt_d.get_depth(),
        "n_leaves":       dt_d.get_n_leaves(),
        "train_fidelity": accuracy_score(y_train, dt_d.predict(X_train)) * 100,
        "test_fidelity":  accuracy_score(y_test,  dt_d.predict(X_test))  * 100,
    })

# ── Print table ──────────────────────────────────────────────────────────
print(f"{'Depth':>6} {'Actual':>7} {'Leaves':>7} {'Train Fid%':>11} {'Test Fid%':>10}")
print("-" * 46)
for r in results_depth:
    marker = "  ◄ Phase 1 point" if r["depth"] == 4 else ""
    print(f"{r['depth']:>6} {r['actual_depth']:>7} {r['n_leaves']:>7} "
          f"{r['train_fidelity']:>10.2f} {r['test_fidelity']:>10.2f}{marker}")

# ── Plot ─────────────────────────────────────────────────────────────────
depths_x      = [r["depth"]          for r in results_depth]
train_fid_y   = [r["train_fidelity"] for r in results_depth]
test_fid_y    = [r["test_fidelity"]  for r in results_depth]
leaves_y      = [r["n_leaves"]       for r in results_depth]

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(depths_x, train_fid_y, "o-",  color="#4e9af1", lw=2, ms=7, label="Train Fidelity")
ax1.plot(depths_x, test_fid_y,  "s--", color="#eb5757", lw=2, ms=7, label="Test Fidelity")
ax1.axvline(x=4, color="grey", ls=":", lw=1.5, alpha=0.7, label="Phase 1 depth (4)")
ax1.set_xlabel("Decision Tree max_depth", fontsize=12)
ax1.set_ylabel("Fidelity to PPO Policy (%)", fontsize=12)
ax1.set_ylim(50, 101)
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(2))
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="lower right", fontsize=10)

ax2 = ax1.twinx()
ax2.bar(depths_x, leaves_y, alpha=0.15, color="#6fcf97", width=0.6, label="# Leaves")
ax2.set_ylabel("Number of Leaves (tree complexity)", fontsize=11, color="#2d9e6b")
ax2.tick_params(axis="y", labelcolor="#2d9e6b")
ax2.legend(loc="center right", fontsize=10)

plt.title(
    "Interpretability–Fidelity Trade-off\n"
    "Decision Tree depth vs. fidelity to PPO policy | highway-v0",
    fontsize=13, fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("depth_fidelity_tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved 'depth_fidelity_tradeoff.png'.")


# %% [markdown]
"""
## Cell 15 — `ego_y` Feature Ablation

**Motivation:**
The Phase 1 feature importance analysis showed that `ego_y` (the ego
vehicle's lateral/lane position) accounts for **97.6%** of the Decision
Tree's Gini reduction. But feature importance in a single tree is
*correlational*, not *causal* — the tree might have chosen `ego_y` simply
because it happened to be the most convenient split variable.

We run a controlled ablation with three conditions:

| Condition | Features used | Expected outcome |
|-----------|--------------|-----------------|
| **Baseline** | All 25 features | Reference fidelity |
| **Ablated** | All 25 except `ego_y` | Fidelity if ego_y is truly critical |
| **Ego-y only** | Only `ego_y` (1 feature) | Upper bound of lane-position alone |

If "Ego-y only" ≈ Baseline: the policy is *essentially a lane-position lookup table*.
If removing `ego_y` causes a large fidelity drop: the finding is causally confirmed.
"""

# %%
ego_y_idx = FEATURE_NAMES.index("ego_y")   # column index of ego_y in the 25-d vector

# ── Condition 1: All 25 features (baseline) ──────────────────────────────
dt_baseline = DecisionTreeClassifier(max_depth=4, random_state=SEED,
                                     min_samples_split=10, min_samples_leaf=5)
dt_baseline.fit(X_train, y_train)
fid_baseline_train = accuracy_score(y_train, dt_baseline.predict(X_train)) * 100
fid_baseline_test  = accuracy_score(y_test,  dt_baseline.predict(X_test))  * 100

# ── Condition 2: Remove ego_y ─────────────────────────────────────────────
ablated_cols = [i for i in range(25) if i != ego_y_idx]
ablated_names = [FEATURE_NAMES[i] for i in ablated_cols]

dt_ablated = DecisionTreeClassifier(max_depth=4, random_state=SEED,
                                    min_samples_split=10, min_samples_leaf=5)
dt_ablated.fit(X_train[:, ablated_cols], y_train)
fid_ablated_train = accuracy_score(y_train, dt_ablated.predict(X_train[:, ablated_cols])) * 100
fid_ablated_test  = accuracy_score(y_test,  dt_ablated.predict(X_test[:, ablated_cols]))  * 100

# ── Condition 3: ego_y only (1 feature) ──────────────────────────────────
X_ego_y_train = X_train[:, [ego_y_idx]]
X_ego_y_test  = X_test[:,  [ego_y_idx]]

dt_ego_only = DecisionTreeClassifier(max_depth=4, random_state=SEED,
                                     min_samples_split=10, min_samples_leaf=5)
dt_ego_only.fit(X_ego_y_train, y_train)
fid_ego_train = accuracy_score(y_train, dt_ego_only.predict(X_ego_y_train)) * 100
fid_ego_test  = accuracy_score(y_test,  dt_ego_only.predict(X_ego_y_test))  * 100

# ── Print results ─────────────────────────────────────────────────────────
print("=" * 60)
print("ego_y Ablation Study — Decision Tree Fidelity to PPO Policy")
print("=" * 60)
print(f"{'Condition':<28} {'Train Fid%':>10} {'Test Fid%':>10}")
print("-" * 50)
print(f"{'Baseline (all 25 features)':<28} {fid_baseline_train:>10.2f} {fid_baseline_test:>10.2f}")
print(f"{'Ablated (ego_y removed)':<28} {fid_ablated_train:>10.2f} {fid_ablated_test:>10.2f}")
print(f"{'ego_y only (1 feature)':<28} {fid_ego_train:>10.2f} {fid_ego_test:>10.2f}")
print("=" * 60)

# Quantify the impact
drop = fid_baseline_test - fid_ablated_test
gain_solo = fid_ego_test
print(f"\n🔍 Interpretation:")
print(f"   Removing ego_y caused a {drop:.2f} pp drop in test fidelity.")
print(f"   ego_y alone achieves {gain_solo:.2f}% test fidelity (out of {fid_baseline_test:.2f}% baseline).")

if gain_solo > 90:
    print("\n   ✅ STRONG FINDING: The PPO policy is largely representable")
    print("      by lane position alone. It has learned a lane-centric strategy.")
elif gain_solo > 75:
    print("\n   🟡 MODERATE FINDING: ego_y dominates but other features")
    print("      contribute meaningfully to full fidelity.")
else:
    print("\n   ⚠️  ego_y importance may be a tree artefact — the feature")
    print("      appears correlative rather than causally dominant.")

# ── Bar chart visualisation ───────────────────────────────────────────────
labels_abl  = ["All 25 features\n(Baseline)", "Without ego_y\n(Ablated)", "ego_y only\n(Single feature)"]
train_scores = [fid_baseline_train, fid_ablated_train, fid_ego_train]
test_scores  = [fid_baseline_test,  fid_ablated_test,  fid_ego_test]

x = np.arange(3)
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, train_scores, width, label="Train Fidelity", color="#4e9af1", edgecolor="grey")
bars2 = ax.bar(x + width/2, test_scores,  width, label="Test Fidelity",  color="#eb5757", edgecolor="grey")

for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels_abl, fontsize=10)
ax.set_ylabel("Fidelity to PPO Policy (%)", fontsize=11)
ax.set_ylim(0, 108)
ax.axhline(fid_baseline_test, color="grey", ls="--", lw=1, alpha=0.6)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_title(
    "Feature Ablation: Role of ego_y in Surrogate Policy Fidelity\n"
    "Decision Tree (max_depth=4) | highway-v0",
    fontsize=12, fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("ego_y_ablation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved 'ego_y_ablation.png'.")


# %% [markdown]
"""
## Cell 16 — Driving Behaviour Metrics

**Motivation:**
Reward alone is insufficient to characterise driving quality. A policy that
scores 26.38 reward might achieve this by speeding on an empty road, by
always staying in one lane, or by occasional lane changes — the reward
doesn't distinguish these.

We run 30 evaluation episodes and measure five behavioural indicators:

| Metric | Definition |
|--------|-----------|
| **Collision rate** | Fraction of episodes ending in a crash |
| **Mean episode speed** | Average speed (m/s) of the ego vehicle |
| **Episode length** | Number of timesteps before termination |
| **Lane changes** | Count of LANE_LEFT (0) + LANE_RIGHT (2) actions |
| **Success rate** | Episodes that ran to full time limit without collision |
"""

# %%
import gymnasium as gym
import highway_env  # noqa — ensures highway-v0 is registered

N_METRIC_EPISODES = 30

episode_rewards     = []
episode_speeds      = []
episode_lengths     = []
episode_lane_changes = []
collision_flags     = []
success_flags       = []

metric_env = gym.make("highway-v0", config=ENV_CONFIG)
obs_m, _ = metric_env.reset(seed=SEED + 200)
ep_reward, ep_speed_sum, ep_steps, ep_lc = 0.0, 0.0, 0, 0
eps_m = 0

print(f"🚗 Collecting driving behaviour metrics over {N_METRIC_EPISODES} episodes...")

while eps_m < N_METRIC_EPISODES:
    action_m, _ = model.predict(obs_m, deterministic=True)
    obs_m, reward_m, terminated_m, truncated_m, info_m = metric_env.step(action_m)

    ep_reward   += reward_m
    ep_steps    += 1
    ep_lc       += 1 if int(action_m) in (0, 2) else 0  # LANE_LEFT or LANE_RIGHT

    # highway-env info dict provides 'speed' (ego vehicle speed in m/s)
    ep_speed_sum += info_m.get("speed", 0.0)

    if terminated_m or truncated_m:
        crashed  = info_m.get("crashed", False)
        episode_rewards.append(ep_reward)
        episode_speeds.append(ep_speed_sum / max(ep_steps, 1))
        episode_lengths.append(ep_steps)
        episode_lane_changes.append(ep_lc)
        collision_flags.append(crashed)
        success_flags.append(not crashed)

        eps_m += 1
        ep_reward, ep_speed_sum, ep_steps, ep_lc = 0.0, 0.0, 0, 0
        obs_m, _ = metric_env.reset()

metric_env.close()

# ── Compute aggregate statistics ──────────────────────────────────────────
collision_rate = np.mean(collision_flags)  * 100
success_rate   = np.mean(success_flags)   * 100
mean_speed     = np.mean(episode_speeds)
std_speed      = np.std(episode_speeds)
mean_length    = np.mean(episode_lengths)
std_length     = np.std(episode_lengths)
mean_lc        = np.mean(episode_lane_changes)
mean_reward_m  = np.mean(episode_rewards)
std_reward_m   = np.std(episode_rewards)

print(f"\n{'=' * 56}")
print(f"  Driving Behaviour Metrics ({N_METRIC_EPISODES} episodes)")
print(f"{'=' * 56}")
print(f"  Mean reward        : {mean_reward_m:.2f} ± {std_reward_m:.2f}")
print(f"  Collision rate     : {collision_rate:.1f}%")
print(f"  Success rate       : {success_rate:.1f}%")
print(f"  Mean speed         : {mean_speed:.2f} ± {std_speed:.2f}  m/s")
print(f"  Mean episode len   : {mean_length:.1f} ± {std_length:.1f}  steps")
print(f"  Mean lane changes  : {mean_lc:.2f}  per episode")
print(f"{'=' * 56}")

# ── Distribution plots ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].hist(episode_rewards,      bins=10, color="#4e9af1", edgecolor="white")
axes[0].set_title("Episode Reward");    axes[0].set_xlabel("Reward")

axes[1].hist(episode_speeds,       bins=10, color="#6fcf97", edgecolor="white")
axes[1].set_title("Mean Speed (m/s)");  axes[1].set_xlabel("Speed")
axes[1].axvline(20, color="red", ls="--", lw=1.2, label="min reward band")
axes[1].axvline(30, color="red", ls="--", lw=1.2, label="max reward band")
axes[1].legend(fontsize=7)

axes[2].hist(episode_lengths,      bins=10, color="#f2c94c", edgecolor="white")
axes[2].set_title("Episode Length");    axes[2].set_xlabel("Timesteps")

axes[3].hist(episode_lane_changes, bins=10, color="#bb6bd9", edgecolor="white")
axes[3].set_title("Lane Changes");      axes[3].set_xlabel("Count")

for ax in axes:
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    f"PPO Driving Behaviour Distributions | highway-v0 | {N_METRIC_EPISODES} episodes\n"
    f"Collision rate: {collision_rate:.1f}%  |  Success rate: {success_rate:.1f}%  |  "
    f"Mean speed: {mean_speed:.1f} m/s",
    fontsize=11, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("driving_behaviour_metrics.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved 'driving_behaviour_metrics.png'.")


# %% [markdown]
"""
## Cell 17 — Generalisation Under Distribution Shift

**Core question:**
The Decision Tree was trained on rollouts from a specific traffic configuration
(20 vehicles). Does it remain faithful when the environment changes?

We test the surrogate model on three traffic density conditions it was never
trained on:
- **Sparse traffic** (10 vehicles) — easier, more open road
- **Medium traffic** (20 vehicles) — same as training (*reference*)
- **Dense traffic** (30 vehicles) — harder, requires more reactive decisions

For each condition we:
1. Collect 15 fresh PPO episodes.
2. Measure how often the pre-trained Decision Tree agrees with PPO.

A high drop in dense traffic would indicate the tree's explanation breaks down
under conditions where the policy becomes more reactive — an important
limitation to report.
"""

# %%
TRAFFIC_CONDITIONS = [
    {"label": "Sparse  (10 vehicles)",  "n_vehicles": 10,  "seeds": range(SEED + 300, SEED + 315)},
    {"label": "Medium  (20 vehicles)",  "n_vehicles": 20,  "seeds": range(SEED + 315, SEED + 330)},
    {"label": "Dense   (30 vehicles)",  "n_vehicles": 30,  "seeds": range(SEED + 330, SEED + 345)},
]

generalisation_results = []

for condition in TRAFFIC_CONDITIONS:
    cond_config = dict(ENV_CONFIG)
    cond_config["vehicles_count"] = condition["n_vehicles"]

    obs_g, act_g = [], []
    g_env = gym.make("highway-v0", config=cond_config)

    for seed_g in condition["seeds"]:
        obs_ep, _ = g_env.reset(seed=int(seed_g))
        done_g = False
        while not done_g:
            obs_g.append(obs_ep.flatten())
            a_g, _ = model.predict(obs_ep, deterministic=True)
            act_g.append(int(a_g))
            obs_ep, _, t_g, tr_g, _ = g_env.step(a_g)
            done_g = t_g or tr_g
    g_env.close()

    X_g = np.array(obs_g)
    y_g = np.array(act_g)

    # Evaluate the Phase-1 baseline tree (trained on 20-vehicle rollouts)
    y_g_pred = dt_baseline.predict(X_g)
    fid_g    = accuracy_score(y_g, y_g_pred) * 100

    generalisation_results.append({
        "label":       condition["label"],
        "n_vehicles":  condition["n_vehicles"],
        "n_samples":   len(X_g),
        "fidelity":    fid_g,
    })
    print(f"   {condition['label']:30s}  n={len(X_g):4d}  fidelity={fid_g:.2f}%")

# ── Plot ─────────────────────────────────────────────────────────────────
cond_labels  = [r["label"]    for r in generalisation_results]
cond_fids    = [r["fidelity"] for r in generalisation_results]
cond_colors  = ["#6fcf97", "#4e9af1", "#eb5757"]

fig, ax = plt.subplots(figsize=(8, 4))
bars_g = ax.bar(cond_labels, cond_fids, color=cond_colors, edgecolor="grey", width=0.5)

for bar_g, val_g in zip(bars_g, cond_fids):
    ax.text(bar_g.get_x() + bar_g.get_width() / 2,
            bar_g.get_height() + 0.4,
            f"{val_g:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Surrogate Fidelity to PPO (%)", fontsize=11)
ax.set_ylim(0, 108)
ax.axhline(fid_baseline_test, color="navy", ls="--", lw=1.5, alpha=0.6,
           label=f"Test fidelity at training density ({fid_baseline_test:.1f}%)")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.set_title(
    "Surrogate Model Generalisation Under Traffic Distribution Shift\n"
    "Decision Tree (depth=4) trained on 20-vehicle rollouts | highway-v0",
    fontsize=11, fontweight="bold", pad=12,
)
plt.tight_layout()
plt.savefig("generalisation_distribution_shift.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Saved 'generalisation_distribution_shift.png'.")


# %% [markdown]
"""
## Cell 18 — Consolidated Research Summary Report

Full tabular summary of all Phase 1 and Phase 2 findings, formatted for
inclusion in a research report or README.
"""

# %%
print("=" * 72)
print("  RESEARCH SUMMARY REPORT")
print("  Interpretable Autonomous Driving via RL Policy Distillation")
print("  highway-v0 | PPO (SB3) + Decision Tree Surrogate (scikit-learn)")
print("=" * 72)

# ── 1. Agent performance ──────────────────────────────────────────────────
print("\n── 1. PPO Agent Performance ─────────────────────────────────────────")
print(f"   Training steps          : {TOTAL_TIMESTEPS:,}")
print(f"   Parallel envs           : {n_envs}")
print(f"   Mean reward (10 eps)    : {mean_reward:.2f} ± {std_reward:.2f}")
print(f"   Mean reward (30 eps)    : {mean_reward_m:.2f} ± {std_reward_m:.2f}")
print(f"   Collision rate          : {collision_rate:.1f}%")
print(f"   Success rate            : {success_rate:.1f}%")
print(f"   Mean speed              : {mean_speed:.2f} ± {std_speed:.2f} m/s")
print(f"   Mean lane changes/ep    : {mean_lc:.2f}")
print(f"   Mean episode length     : {mean_length:.1f} ± {std_length:.1f} steps")

# ── 2. Honest fidelity ───────────────────────────────────────────────────
print("\n── 2. Surrogate Fidelity (Train/Test Split) ─────────────────────────")
print(f"   Total transitions       : {len(X_all):,}")
print(f"   Train transitions       : {len(X_train):,} (80%)")
print(f"   Test transitions        : {len(X_test):,}  (20%)")
print(f"   Training fidelity       : {train_fidelity * 100:.2f}%")
print(f"   Test fidelity           : {test_fidelity  * 100:.2f}%  ← honest score")
print(f"   Generalisation gap      : {(train_fidelity - test_fidelity)*100:.2f} pp")

# ── 3. Depth sweep ───────────────────────────────────────────────────────
print("\n── 3. Depth vs. Fidelity Sweep ──────────────────────────────────────")
print(f"   {'Depth':>5}  {'Leaves':>6}  {'Train%':>7}  {'Test%':>7}")
for r in results_depth:
    mark = " ◄" if r["depth"] == 4 else ""
    print(f"   {r['depth']:>5}  {r['n_leaves']:>6}  {r['train_fidelity']:>7.2f}  {r['test_fidelity']:>7.2f}{mark}")

# ── 4. Ablation ──────────────────────────────────────────────────────────
print("\n── 4. ego_y Feature Ablation (Test Fidelity) ────────────────────────")
print(f"   Baseline (25 features)  : {fid_baseline_test:.2f}%")
print(f"   Ablated  (ego_y removed): {fid_ablated_test:.2f}%   drop = {fid_baseline_test - fid_ablated_test:.2f} pp")
print(f"   ego_y only (1 feature)  : {fid_ego_test:.2f}%")

# ── 5. Generalisation ────────────────────────────────────────────────────
print("\n── 5. Generalisation Under Distribution Shift ───────────────────────")
for r in generalisation_results:
    print(f"   {r['label']:32s}: {r['fidelity']:.2f}%  (n={r['n_samples']})")

# ── 6. Saved artefacts ───────────────────────────────────────────────────
print("\n── 6. Saved Artefacts ───────────────────────────────────────────────")
artefacts = [
    ("ppo_highway_agent.zip",                "PPO weights"),
    ("ppo_highway_agent.mp4",                "Agent video"),
    ("decision_tree_policy.png",             "Phase 1 tree diagram"),
    ("feature_importance.png",               "Phase 1 feature importance"),
    ("depth_fidelity_tradeoff.png",          "Depth sweep curve"),
    ("ego_y_ablation.png",                   "Ablation bar chart"),
    ("driving_behaviour_metrics.png",        "Behaviour metric distributions"),
    ("generalisation_distribution_shift.png","Distribution shift bar chart"),
]
for fname, desc in artefacts:
    print(f"   {fname:45s} {desc}")

print("\n" + "=" * 72)
print("  Research Question Addressed:")
print("  'How accurately can a highway RL policy be represented by a")
print("   compact interpretable surrogate, and how does fidelity change")
print("   with model complexity and distribution shift?'")
print("=" * 72)
print("✅  Phase 2 complete. All experiments executed successfully.")
print("=" * 72)
