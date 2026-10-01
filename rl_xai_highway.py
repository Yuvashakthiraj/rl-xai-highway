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
