# Autonomous Dragon Spaceship

A Dragon spacecraft ISS docking autonomous driving system built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO.

The agent connects to the real [SpaceX ISS Docking Simulator](https://iss-sim.spacex.com/)
running in a Chrome browser, reads state data from the page DOM, and clicks
the control buttons to manoeuvre the Dragon spacecraft to a successful soft dock.

## Overview

The SpaceX ISS Docking Simulator presents a browser-based interface that
familiarises users with the controls used by NASA astronauts. Successful
docking requires all six error readings (position offsets x/y/z and attitude
errors roll/pitch/yaw), the approach rate, and the range to all fall below 0.2.

### Environment

| Property | Value |
|---|---|
| Observation space | 11-D continuous — x, y, z (m), roll (°), roll rate (°/s), range (m), yaw (°), yaw rate (°/s), rate (m/s), pitch (°), pitch rate (°/s) |
| Action space | MultiDiscrete([3, 3, 3, 3, 3, 3]) — Simultaneous multi-axis control |
| Step delay | Wait for physics to settle after each button press ensemble |
| State Normalization | Handled via `VecNormalize` in Stable-Baselines3 |
| Max episode length | 3 000 steps |

**Actions**

The policy outputs an array of 6 discrete values, one for each degree of freedom (Translation: X, Y, Z / Rotation: Roll, Pitch, Yaw). For each dimension:

- `0`: NO_OP
- `1`: Positive control (e.g. forward, up, right, roll_right, pitch_up, yaw_right)
- `2`: Negative control (e.g. backward, down, left, roll_left, pitch_down, yaw_left)

This MultiDiscrete setup enables the agent to issue simultaneous thrust instructions (e.g., pitch up while moving forward).

**Episode termination conditions**

- ✅ **Success** — all readings are within ±0.2
- 🧭 **Attitude limit** — `|roll|` or `|yaw|` or `|pitch|` > 80°
- 🚫 **Out of range** — range > 350 m
- ⏱ **Timeout** — 3 000 steps elapsed

**Reward shaping (Dense Guidance & Targeted Credit Assignment)**

- `+1000` for a successful docking.
- `-1000` for terminal failures (out-of-range or extreme attitude failures).
- Penalty for every button press (`-0.01`) to encourage letting inertia do the work (NO_OP).
- **Targeted Branch Rewards:** The environment implements direct credit assignment. When the agent uses a specific axis control (e.g. Yaw), it receives targeted scaled bonuses/penalties based explicitly on whether that dimension improved or worsened.
- **Extreme Angular Rate Penalty:** Any rotation rate (roll/pitch/yaw) exceeding 0.2 °/s receives an exponentially increasing penalty (e.g., reaching 0.8 °/s heavily penalizes the agent). This forces the agent to use counter-burns to stabilize momentum before it spins out of control.
- Severe penalties for high approach speeds (`Rate <= -0.2 m/s`) when very close (< 5m) to the ISS and stagnation at long distances.

### Algorithm

Proximal Policy Optimization (**PPO**) from Stable-Baselines3 with an MLP policy.
Training supports vectorized multi-environment sampling configurable via
`--num-envs` (default `5`). State input is standardized dynamically using `VecNormalize` to guarantee numerical stability.

## Project Structure

```text
.
├── docking/
│   ├── __init__.py      # Package — exports IssDockingEnv
│   ├── browser.py       # Browser automation layer (CDP via Playwright)
│   └── environment.py   # Custom Gymnasium environment with MultiDiscrete action space
├── train.py             # PPO training, env vectorization, stats saving & checkpointing
├── evaluate.py          # Loads model & env running stats for inference
├── requirements.txt     # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

### Train

```bash
# Default: 5 envs in parallel (one browser per env; managed mode auto-enabled)
python train.py --headless

# Single-environment CDP mode — connect to a manually-opened Chrome
python train.py --num-envs 1

# Optional: one shared browser with multiple tabs
python train.py --headless --num-envs 5 --shared-browser-tabs
```

When training, two things are saved periodically: the policy weights (`.zip`) and the environment normalization statistics (`_vec_normalize.pkl`). Both are required to resume training or evaluate.

Example — **resume a previous run**:

```bash
python train.py --launch-browser --resume --model-path models/ppo_docking --timesteps 1000000
```

### Stop training safely (without losing progress)

You can stop training any time with `Ctrl+C`.
The script automatically intercepts `KeyboardInterrupt`, saves the current model and statistics, and exits cleanly.

### Evaluate

Inference evaluation loads the model and perfectly replicates the input distribution sizes from training by loading `VecNormalize` stats.

```bash
# Managed mode
python evaluate.py --launch-browser --model models/ppo_docking --episodes 10
```

## License

See [LICENSE](LICENSE).
