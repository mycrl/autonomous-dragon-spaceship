# Autonomous Dragon Spaceship

A Dragon spacecraft ISS docking autonomous driving system built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) PPO.

The agent trains using a blazingly fast pure-Python simulation of the 
SpaceX ISS Docking Simulator, and when evaluated, connects to the real 
[SpaceX ISS Docking Simulator](https://iss-sim.spacex.com/) running in a 
Chrome browser to demonstrate the precise docking manoeuvre visually.

## Overview

The standard browser-based ISS Docking Simulator represents a significant bottleneck for RL 
training due to real-time constraints and UI rendering. To solve this, this repository contains 
two environments:

1. **`FastIssDockingEnv` (Training):** A lightweight, pure-Python 1:1 physics recreation of the simulator. It eliminates Playwright dependencies entirely and executes thousands of steps per second on CPU.
2. **`IssDockingEnv` (Evaluation):** Connects to the real SpaceX Simulator via Playwright and Chrome DevTools Protocol to visually evaluate the agent's performance.

Successful docking requires all six error readings (position offsets x/y/z and attitude
errors roll/pitch/yaw), the approach rate, and the range to all fall below 0.2.

### Environment

| Property | Value |
|---|---|
| Observation space | 11-D continuous – x, y, z (m), roll (°), roll rate (°/s), range (m), yaw (°), yaw rate (°/s), rate (m/s), pitch (°), pitch rate (°/s) |
| Action space | MultiDiscrete([3, 3, 3, 3, 3, 3]) – Simultaneous multi-axis control |
| Step delay | Wait for physics to settle after each button press ensemble |
| State Normalization | Handled via `VecNormalize` in Stable-Baselines3 |
| Max episode length | 3 000 steps |

**Actions**

The policy outputs an array of 6 discrete values, one for each degree of freedom
(Translation: X, Y, Z / Rotation: Roll, Pitch, Yaw). For each dimension:
- `0`: NO_OP
- `1`: Positive control (e.g. forward, up, right, roll_right, pitch_up, yaw_right)
- `2`: Negative control (e.g. backward, down, left, roll_left, pitch_down, yaw_left)

This MultiDiscrete setup enables the agent to issue simultaneous thrust instructions (e.g., pitch up while moving forward).

**Episode termination conditions**

- ✅ **Success** – all readings are within ±0.2
- 💥 **Attitude limit** – `|roll|` or `|yaw|` or `|pitch|` > 30°
- 🛸 **Out of range** – range > 350 m
- ⏱ **Timeout** – 3 000 steps elapsed

**Reward shaping (Dense Guidance & Targeted Credit Assignment)**

- `+1000` for a successful docking.
- `-1000` for terminal failures (out-of-range or extreme attitude failures).
- Penalty for every button press (`-0.01`) to encourage letting inertia do the work (NO_OP).
- **Targeted Branch Rewards:** The environment implements direct credit assignment. When the agent uses a specific axis control (e.g. Yaw), it receives targeted scaled bonuses/penalties based explicitly on whether that dimension improved or worsened.
- **Extreme Angular Rate Penalty:** Any rotation rate (roll/pitch/yaw) exceeding 0.2 °/s receives an exponentially increasing penalty. This forces the agent to use counter-burns to stabilize momentum before it spins out of control.
- Severe penalties for high approach speeds (`Rate <= -0.2 m/s`) when very close (< 5m) to the ISS and stagnation at long distances.

### Algorithm

Proximal Policy Optimization (**PPO**) from Stable-Baselines3 with an MLP policy.
Training uses vectorized multi-environment sampling configurable via `--num-envs` (default `16`). 
State input is standardized dynamically using `VecNormalize` to guarantee numerical stability.

## Project Structure

```text
.
├── docking/
│   ├── __init__.py           # Package – exports environments
│   ├── fast_environment.py   # Pure Python, ultra-fast 1:1 physics simulator
│   ├── browser.py            # Browser automation layer (CDP via Playwright)
│   └── environment.py        # Real browser Gymnasium environment for evaluation
├── train.py                  # Fast offline PPO training on FastIssDockingEnv
├── evaluate.py               # Visual evaluation on the real browser via Playwright
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

### Train (Fast offline Mode)

Thanks to the pure Python environment, training is now extremely fast and does not require a browser. 

```bash
# Default: Train with 16 parallel vector environments utilizing all CPU cores
python train.py

# Customizing training specs:
python train.py --num-envs 8 --timesteps 5000000 --checkpoint-freq 100000
```

When training, two things are saved periodically: the policy weights (`.zip`) and the environment normalization statistics (`_vec_normalize.pkl`). Both are required to evaluate or resume training.

Example – **resume a previous run**:

```bash
python train.py --resume --model-path models/ppo_docking --timesteps 1000000
```

### Stop training safely (without losing progress)

You can stop training any time with `Ctrl+C`.
The script automatically intercepts `KeyboardInterrupt`, saves the current model and statistics, and exits cleanly.

### Evaluate (Real Browser Mode)

Inference evaluation loads the model and connects to the **real SpaceX simulator in your browser**. It normalizes inputs perfectly using the training statistics (`_vec_normalize.pkl`).

```bash
# Launch Playwright browser locally and run 10 episodes visually
python evaluate.py --launch-browser --model models/ppo_docking --episodes 10
```

## License

See [LICENSE](LICENSE).
