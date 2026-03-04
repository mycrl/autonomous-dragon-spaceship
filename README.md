# rl-iss-docking-simulator

A Dragon spacecraft ISS docking autonomous driving system built with a custom
[Gymnasium](https://gymnasium.farama.org/) environment and trained with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

## Overview

The simulator models a SpaceX Dragon spacecraft approaching the International
Space Station (ISS) docking port using simplified Newtonian mechanics in 3D
space. A reinforcement-learning agent controls the spacecraft's Draco thrusters
to navigate from an initial stand-off distance of ~100 m to a successful soft
dock (within 0.5 m, below 0.3 m/s).

### Environment

| Property | Value |
|---|---|
| Observation space | 6-D continuous — position (x, y, z) + velocity (vx, vy, vz) |
| Action space | 3-D continuous \[−1, 1\] — normalised thrust per axis |
| Spacecraft mass | 12 000 kg |
| Max thrust per axis | 400 N |
| Simulation time-step | 0.5 s |
| Max episode length | 2 000 steps |

**Episode termination conditions**

- ✅ **Success** — distance ≤ 0.5 m and approach speed ≤ 0.3 m/s  
- 💥 **Crash** — reached the port with speed > 0.3 m/s  
- 🚀 **Out of bounds** — drifted beyond 300 m or 50 m lateral offset  
- ⏱ **Timeout** — 2 000 steps elapsed  

**Reward shaping**

- `+100` for a successful soft dock  
- `−50` for a crash or going out of bounds  
- `+0.1` per metre of distance closed each step  
- `−0.005 × lateral_offset` per step (encourages staying on the approach axis)  
- `−0.01` per step (time penalty, encourages efficiency)  

### Algorithm

Proximal Policy Optimisation (PPO) from Stable-Baselines3 is used to train the
agent with a multi-layer perceptron (MLP) policy.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train.py
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | 500 000 | Total training timesteps |
| `--save-path` | `models/ppo_docking` | Where to save the model |
| `--n-envs` | 4 | Number of parallel environments |

Example — long training run:

```bash
python train.py --timesteps 2000000 --save-path models/ppo_docking_2M
```

### Evaluate

```bash
python evaluate.py --model models/ppo_docking --episodes 20
```

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `models/ppo_docking` | Path to trained model |
| `--episodes` | 10 | Number of evaluation episodes |

## File Structure

```
.
├── environment.py   # Custom Gymnasium environment (IssDockingEnv)
├── train.py         # PPO training script
├── evaluate.py      # Model evaluation script
├── requirements.txt # Python dependencies
└── README.md
```

## License

See [LICENSE](LICENSE).
