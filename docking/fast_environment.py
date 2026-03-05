"""
Pure Python simulation of the SpaceX ISS Docking Simulator.

This environment perfectly mirrors the state space, action space, 
and reward scale of the actual simulator (IssDockingEnv) but runs 
locally at thousands of steps per second without Playwright or a browser.
"""

import logging
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger(__name__)

class FastIssDockingEnv(gym.Env):
    """
    Lightning-fast Gymnasium environment mirroring the SpaceX ISS Docking Simulator.
    Runs entirely in Python.
    """

    metadata = {"render_modes": []}

    OBS_KEYS: list[str] = [
        "x", "y", "z",
        "roll", "roll_rate",
        "range",
        "yaw", "yaw_rate",
        "rate",
        "pitch", "pitch_rate",
    ]
    OBS_HIGH: np.ndarray = np.array(
        [300.0, 300.0, 300.0, 180.0, 10.0, 500.0, 180.0, 10.0, 5.0, 180.0, 10.0],
        dtype=np.float32,
    )

    SUCCESS_THRESHOLD: float = 0.2
    MAX_RANGE: float = 350.0   # metres
    MAX_ATTITUDE: float = 30.0   # degrees
    MAX_SAFE_RATE: float = 0.2   # m/s
    NEAR_DISTANCE: float = 5.0   # metres
    ATTITUDE_KEYS: tuple[str, ...] = ("roll", "yaw", "pitch")

    def __init__(
        self,
        step_delay: float = 0.5,
        max_steps: int = 3000,
        render_mode=None,
        **kwargs
    ) -> None:
        super().__init__()

        # step_delay is now used entirely as our local physics 'dt' (delta time)
        self.dt = step_delay
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        self.observation_space = spaces.Box(
            low=-self.OBS_HIGH,
            high=self.OBS_HIGH,
            dtype=np.float32,
        )

        self._steps: int = 0
        self.fuel_used: int = 0
        self._prev_state = {}

        # Defines impulses for each action
        # Translating imparts velocity. Rotating imparts angular velocity.
        self.TRANSLATION_PULSE = 0.06  # m/s per click
        self.ROTATION_PULSE = 0.1      # deg/s per click

        self._dim_to_state_keys = {
            0: ("range", "rate"), 
            1: ("y", None),
            2: ("x", None),
            3: ("roll", "roll_rate"),
            4: ("pitch", "pitch_rate"),
            5: ("yaw", "yaw_rate"),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self.fuel_used = 0
        
        # Initialize reasonable starting state similar to real simulator
        self.state_vars = {
            "x": self.np_random.uniform(-15.0, 15.0),
            "vx": self.np_random.uniform(-0.02, 0.02),
            "y": self.np_random.uniform(-15.0, 15.0),
            "vy": self.np_random.uniform(-0.02, 0.02),
            "z": self.np_random.uniform(180.0, 220.0), # z and range start around 200m
            "vz": self.np_random.uniform(-0.02, 0.02),
            
            "roll": self.np_random.uniform(-15.0, 15.0),
            "roll_rate": self.np_random.uniform(-0.2, 0.2),
            "pitch": self.np_random.uniform(-15.0, 15.0),
            "pitch_rate": self.np_random.uniform(-0.2, 0.2),
            "yaw": self.np_random.uniform(-15.0, 15.0),
            "yaw_rate": self.np_random.uniform(-0.2, 0.2),
        }
        
        self.state_vars["range"] = math.sqrt(self.state_vars["x"]**2 + self.state_vars["y"]**2 + self.state_vars["z"]**2)
        # Assuming closing speed is negative rate (distance drops per second)
        self.state_vars["rate"] = 0.0

        obs = self._get_obs()
        self._prev_state = self._obs_to_dict(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        button_presses = 0
        active_dims = set()
        
        # Process actions (apply impulses)
        for dim, act_val in enumerate(action):
            act_val = int(act_val)
            if act_val in (1, 2):
                button_presses += 1
                active_dims.add(dim)
                
                direction = 1 if act_val == 1 else -1

                if dim == 0:   # Forward(1)/Backward(2) -> affects Z velocity. Translating forward (1) makes Z decrease -> vz goes negative
                    self.state_vars["vz"] -= direction * self.TRANSLATION_PULSE
                elif dim == 1: # Up(1)/Down(2) -> affects Y velocity
                    self.state_vars["vy"] += direction * self.TRANSLATION_PULSE
                elif dim == 2: # Right(1)/Left(2) -> affects X velocity
                    self.state_vars["vx"] += direction * self.TRANSLATION_PULSE
                elif dim == 3: # Roll Right(1)/Left(2)
                    self.state_vars["roll_rate"] += direction * self.ROTATION_PULSE
                elif dim == 4: # Pitch Up(1)/Down(2) -> Right-hand rule / UI consistency. Just map it explicitly.
                    self.state_vars["pitch_rate"] += direction * self.ROTATION_PULSE
                elif dim == 5: # Yaw Right(1)/Left(2)
                    self.state_vars["yaw_rate"] += direction * self.ROTATION_PULSE

        if button_presses > 0:
            self.fuel_used += button_presses

        self._steps += 1

        # Euler integrate Physics!
        # Save range for rate calculation
        old_range = self.state_vars["range"]

        # Positions
        self.state_vars["x"] += self.state_vars["vx"] * self.dt
        self.state_vars["y"] += self.state_vars["vy"] * self.dt
        self.state_vars["z"] += self.state_vars["vz"] * self.dt
        
        # Angles
        self.state_vars["roll"] += self.state_vars["roll_rate"] * self.dt
        self.state_vars["pitch"] += self.state_vars["pitch_rate"] * self.dt
        self.state_vars["yaw"] += self.state_vars["yaw_rate"] * self.dt

        # Recalculate Range & Rate
        new_range = math.sqrt(self.state_vars["x"]**2 + self.state_vars["y"]**2 + self.state_vars["z"]**2)
        self.state_vars["range"] = new_range
        self.state_vars["rate"] = (new_range - old_range) / self.dt

        obs = self._get_obs()
        state = self._obs_to_dict(obs)

        # =========================================================
        # REWARD COMPUTATION
        # =========================================================
        reward = 0.0

        # 1. Action/Fuel penalty
        reward -= 0.01 * button_presses

        # 2. General Improvements (Global Dense Reward)
        range_diff = self._prev_state["range"] - state["range"]
        reward += range_diff * 1.5

        prev_attitude_error = sum(abs(self._prev_state[k]) for k in self.ATTITUDE_KEYS)
        curr_attitude_error = sum(abs(state[k]) for k in self.ATTITUDE_KEYS)
        reward += (prev_attitude_error - curr_attitude_error) * 2.0

        prev_pos_error = abs(self._prev_state["x"]) + abs(self._prev_state["y"]) + abs(self._prev_state["z"])
        curr_pos_error = abs(state["x"]) + abs(state["y"]) + abs(state["z"])
        reward += (prev_pos_error - curr_pos_error) * 1.5

        # 3. Targeted Branch Rewards (Credit Assignment)
        for dim in active_dims:
            pos_key, _ = self._dim_to_state_keys[dim]
            if pos_key:
                improvement = abs(self._prev_state[pos_key]) - abs(state[pos_key])
                reward += improvement * 5.0  

        # 4. Safety Constraints & Violations (Extreme Penalties)
        current_range = state["range"]
        if current_range < self.NEAR_DISTANCE and state["rate"] < -self.MAX_SAFE_RATE:
            reward -= 10.0

        if current_range > 15.0 and abs(state["rate"]) < 0.01:
            reward -= 0.1

        for key in ("roll_rate", "yaw_rate", "pitch_rate"):
            rate_val = abs(state[key])
            if rate_val > 0.2:
                excess = rate_val - 0.2
                reward -= (excess * 10.0) ** 2  

        # 5. Terminal Conditions
        terminated = False
        truncated = False
        success = False

        if self._is_docked(state):
            reward += 1000.0
            terminated = True
            success = True
        elif current_range > self.MAX_RANGE:
            reward -= 1000.0
            terminated = True
        elif any(abs(state[k]) > self.MAX_ATTITUDE for k in self.ATTITUDE_KEYS):
            reward -= 1000.0
            terminated = True
        elif self._steps >= self.max_steps:
            truncated = True

        self._prev_state = state

        info = {
            "steps": self._steps,
            "fuel_used": int(self.fuel_used),
            "success": success,
            "button_presses": int(button_presses),
            **state,
        }

        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        pass # No browser to close!

    def _get_obs(self) -> np.ndarray:
        obs = np.array([self.state_vars[k] for k in self.OBS_KEYS], dtype=np.float32)
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        return all(abs(v) <= FastIssDockingEnv.SUCCESS_THRESHOLD for v in state.values())

