"""
Custom Gymnasium environment for the SpaceX ISS Docking Simulator.

Wraps the browser-controlled SpaceX ISS Docking Simulator
(https://iss-sim.spacex.com/) as a Gymnasium environment with continuous
observation space and MultiDiscrete action space.
"""

import logging
import time
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .browser import SimulatorBrowser

logger = logging.getLogger(__name__)


class IssDockingEnv(gym.Env):
    """
    Gymnasium environment that wraps the SpaceX ISS Docking Simulator.
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
        launch_browser: bool = False,
        headless: bool = False,
        cdp_url: str = SimulatorBrowser.CDP_URL,
        shared_browser_tabs: bool = False,
        expected_shared_tabs: int | None = None,
        step_delay: float = 0.5,
        reset_wait: float = 3.0,
        max_steps: int = 3000,
        render_mode=None,
        **kwargs
    ) -> None:
        super().__init__()

        self.step_delay = step_delay
        self.reset_wait = reset_wait
        self.max_steps = max_steps
        self.render_mode = render_mode

        self._browser = SimulatorBrowser(
            launch=launch_browser,
            headless=headless,
            cdp_url=cdp_url,
            shared_launch=shared_browser_tabs,
            expected_shared_tabs=expected_shared_tabs,
        )
        self._browser.connect()

        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
        self.observation_space = spaces.Box(
            low=-self.OBS_HIGH,
            high=self.OBS_HIGH,
            dtype=np.float32,
        )

        self._steps: int = 0
        self.fuel_used: int = 0
        self._prev_state = {}

        self._action_map = {
            0: {1: "translate_forward", 2: "translate_backward"},
            1: {1: "translate_up", 2: "translate_down"},
            2: {1: "translate_right", 2: "translate_left"},
            3: {1: "roll_right", 2: "roll_left"},
            4: {1: "pitch_up", 2: "pitch_down"},
            5: {1: "yaw_right", 2: "yaw_left"},
        }

        # Maps dimension to state keys for targeted reward shaping:
        # Tuple of (position_error_key, rate_error_key)
        self._dim_to_state_keys = {
            0: ("range", "rate"), # Forward/backward affects range/rate
            1: ("y", None),
            2: ("x", None),
            3: ("roll", "roll_rate"),
            4: ("pitch", "pitch_rate"),
            5: ("yaw", "yaw_rate"),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._browser.reset(wait=self.reset_wait)
        self._steps = 0
        self.fuel_used = 0
        
        obs = self._get_obs()
        self._prev_state = self._obs_to_dict(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        button_presses = 0
        active_dims = set()
        
        for dim, act_val in enumerate(action):
            act_val = int(act_val)
            if act_val in (1, 2):
                btn = self._action_map[dim][act_val]
                self._browser.click_action(btn)
                button_presses += 1
                active_dims.add(dim)

        if button_presses > 0:
            self.fuel_used += button_presses

        time.sleep(self.step_delay)
        self._steps += 1

        obs = self._get_obs()
        state = self._obs_to_dict(obs)

        # =========================================================
        # REWARD COMPUTATION
        # =========================================================
        reward = 0.0

        # 1. Action/Fuel penalty
        reward -= 0.01 * button_presses

        # 2. General Improvements (Global Dense Reward)
        # Keeps AI on track even when coasting (NO_OP)
        range_diff = self._prev_state["range"] - state["range"]
        reward += range_diff * 1.5

        prev_attitude_error = sum(abs(self._prev_state[k]) for k in self.ATTITUDE_KEYS)
        curr_attitude_error = sum(abs(state[k]) for k in self.ATTITUDE_KEYS)
        reward += (prev_attitude_error - curr_attitude_error) * 2.0

        prev_pos_error = abs(self._prev_state["x"]) + abs(self._prev_state["y"]) + abs(self._prev_state["z"])
        curr_pos_error = abs(state["x"]) + abs(state["y"]) + abs(state["z"])
        reward += (prev_pos_error - curr_pos_error) * 1.5

        # 3. Targeted Branch Rewards (Credit Assignment)
        # If the agent took an action on a specific axis, give a sharp scalar bonus/penalty 
        # strictly based on whether that specific axis error improved. This helps PPO map
        # the global scalar reward variance directly to the action component taken.
        for dim in active_dims:
            pos_key, _ = self._dim_to_state_keys[dim]
            if pos_key:
                improvement = abs(self._prev_state[pos_key]) - abs(state[pos_key])
                reward += improvement * 5.0  # High targeted multiplier!

        # 4. Safety Constraints & Violations (Extreme Penalties)
        # a) Overspeed when closing in
        current_range = state["range"]
        if current_range < self.NEAR_DISTANCE and state["rate"] < -self.MAX_SAFE_RATE:
            reward -= 10.0

        # b) Stagnation when far away
        if current_range > 15.0 and abs(state["rate"]) < 0.01:
            reward -= 0.1

        # c) EXTREME Angular Rate Penalty (Strict limit around 0.2 deg/s)
        # If any rotation rate exceeds 0.2, apply a geometrically increasing penalty.
        for key in ("roll_rate", "yaw_rate", "pitch_rate"):
            rate_val = abs(state[key])
            if rate_val > 0.2:
                # E.g., rate=0.8 -> excess=0.6 -> (0.6 * 10)^2 = 36.0 penalty
                # Ensures mild drifting is forgiven, but erratic spinning is killed.
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
        self._browser.disconnect()

    def _get_obs(self) -> np.ndarray:
        raw = self._browser.read_state()
        obs = np.array([raw[k] for k in self.OBS_KEYS], dtype=np.float32)
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        return all(abs(v) <= IssDockingEnv.SUCCESS_THRESHOLD for v in state.values())

