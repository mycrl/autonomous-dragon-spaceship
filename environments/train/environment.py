"""
Pure Python simulation of the SpaceX ISS Docking Simulator.

This environment perfectly mirrors the state space, action space, 
and reward scale of the actual simulator (IssDockingEnv) but runs 
locally at thousands of steps per second without Playwright or a browser.
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .simulator import TrainDockingSimulator

logger = logging.getLogger(__name__)

class TrainIssDockingEnv(gym.Env):
    """
    Lightning-fast Gymnasium environment mirroring the SpaceX ISS Docking Simulator.
    Runs entirely in Python.
    """

    metadata = {"render_modes": []}

    INITIAL_FUEL: float = 800.0
    FUEL_PER_BUTTON: float = 1.0

    OBS_KEYS: list[str] = [
        "x", "y", "z",
        "roll", "roll_rate",
        "range",
        "yaw", "yaw_rate",
        "rate",
        "pitch", "pitch_rate",
        "fuel",
    ]
    OBS_HIGH: np.ndarray = np.array(
        [300.0, 300.0, 300.0, 180.0, 10.0, 500.0, 180.0, 10.0, 5.0, 180.0, 10.0, 1.0],
        dtype=np.float32,
    )

    SUCCESS_THRESHOLD: float = 0.2
    GOOD_POS_THRESHOLD: float = 0.2
    GOOD_ATTITUDE_THRESHOLD: float = 0.2
    GOOD_ANG_RATE_THRESHOLD: float = 0.25
    GOOD_RANGE_THRESHOLD: float = 2.0
    MAX_RANGE: float = 350.0   # metres
    MAX_ATTITUDE: float = 30.0   # degrees
    MIN_SAFE_RATE: float = 0.1  # m/s (closing speed magnitude)
    MAX_SAFE_RATE: float = 0.25   # m/s (closing speed magnitude)
    NEAR_DISTANCE: float = 5.0   # metres
    ATTITUDE_KEYS: tuple[str, ...] = ("roll", "yaw", "pitch")
    ACTION_MAP: dict[int, dict[int, str]] = {
        0: {1: "translate_forward", 2: "translate_backward"},
        1: {1: "translate_up", 2: "translate_down"},
        2: {1: "translate_right", 2: "translate_left"},
        3: {1: "roll_right", 2: "roll_left"},
        4: {1: "pitch_up", 2: "pitch_down"},
        5: {1: "yaw_right", 2: "yaw_left"},
    }
    SIMPLE_BAND: float = 0.2
    BAND_REWARD: float = 0.3
    BAND_PENALTY_GAIN: float = 0.04

    RATE_BAND_LOW: float = -0.23
    RATE_BAND_HIGH: float = -0.15
    RATE_BAND_EPS: float = 1e-6
    RATE_BAND_REWARD: float = 0.5
    RATE_MISS_PENALTY_GAIN: float = 0.6

    X_ABS_REWARD_GAIN: float = 8.0
    RANGE_ABS_REWARD_GAIN: float = 10.0
    X_PROGRESS_GAIN: float = 0.08
    RANGE_PROGRESS_GAIN: float = 0.1
    X_INCREASE_EXTRA_PENALTY_GAIN: float = 0.12
    RANGE_INCREASE_EXTRA_PENALTY_GAIN: float = 0.15

    LATERAL_FAIL_LIMIT: float = 35.0

    def __init__(
        self,
        step_delay: float = 0.5,
        max_steps: int | None = None,
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
        self.fuel_remaining: float = self.INITIAL_FUEL
        self._prev_state = {}
        self._prev_action: np.ndarray = np.zeros(6, dtype=np.int8)
        self._sim = TrainDockingSimulator(dt=self.dt)
        self.state_vars: dict[str, float] = {}

    def reset(self, *, seed=None):
        """Reset environment and simulator, returning initial observation.

        Resets internal episode counters, simulator state and returns the
        initial observation array plus an empty info dict, matching
        Gymnasium's API (obs, info).
        """
        super().reset(seed=seed)
        self._steps = 0
        self._prev_action.fill(0)
        self._sim.reset(self.np_random)
        self._sync_from_sim()

        obs = self._get_obs()
        self._prev_state = self._obs_to_dict(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        """Apply `action`, advance the simulator, compute reward and info.

        The action is a 6-dim MultiDiscrete vector where each element is
        0=no-op, 1=positive button, 2=negative button. This method drives
        the simulator, computes shaped rewards, terminal conditions and
        returns `(obs, reward, terminated, truncated, info)`.
        """
        for dim, act_val_raw in enumerate(action):
            act_val = int(act_val_raw)
            if act_val in (1, 2):
                action_name = self.ACTION_MAP[dim][act_val]
                self._sim.click_action(action_name)

        self._sync_from_sim(drive=True)
        self._steps += 1

        button_presses = int(self._sim.button_presses)

        obs = self._get_obs()
        state = self._obs_to_dict(obs)

        # =========================================================
        # REWARD COMPUTATION
        # =========================================================
        reward_components: dict[str, float] = {}

        # 1) For all state channels except (rate, x, range):
        #    inside [-0.2, 0.2] => reward, outside => penalty.
        band_metrics = (
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "roll_rate",
            "pitch_rate",
            "yaw_rate",
        )
        for key in band_metrics:
            abs_val = abs(float(state[key]))
            if abs_val <= self.SIMPLE_BAND:
                self._add_reward_component(reward_components, f"band_hit_{key}", self.BAND_REWARD)
            else:
                self._add_reward_component(
                    reward_components,
                    f"band_miss_{key}",
                    -((abs_val - self.SIMPLE_BAND) * self.BAND_PENALTY_GAIN),
                )

        # 2) rate: reward only when inside [-0.23, -0.15], otherwise penalty.
        rate_val = float(state["rate"])
        if (self.RATE_BAND_LOW - self.RATE_BAND_EPS) <= rate_val <= (self.RATE_BAND_HIGH + self.RATE_BAND_EPS):
            self._add_reward_component(reward_components, "rate_hit_band", self.RATE_BAND_REWARD)
        else:
            # Penalize by distance to the nearest edge of the valid band.
            rate_band_dist = (
                self.RATE_BAND_LOW - rate_val
                if rate_val < self.RATE_BAND_LOW
                else rate_val - self.RATE_BAND_HIGH
            )
            self._add_reward_component(
                reward_components,
                "rate_miss_band",
                -(rate_band_dist * self.RATE_MISS_PENALTY_GAIN),
            )

        # 3) x and range: smaller is better; if larger than previous step, penalize.
        x_curr = abs(float(state["x"]))
        x_prev = abs(float(self._prev_state["x"]))
        range_curr = max(0.0, float(state["range"]))
        range_prev = max(0.0, float(self._prev_state["range"]))

        self._add_reward_component(
            reward_components,
            "x_small_bonus",
            self.X_ABS_REWARD_GAIN / (1.0 + x_curr),
        )
        self._add_reward_component(
            reward_components,
            "range_small_bonus",
            self.RANGE_ABS_REWARD_GAIN / (1.0 + range_curr),
        )

        x_delta = x_curr - x_prev
        range_delta = range_curr - range_prev

        self._add_reward_component(
            reward_components,
            "x_progress",
            -(x_delta * self.X_PROGRESS_GAIN),
        )
        self._add_reward_component(
            reward_components,
            "range_progress",
            -(range_delta * self.RANGE_PROGRESS_GAIN),
        )

        if x_delta > 0.0:
            self._add_reward_component(
                reward_components,
                "x_increase_penalty",
                -(x_delta * self.X_INCREASE_EXTRA_PENALTY_GAIN),
            )
        if range_delta > 0.0:
            self._add_reward_component(
                reward_components,
                "range_increase_penalty",
                -(range_delta * self.RANGE_INCREASE_EXTRA_PENALTY_GAIN),
            )

        current_range = state["range"]
        progress_component_scores: dict[str, float] = {}
        noop_component_scores: dict[str, float] = {}

        # 5. Terminal Conditions
        terminated = False
        truncated = False
        success = False

        if self._is_docked(state):
            # Very large positive terminal bonus on success to ensure the
            # sparse objective dominates local shaping when docking occurs.
            self._add_reward_component(reward_components, "terminal_success", 1000.0)
            terminated = True
            success = True
        elif self.fuel_remaining <= 0.0:
            self._add_reward_component(reward_components, "terminal_fuel_empty", -300.0)
            terminated = True
        elif abs(state["y"]) > self.LATERAL_FAIL_LIMIT or abs(state["z"]) > self.LATERAL_FAIL_LIMIT:
            self._add_reward_component(reward_components, "terminal_lateral_limit", -1000.0)
            terminated = True
        elif current_range > self.MAX_RANGE:
            self._add_reward_component(reward_components, "terminal_range_limit", -1000.0)
            terminated = True
        elif any(abs(state[k]) > self.MAX_ATTITUDE for k in self.ATTITUDE_KEYS):
            self._add_reward_component(reward_components, "terminal_attitude_limit", -1000.0)
            terminated = True
        elif self.max_steps is not None and self._steps >= self.max_steps:
            truncated = True

        reward = float(sum(reward_components.values()))

        self._prev_state = state
        for dim in range(6):
            v = int(action[dim])
            self._prev_action[dim] = v if v in (1, 2) else 0

        info = {
            "steps": self._steps,
            "fuel_used": int(self.fuel_used),
            "fuel_remaining": float(self.fuel_remaining),
            "success": success,
            "button_presses": int(button_presses),
            "translation_pending_pulses": int(len(self._sim.translation_pending)),
            "reward_components": reward_components,
            "progress_component_scores": progress_component_scores,
            "noop_component_scores": noop_component_scores,
            **state,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        # No external resources to release for the pure-Python env.
        pass

    def _sync_from_sim(self, drive: bool = False) -> None:
        """Update environment-visible `state_vars` from the simulator.

        If `drive` is True, advance the simulator by one step; otherwise
        just obtain a snapshot without advancing time.
        """
        self.state_vars = self._sim.read_state() if drive else self._sim.get_state_snapshot()
        self.fuel_used = self._sim.fuel_used
        self.fuel_remaining = self._sim.fuel_remaining

    def _get_obs(self) -> np.ndarray:
        """Return clipped numpy observation vector matching `OBS_KEYS`.

        The last observation element is normalized remaining fuel in [0,1].
        """
        obs = np.array(
            [self.state_vars[k] for k in self.OBS_KEYS if k != "fuel"] + [self.fuel_remaining / self.INITIAL_FUEL],
            dtype=np.float32,
        )
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    @staticmethod
    def _add_reward_component(components: dict[str, float], key: str, value: float) -> None:
        components[key] = components.get(key, 0.0) + float(value)

    @staticmethod
    def _is_docked(state: dict[str, float]) -> bool:
        return (
            abs(state["x"]) <= TrainIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["y"]) <= TrainIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["z"]) <= TrainIssDockingEnv.GOOD_POS_THRESHOLD
            and abs(state["roll"]) <= TrainIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["pitch"]) <= TrainIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["yaw"]) <= TrainIssDockingEnv.GOOD_ATTITUDE_THRESHOLD
            and abs(state["roll_rate"]) <= TrainIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and abs(state["pitch_rate"]) <= TrainIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and abs(state["yaw_rate"]) <= TrainIssDockingEnv.GOOD_ANG_RATE_THRESHOLD
            and -TrainIssDockingEnv.MAX_SAFE_RATE <= state["rate"] <= -TrainIssDockingEnv.MIN_SAFE_RATE
            and state["range"] < TrainIssDockingEnv.GOOD_RANGE_THRESHOLD
        )

