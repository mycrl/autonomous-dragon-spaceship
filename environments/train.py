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
    MIN_SAFE_RATE: float = 0.02  # m/s
    MAX_SAFE_RATE: float = 0.2   # m/s
    NEAR_DISTANCE: float = 5.0   # metres
    ATTITUDE_KEYS: tuple[str, ...] = ("roll", "yaw", "pitch")
    HARD_START_PROB: float = 0.35
    TRANSLATION_EFFECT_DELAY_STEPS: int = 3
    TRANSLATION_OBSERVE_WINDOW_STEPS: int = 4
    TRANSLATION_FIRST_PULSE_SCALE: float = 0.6
    TRANSLATION_SECOND_PULSE_SCALE: float = 0.85
    TRANSLATION_REVERSE_SCALE: float = 0.5
    TRANSLATION_QUICK_REPEAT_PENALTY: float = 0.12
    TRANSLATION_DIRECTION_FLIP_PENALTY: float = 0.16
    METRIC_REWARD_WEIGHTS: dict[str, float] = {
        "x": 0.9,
        "y": 0.9,
        "z": 0.9,
        "roll": 1.0,
        "pitch": 1.0,
        "yaw": 1.0,
        "roll_rate": 1.1,
        "pitch_rate": 1.1,
        "yaw_rate": 1.1,
        "rate": 1.4,
        "range": 1.2,
    }

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
        self._translation_pending: list[tuple[int, np.ndarray, int]] = []
        self._translation_last_command_step = np.full(3, -10_000, dtype=np.int32)
        self._translation_last_command_value = np.zeros(3, dtype=np.int8)
        self._translation_command_streak = np.zeros(3, dtype=np.int16)

        # Defines impulses for each action
        # Translating imparts velocity. Rotating imparts angular velocity.
        self.TRANSLATION_PULSE = 0.06  # m/s per click
        self.ROTATION_PULSE = 0.1      # deg/s per click

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        self.fuel_used = 0
        self.fuel_remaining = self.INITIAL_FUEL
        self._prev_action.fill(0)
        self._translation_pending.clear()
        self._translation_last_command_step.fill(-10_000)
        self._translation_last_command_value.fill(0)
        self._translation_command_streak.fill(0)

        # Domain-randomized spawn: include both near-aligned and strongly
        # off-axis starts so policy does not overfit to one reset pattern.
        hard_start = bool(self.np_random.random() < self.HARD_START_PROB)

        if hard_start:
            x = self.np_random.uniform(90.0, 260.0)
            y = self.np_random.uniform(-90.0, 90.0)
            z = self.np_random.uniform(-90.0, 90.0)
            vx = self.np_random.uniform(-0.12, 0.08)
            vy = self.np_random.uniform(-0.08, 0.08)
            vz = self.np_random.uniform(-0.08, 0.08)

            roll = self.np_random.uniform(-25.0, 25.0)
            pitch = self.np_random.uniform(-25.0, 25.0)
            yaw = self.np_random.uniform(-25.0, 25.0)
            roll_rate = self.np_random.uniform(-0.7, 0.7)
            pitch_rate = self.np_random.uniform(-0.7, 0.7)
            yaw_rate = self.np_random.uniform(-0.7, 0.7)
        else:
            x = self.np_random.uniform(170.0, 230.0)
            y = self.np_random.uniform(-25.0, 25.0)
            z = self.np_random.uniform(-25.0, 25.0)
            vx = self.np_random.uniform(-0.05, 0.03)
            vy = self.np_random.uniform(-0.03, 0.03)
            vz = self.np_random.uniform(-0.03, 0.03)

            roll = self.np_random.uniform(-18.0, 18.0)
            pitch = self.np_random.uniform(-18.0, 18.0)
            yaw = self.np_random.uniform(-18.0, 18.0)
            roll_rate = self.np_random.uniform(-0.3, 0.3)
            pitch_rate = self.np_random.uniform(-0.3, 0.3)
            yaw_rate = self.np_random.uniform(-0.3, 0.3)

        self.state_vars = {
            "x": x,
            "vx": vx,
            "y": y,
            "vy": vy,
            "z": z,
            "vz": vz,
            "roll": roll,
            "roll_rate": roll_rate,
            "pitch": pitch,
            "pitch_rate": pitch_rate,
            "yaw": yaw,
            "yaw_rate": yaw_rate,
        }

        self.state_vars["range"] = math.sqrt(x**2 + y**2 + z**2)
        if self.state_vars["range"] > 1e-6:
            radial_velocity = (x * vx + y * vy + z * vz) / self.state_vars["range"]
        else:
            radial_velocity = 0.0
        self.state_vars["rate"] = radial_velocity

        obs = self._get_obs()
        self._prev_state = self._obs_to_dict(obs)
        return obs, {}

    def step(self, action: np.ndarray):
        step_idx = self._steps
        button_presses = 0
        active_dims = set()
        quick_repeat_translation_dims: set[int] = set()
        flip_translation_dims: set[int] = set()
        
        # Process actions (apply impulses)
        for dim, act_val in enumerate(action):
            act_val = int(act_val)
            if act_val in (1, 2):
                button_presses += 1
                active_dims.add(dim)
                
                direction = 1 if act_val == 1 else -1

                if dim in (0, 1, 2):
                    axis_idx = dim
                    since_last_cmd = step_idx - int(self._translation_last_command_step[axis_idx])
                    prev_cmd = int(self._translation_last_command_value[axis_idx])

                    if 0 <= since_last_cmd <= self.TRANSLATION_OBSERVE_WINDOW_STEPS:
                        quick_repeat_translation_dims.add(axis_idx)

                    recent_flip = (
                        prev_cmd in (1, 2)
                        and prev_cmd != act_val
                        and 0 <= since_last_cmd <= self.TRANSLATION_OBSERVE_WINDOW_STEPS
                    )
                    if recent_flip:
                        flip_translation_dims.add(axis_idx)

                    if prev_cmd == act_val:
                        self._translation_command_streak[axis_idx] += 1
                    else:
                        self._translation_command_streak[axis_idx] = 1

                    self._translation_last_command_step[axis_idx] = step_idx
                    self._translation_last_command_value[axis_idx] = act_val

                    pulse_scale = 1.0
                    if self._translation_command_streak[axis_idx] == 1:
                        pulse_scale *= self.TRANSLATION_FIRST_PULSE_SCALE
                    elif self._translation_command_streak[axis_idx] == 2:
                        pulse_scale *= self.TRANSLATION_SECOND_PULSE_SCALE
                    if recent_flip:
                        pulse_scale *= self.TRANSLATION_REVERSE_SCALE

                    # Translation commands are defined in spacecraft body frame.
                    # Projecting them into world frame creates realistic axis
                    # coupling when attitude is not perfectly aligned.
                    if dim == 0:      # Forward / backward
                        body_vec = np.array([-direction, 0.0, 0.0], dtype=np.float32)
                    elif dim == 1:    # Up / down
                        body_vec = np.array([0.0, 0.0, direction], dtype=np.float32)
                    else:             # Right / left
                        body_vec = np.array([0.0, direction, 0.0], dtype=np.float32)

                    world_vec = self._body_to_world(
                        body_vec,
                        roll_deg=self.state_vars["roll"],
                        pitch_deg=self.state_vars["pitch"],
                        yaw_deg=self.state_vars["yaw"],
                    )
                    delta_v = world_vec * (self.TRANSLATION_PULSE * pulse_scale)
                    self._translation_pending.append(
                        (
                            self.TRANSLATION_EFFECT_DELAY_STEPS,
                            delta_v.astype(np.float32),
                            axis_idx,
                        )
                    )
                elif dim == 3: # Roll Right(1)/Left(2) -> right increases displayed roll_rate
                    self.state_vars["roll_rate"] += direction * self.ROTATION_PULSE
                elif dim == 4: # Pitch Up(1)/Down(2) -> up decreases displayed pitch_rate
                    self.state_vars["pitch_rate"] -= direction * self.ROTATION_PULSE
                elif dim == 5: # Yaw Right(1)/Left(2) -> right increases displayed yaw_rate
                    self.state_vars["yaw_rate"] += direction * self.ROTATION_PULSE
        if button_presses > 0:
            fuel_spent = button_presses * self.FUEL_PER_BUTTON
            self.fuel_used += int(fuel_spent)
            self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_spent)

        self._steps += 1

        # Delayed translation actuation: pulses affect velocity after a short lag.
        pending_next: list[tuple[int, np.ndarray, int]] = []
        for wait_steps, delta_v, axis_idx in self._translation_pending:
            if wait_steps <= 0:
                self.state_vars["vx"] += float(delta_v[0])
                self.state_vars["vy"] += float(delta_v[1])
                self.state_vars["vz"] += float(delta_v[2])
            else:
                pending_next.append((wait_steps - 1, delta_v, axis_idx))
        self._translation_pending = pending_next

        # Euler integrate Physics!
        # Save range for rate calculation
        old_range = self.state_vars["range"]

        # Positions
        self.state_vars["x"] += self.state_vars["vx"] * self.dt
        self.state_vars["y"] += self.state_vars["vy"] * self.dt
        self.state_vars["z"] += self.state_vars["vz"] * self.dt
        
        # In the browser simulator UI, displayed attitude angle changes opposite
        # to displayed angular-rate sign for all rotational axes.
        self.state_vars["roll"] -= self.state_vars["roll_rate"] * self.dt
        self.state_vars["pitch"] -= self.state_vars["pitch_rate"] * self.dt
        self.state_vars["yaw"] -= self.state_vars["yaw_rate"] * self.dt

        # Recalculate Range & Rate
        new_range = math.sqrt(self.state_vars["x"]**2 + self.state_vars["y"]**2 + self.state_vars["z"]**2)
        self.state_vars["range"] = new_range
        self.state_vars["rate"] = (new_range - old_range) / self.dt

        obs = self._get_obs()
        state = self._obs_to_dict(obs)

        # =========================================================
        # REWARD COMPUTATION
        # =========================================================
        reward_components: dict[str, float] = {}

        # 1. Action/Fuel penalty (per control dimension)
        for dim in active_dims:
            self._add_reward_component(reward_components, f"fuel_dim_{dim}", -0.03)

        for dim in sorted(quick_repeat_translation_dims):
            self._add_reward_component(
                reward_components,
                f"translation_quick_repeat_dim{dim}",
                -self.TRANSLATION_QUICK_REPEAT_PENALTY,
            )

        for dim in sorted(flip_translation_dims):
            self._add_reward_component(
                reward_components,
                f"translation_flip_dim{dim}",
                -self.TRANSLATION_DIRECTION_FLIP_PENALTY,
            )

        progress_component_scores: dict[str, float] = {}
        noop_component_scores: dict[str, float] = {}

        # Local per-dimension credit: each action only affects its own mapped metrics.
        dim_to_metrics: dict[int, tuple[str, ...]] = {
            0: ("range", "rate"),
            1: ("z",),
            2: ("y", "x"),
            3: ("roll", "roll_rate"),
            4: ("pitch", "pitch_rate"),
            5: ("yaw", "yaw_rate"),
        }

        for dim, metrics in dim_to_metrics.items():
            act_val = int(action[dim])
            is_active = act_val in (1, 2)
            prev_same_dir = bool(self._prev_action[dim] == act_val and is_active)

            for metric in metrics:
                improvement = self._metric_improvement(metric, self._prev_state, state)
                weight = self.METRIC_REWARD_WEIGHTS.get(metric, 1.0)
                progress_score = float(np.clip(improvement * weight, -0.8, 0.8))
                progress_component_scores[f"dim{dim}_{metric}"] = progress_score

                if is_active:
                    # Translation has delayed response, so immediate reward for
                    # translation actions is intentionally softened.
                    local_progress = progress_score * (0.35 if dim < 3 else 1.0)
                    self._add_reward_component(
                        reward_components,
                        f"active_dim{dim}_{metric}",
                        local_progress,
                    )
                    # If metric already improving, repeating same direction is over-excited.
                    if improvement > 0.0 and prev_same_dir:
                        repeat_penalty = -0.08 if dim < 3 else -0.12
                        self._add_reward_component(
                            reward_components,
                            f"repeat_dim{dim}_{metric}",
                            repeat_penalty,
                        )
                    if improvement <= 0.0 and dim >= 3:
                        self._add_reward_component(
                            reward_components,
                            f"ineffective_dim{dim}_{metric}",
                            -0.06,
                        )
                else:
                    violation = self._metric_violation(metric, state)
                    observe_window = (
                        dim < 3
                        and 0 < (step_idx - int(self._translation_last_command_step[dim]))
                        <= self.TRANSLATION_OBSERVE_WINDOW_STEPS
                    )
                    if improvement > 0.0:
                        hold_reward = float(np.clip(improvement * 0.5, 0.0, 0.25))
                        noop_component_scores[f"dim{dim}_{metric}"] = hold_reward
                        self._add_reward_component(
                            reward_components,
                            f"hold_dim{dim}_{metric}",
                            hold_reward,
                        )
                        if observe_window:
                            observe_reward = float(np.clip(improvement * 0.8, 0.0, 0.2))
                            self._add_reward_component(
                                reward_components,
                                f"observe_dim{dim}_{metric}",
                                observe_reward,
                            )
                    elif violation > 0.0:
                        lazy_penalty = -float(np.clip((violation + (-improvement)) * 0.35, 0.0, 0.25))
                        noop_component_scores[f"dim{dim}_{metric}"] = lazy_penalty
                        self._add_reward_component(
                            reward_components,
                            f"lazy_dim{dim}_{metric}",
                            lazy_penalty,
                        )

        # 4. Safety Constraints & Violations (Extreme Penalties)
        current_range = state["range"]
        if current_range < self.NEAR_DISTANCE and state["rate"] > self.MAX_SAFE_RATE:
            self._add_reward_component(reward_components, "near_overspeed", -10.0)

        # Global approach-rate safety shaping: regardless of distance, heavily
        # discourage unsafe closing speed beyond configured limit.
        # Per requested rule: rate can be negative physically, but negative means backing away and is penalized.
        if state["rate"] < 0.0:
            reverse_speed = -state["rate"]
            self._add_reward_component(
                reward_components,
                "rate_reverse",
                -((reverse_speed) * 30.0) ** 2,
            )
        elif state["rate"] < self.MIN_SAFE_RATE:
            under_speed = self.MIN_SAFE_RATE - state["rate"]
            self._add_reward_component(
                reward_components,
                "rate_under",
                -((under_speed) * 20.0) ** 2,
            )
        elif state["rate"] > self.MAX_SAFE_RATE:
            overspeed = state["rate"] - self.MAX_SAFE_RATE
            self._add_reward_component(
                reward_components,
                "rate_overspeed",
                -((overspeed) * 30.0) ** 2,
            )

        if current_range > 15.0 and 0.0 <= state["rate"] < self.MIN_SAFE_RATE:
            self._add_reward_component(reward_components, "far_stagnation", -0.1)

        # c) Angular-rate target-band shaping (per-axis, no global coupling).
        axis_to_rate = {
            "roll": "roll_rate",
            "pitch": "pitch_rate",
            "yaw": "yaw_rate",
        }
        for axis, rate_key in axis_to_rate.items():
            target_rate = float(np.clip(abs(state[axis]) * 0.02, 0.0, self.GOOD_ANG_RATE_THRESHOLD))
            delta = abs(abs(state[rate_key]) - target_rate)
            self._add_reward_component(reward_components, f"angular_target_{axis}", -delta * 0.8)

        # Keep hard punishment only for clearly unsafe spin rates.
        for key in ("roll_rate", "yaw_rate", "pitch_rate"):
            rate_val = abs(state[key])
            if rate_val > self.GOOD_ANG_RATE_THRESHOLD:
                self._add_reward_component(
                    reward_components,
                    f"spin_overspeed_{key}",
                    -((rate_val - self.GOOD_ANG_RATE_THRESHOLD) * 12.0) ** 2,
                )

        # 5. Terminal Conditions
        terminated = False
        truncated = False
        success = False

        if self._is_docked(state):
            self._add_reward_component(reward_components, "terminal_success", 1000.0)
            terminated = True
            success = True
        elif self.fuel_remaining <= 0.0:
            self._add_reward_component(reward_components, "terminal_fuel_empty", -300.0)
            terminated = True
        elif state["rate"] > 0.8:
            self._add_reward_component(reward_components, "terminal_rate_overspeed", -500.0)
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
            "translation_pending_pulses": int(len(self._translation_pending)),
            "reward_components": reward_components,
            "progress_component_scores": progress_component_scores,
            "noop_component_scores": noop_component_scores,
            **state,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        pass # No browser to close!

    def _get_obs(self) -> np.ndarray:
        obs = np.array(
            [self.state_vars[k] for k in self.OBS_KEYS if k != "fuel"] + [self.fuel_remaining / self.INITIAL_FUEL],
            dtype=np.float32,
        )
        return np.clip(obs, -self.OBS_HIGH, self.OBS_HIGH)

    def _obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return dict(zip(self.OBS_KEYS, obs.tolist()))

    def _metric_violation(self, key: str, state: dict[str, float]) -> float:
        if key in ("x", "y", "z"):
            return max(0.0, abs(state[key]) - self.GOOD_POS_THRESHOLD)
        if key in ("roll", "pitch", "yaw"):
            return max(0.0, abs(state[key]) - self.GOOD_ATTITUDE_THRESHOLD)
        if key in ("roll_rate", "pitch_rate", "yaw_rate"):
            return max(0.0, abs(state[key]) - self.GOOD_ANG_RATE_THRESHOLD)
        if key == "range":
            return max(0.0, state["range"] - self.GOOD_RANGE_THRESHOLD)
        if key == "rate":
            return max(0.0, self.MIN_SAFE_RATE - state["rate"]) + max(0.0, state["rate"] - self.MAX_SAFE_RATE)
        return 0.0

    def _metric_improvement(
        self,
        key: str,
        prev_state: dict[str, float],
        curr_state: dict[str, float],
    ) -> float:
        return self._metric_violation(key, prev_state) - self._metric_violation(key, curr_state)

    @staticmethod
    def _add_reward_component(components: dict[str, float], key: str, value: float) -> None:
        components[key] = components.get(key, 0.0) + float(value)

    @staticmethod
    def _body_to_world(
        body_vec: np.ndarray,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> np.ndarray:
        """Rotate a body-frame vector into world frame using Z-Y-X Euler order."""
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)

        cr = math.cos(roll)
        sr = math.sin(roll)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        rx = np.array(
            [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
            dtype=np.float32,
        )
        ry = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
            dtype=np.float32,
        )
        rz = np.array(
            [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        rot = rz @ ry @ rx
        return rot @ body_vec

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
            and TrainIssDockingEnv.MIN_SAFE_RATE <= state["rate"] <= TrainIssDockingEnv.MAX_SAFE_RATE
            and state["range"] < TrainIssDockingEnv.GOOD_RANGE_THRESHOLD
        )

