"""
Evaluation script for the SpaceX ISS Docking Simulator.

Loads a trained PPO model and runs it on the browser-based SpaceX ISS Docking
Simulator in deterministic mode, then prints per-episode and aggregate stats.
"""

import argparse
import logging

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from docking import IssDockingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(
    model_path: str,
    n_episodes: int,
    launch_browser: bool,
    headless: bool,
) -> None:
    env = IssDockingEnv(launch_browser=launch_browser, headless=headless)
    stats_path = model_path + "_vec_normalize.pkl"
    vec_env = VecNormalize.load(stats_path, DummyVecEnv([lambda: env]))
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    episode_rewards: list[float] = []
    successes = 0

    for episode in range(n_episodes):
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            total_reward += rewards[0]
            done = dones[0]
            info = infos[0]

        episode_rewards.append(total_reward)
        if info.get("success", False):
            successes += 1

        print(
            f"Episode {episode + 1:3d}/{n_episodes}: "
            f"reward={total_reward:8.2f}  "
            f"range={info.get('range', 0.0):.2f} m  "
            f"rate={info.get('rate', 0.0):.3f} m/s  "
            f"steps={info.get('steps', 0)}"
        )

    vec_env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Episodes     : {n_episodes}")
    print(f"Success rate : {successes / n_episodes * 100:.1f}%")
    print(f"Mean reward  : {np.mean(episode_rewards):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on the SpaceX ISS Docking Simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/ppo_docking",
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes to run.",
    )
    parser.add_argument(
        "--launch-browser",
        action="store_true",
        help=(
            "Let Playwright launch a Chromium browser automatically."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser without a visible window.",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        launch_browser=args.launch_browser,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
