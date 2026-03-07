"""Quick debug: initialize one env and take random actions. No policy server needed.

Usage:
    uv run python scripts/debug_eval_env.py
    POLARIS_CUDA_MEMORY_FRACTION=0.45 uv run python scripts/debug_eval_env.py --env DROID-BlockStackKitchen
"""

from __future__ import annotations

import argparse
import os
import time

import torch

# Must run before any CUDA allocation (e.g. when sharing GPU with policy server)
if "POLARIS_CUDA_MEMORY_FRACTION" in os.environ:
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ["POLARIS_CUDA_MEMORY_FRACTION"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="DROID-BlockStackKitchen")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()

    print(f"=== Debug env: {args.env} ===")
    print(f"GPU: {args.gpu_id}, Headless: {args.headless}, Steps: {args.steps}")
    print()

    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print()

    # >>>> Isaac Sim App Launcher (MUST happen before isaaclab imports) <<
    print("Launching Isaac Sim app...")
    t0 = time.time()
    from isaaclab.app import AppLauncher
    launcher_parser = argparse.ArgumentParser()
    launcher_args, _ = launcher_parser.parse_known_args()
    launcher_args.enable_cameras = True
    launcher_args.headless = args.headless
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app
    print(f"App launched in {time.time() - t0:.1f}s")
    print()
    # >>>> Isaac Sim App Launcher <<

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )

    # Create env
    print(f"Creating env: {args.env}...")
    t0 = time.time()
    env_cfg = parse_env_cfg(
        args.env,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(args.env, cfg=env_cfg)
    print(f"Env created in {time.time() - t0:.1f}s")
    print()

    # Get action space info
    action_space = env.action_space
    print(f"Action space: {action_space}")
    print()

    # Reset
    print("Resetting env...")
    obs, info = env.reset()
    print(f"Obs keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}")
    print()

    # Random actions
    print(f"Taking {args.steps} random actions...")
    for i in range(args.steps):
        action = torch.tensor(action_space.sample()).reshape(1, -1)
        obs, reward, term, trunc, info = env.step(action)
        print(f"  Step {i+1}: reward={reward}, term={term}, trunc={trunc}")
        if term[0] or trunc[0]:
            print("  Env done, resetting...")
            obs, info = env.reset()

    print()
    print("=== Debug complete ===")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()