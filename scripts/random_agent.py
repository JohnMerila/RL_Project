"""Finite random-action smoke test for the navigation environment."""

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-DiffDrive-Lidar-Nav-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--action_mode", choices=("random", "forward", "turn", "stop"), default="random")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import diffdrive_nav.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    print("[INFO] Creating Gym environment...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[INFO] Gym environment created.", flush=True)
    print("[INFO] Resetting environment...", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment reset complete.", flush=True)
    print(f"[INFO] policy observations: {tuple(obs['policy'].shape)}", flush=True)
    print(f"[INFO] action space: {env.action_space}", flush=True)
    for step in range(args_cli.steps):
        with torch.inference_mode():
            if args_cli.action_mode == "random":
                actions = 2.0 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1.0
            else:
                actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                actions[:, 0] = 1.0 if args_cli.action_mode == "forward" else -1.0
                actions[:, 1] = 1.0 if args_cli.action_mode == "turn" else 0.0
            obs, rewards, terminated, truncated, _ = env.step(actions)
        if step % 100 == 0:
            print(
                f"[INFO] step={step} reward={rewards.mean().item():.3f} "
                f"done={(terminated | truncated).sum().item()}",
                flush=True,
            )
    local_position = env.unwrapped._robot.data.root_pos_w[:, :2] - env.unwrapped.scene.env_origins[:, :2]
    drive_wheel_velocity = env.unwrapped._robot.data.joint_vel[:, env.unwrapped._wheel_joint_ids]
    print(f"[INFO] final mean local xy: {local_position.mean(dim=0).tolist()}", flush=True)
    print(
        f"[INFO] final mean body vx: {env.unwrapped._robot.data.root_lin_vel_b[:, 0].mean().item():.3f} m/s; "
        f"yaw rate: {env.unwrapped._robot.data.root_ang_vel_b[:, 2].mean().item():.3f} rad/s; "
        f"drive wheel rad/s: {drive_wheel_velocity.mean(dim=0).tolist()}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # Isaac Sim replaces sys.excepthook, so print smoke-test failures explicitly.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
