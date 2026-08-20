# TLDR

This project trains a policy for a differential drive robot in IsaacLab that is tested via transfer learning on a physical robot

Demo video on sim
https://youtu.be/qy_7rYX2ZnI

Demo video on physical robot
https://youtube.com/shorts/kGsl8ITXeks

Link to robot Github:
https://github.com/JohnMerila/EDUBot_RL

## Actions

A linear and angular velocity are applied to the robot in m/s and radians/s


## Observations

scan - 72 bins with the minimum distance in the 5° window clamped to the max_lidar_range

goal - normalized and consisting of the following:
dist = goal.clamp(max_lidar_range)/max_lidar_range
sin(bearing)
cos(bearing)

velocity - linear and angular velocity normalized by the maximum speed and clamped between -2.0 and 2.0

current action

## Rewards

The reward is comprised of the progress towards the goal, a reward when the goal is reached, a collision penalty, a penalty increasing as the distance to obstacles are reduced, a penalty for changing actions, a penalty for angular speed, and a reward for living.

These rewards are summed to produce the supplied rward



## The following is a more detailed description and overview





# Differential-Drive LiDAR Navigation in Isaac Lab

This project trains a local point-goal navigation policy for a differential-drive robot in Isaac Lab. Every episode
randomizes the robot heading, goal, obstacle count, obstacle positions, and obstacle orientations. The policy receives
a 360-degree LiDAR scan, relative goal, measured base velocity, and its previous action; it outputs normalized linear
and angular velocity commands.

The implementation targets the Isaac Lab checkout already installed on this machine (`v2.3.2` at
`/home/megrad/IsaacLab`) and uses RSL-RL PPO. It includes a self-contained training robot URDF, so the first run does
not need the older JetBot USD or a Nucleus asset server.

The longer design and sim-to-real discussion is in
[the existing Obsidian note](Obsidian1/RL_Project/Training%20a%20Differential-Drive%20LiDAR%20Navigation%20Policy%20in%20Isaac%20Lab.md).

## What is included

```text
.
├── policy_contract.yaml                 # exact deployed observation/action interface
├── scripts/
│   ├── random_agent.py                  # finite simulator smoke test
│   └── rsl_rl/
│       ├── train.py                     # PPO training/resume/video
│       └── play.py                      # checkpoint playback + ONNX/JIT export
└── source/diffdrive_nav/diffdrive_nav/
    ├── assets/differential_drive.urdf
    └── tasks/direct/diffdrive_nav/
        ├── diffdrive_nav_env_cfg.py     # environment and tuneable parameters
        ├── diffdrive_nav_env.py         # simulation, reset, observations, rewards
        └── agents/rsl_rl_ppo_cfg.py     # PPO/network configuration
```

Registered tasks:

- `Isaac-DiffDrive-Lidar-Nav-Direct-v0`: noisy, randomized training preset; 512 environments by default.
- `Isaac-DiffDrive-Lidar-Nav-Play-v0`: 16-environment playback preset with sensor noise disabled and a bright green
  goal disc. The disc radius is the configured success tolerance; it is visual-only and does not affect collisions.

## Setup

From this repository:

```bash
cd /home/megrad/RL_Project
/home/megrad/miniconda3/envs/env_isaaclab/bin/python -m pip install -e source/diffdrive_nav
```

The editable install only registers this local extension; it does not reinstall Isaac Lab or Isaac Sim. Re-run it if
the package is moved. The commands below name the `env_isaaclab` interpreter explicitly and therefore also work while
the shell prompt shows `(base)`. Alternatively, run `conda activate env_isaaclab` first and use
`/home/megrad/IsaacLab/isaaclab.sh -p` in place of the explicit Python path.

## Validate before training

Start small and headless:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/random_agent.py \
  --task Isaac-DiffDrive-Lidar-Nav-Direct-v0 \
  --num_envs 4 --steps 500 --headless
```

Then inspect one environment in the GUI:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/random_agent.py \
  --task Isaac-DiffDrive-Lidar-Nav-Play-v0 \
  --num_envs 1 --steps 2000
```

Validate the drive convention explicitly before a long run:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/random_agent.py \
  --task Isaac-DiffDrive-Lidar-Nav-Play-v0 \
  --num_envs 1 --steps 200 --action_mode forward --headless

/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/random_agent.py \
  --task Isaac-DiffDrive-Lidar-Nav-Play-v0 \
  --num_envs 1 --steps 200 --action_mode turn --headless
```

Check that positive forward commands move along the robot's +X direction, positive yaw commands turn
counterclockwise, reported LiDAR ranges match the wall/obstacle geometry, obstacle layouts change on reset, and the
chassis does not bounce or scrape the ground.

## Training regimen

The default configuration is intended as a useful baseline, not a promise that one run will converge. Use this staged
regimen and preserve checkpoints between stages.

### 1. Motion and goal-seeking smoke run

Temporarily set `min_obstacles = 0`, `max_obstacles = 0`, `lidar_noise_std = 0`, and
`lidar_dropout_probability = 0` in `diffdrive_nav_env_cfg.py`. Train 64 environments for 100-200 iterations:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Isaac-DiffDrive-Lidar-Nav-Direct-v0 \
  --num_envs 64 --max_iterations 200 --headless \
  --run_name stage1_empty
```

The policy should learn to reduce goal distance reliably. If not, inspect wheel direction, goal bearing, progress
reward, resets, and action saturation before increasing scale.

### 2. Sparse clutter

Set `min_obstacles = 1`, `max_obstacles = 4`, keep dropout off, and resume the latest stage-1 checkpoint:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Isaac-DiffDrive-Lidar-Nav-Direct-v0 \
  --num_envs 256 --max_iterations 600 --headless \
  --resume --load_run '.*stage1_empty' --checkpoint 'model_.*.pt' \
  --run_name stage2_sparse
```

### 3. Full randomized layouts

Restore the checked-in obstacle range of 2-8, train 512-1024 environments, and use at least three seeds:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/train.py \
  --task Isaac-DiffDrive-Lidar-Nav-Direct-v0 \
  --num_envs 512 --max_iterations 2000 --headless \
  --seed 42 --run_name full_seed42
```

Repeat with `--seed 43` and `--seed 44`. Watch success, collision, timeout, minimum clearance, every reward component,
and policy action saturation in TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/diffdrive_lidar_nav
```

### 4. Sensor and dynamics gap

After the nominal task succeeds, gradually expand `lidar_noise_std`, `lidar_dropout_probability`, acceleration limits,
friction, wheel radius/track uncertainty, and action latency. The checked-in task includes LiDAR noise/dropout and
command acceleration limiting; hardware-specific actuator delay and dynamics ranges should be added only after they
are measured on the target robot.

Do not maximize all randomization ranges at once. A policy that never learns the nominal task does not become robust
by making its training distribution harder.

## Play, evaluate, and export

`play.py` chooses the latest run/checkpoint unless one is supplied:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Isaac-DiffDrive-Lidar-Nav-Play-v0 \
  --num_envs 16
```

Select a checkpoint explicitly:

```bash
/home/megrad/miniconda3/envs/env_isaaclab/bin/python scripts/rsl_rl/play.py \
  --task Isaac-DiffDrive-Lidar-Nav-Play-v0 \
  --checkpoint /absolute/path/to/model_2000.pt \
  --num_envs 4 --video --video_length 1000 --headless
```

Playback exports `policy.pt` and `policy.onnx` into the run's `exported/` directory. Ship one of those models together
with `policy_contract.yaml` and the saved `params/env.yaml` and `params/agent.yaml` from the run.
Use `--steps N` for a finite headless evaluation; its default of `0` keeps interactive playback running.
The playback preset displays one bright green goal-tolerance disc per environment and moves it whenever a goal resets.
The training preset leaves these markers disabled to avoid viewport overhead.

## Policy contract

The actor input is a fixed 79-element `float32` vector:

| Slice | Values | Normalization |
|---|---|---|
| `0:72` | 360-degree LiDAR in 5-degree bins | clip to 8 m, divide by 8 m |
| `72:75` | goal distance, `sin(bearing)`, `cos(bearing)` | distance divided by 8 m |
| `75:77` | forward speed, yaw rate | divide by 0.8 m/s and 1.5 rad/s |
| `77:79` | previous normalized policy action | already in `[-1, 1]` |

The two actions map to:

```text
linear_mps  = 0.4 * (action[0] + 1)   # 0.0 to 0.8 m/s
angular_rps = 1.5 * action[1]         # -1.5 to 1.5 rad/s
```

The controller applies linear/angular acceleration limits before converting to wheel speed. See
`policy_contract.yaml` for the machine-readable form. Do not change observation order, normalization, LiDAR beam
convention, or action mapping after training and then reuse the same checkpoint.

For training throughput, the planar LiDAR is a tensorized ray/oriented-box intersection against the exact same wall
and randomized obstacle poses used by PhysX. It produces the 72 ideal range samples first, then applies range noise
and beam dropout. This is appropriate for the box-obstacle task and avoids one RTX sensor per parallel environment.
If you add non-box geometry, replace `_compute_lidar_ranges()` with Isaac Lab's multi-mesh ray caster or RTX LiDAR and
keep the policy contract unchanged.

## Parameters to tune

Environment and robot settings live in
`source/diffdrive_nav/diffdrive_nav/tasks/direct/diffdrive_nav/diffdrive_nav_env_cfg.py`.

| Parameter | Current value | Tune when |
|---|---:|---|
| `scene.num_envs` | 512 | Reduce for VRAM/startup; increase for throughput |
| `decimation`, `sim.dt` | 6, 1/120 s | Must match a 20 Hz deployed policy loop |
| `wheel_radius`, `track_width` | 0.055 m, 0.335 m | Replace with measured robot geometry |
| `max_linear_speed`, `max_angular_speed` | 0.8 m/s, 1.5 rad/s | Match safe actuator limits |
| `max_linear_accel`, `max_angular_accel` | 1.5 m/s², 3 rad/s² | Match measured step response/braking |
| `min_obstacles`, `max_obstacles` | 2, 8 | Curriculum and layout density |
| `min_goal_distance`, `max_goal_distance` | 1.5 m, 3.1 m | Curriculum and episode difficulty |
| `goal_tolerance` | 0.30 m | Match localization/control accuracy |
| `lidar_noise_std` | 0.015 m | Use recorded scan residuals |
| `lidar_dropout_probability` | 0.01 | Use measured invalid-return rate |
| `collision_lidar_range` | 0.19 m | Tune the LiDAR-origin safety threshold |
| `robot_collision_radius` | 0.20 m | Match a conservative circular footprint |
| `near_obstacle_distance` | 0.42 m | Increase for more conservative clearance |
| `rew_progress` | 8.0 | Increase if the robot stalls; inspect exploits |
| `rew_goal`, `rew_collision` | +20, -20 | Balance completion against safety |
| `rew_near_obstacle` | -0.35 | Too large can make the policy freeze |
| `rew_action_rate` | -0.08 | Increase magnitude for smoother commands |

PPO settings live in `agents/rsl_rl_ppo_cfg.py`. Tune learning rate, entropy, rollout length, network width, and batch
layout after validating environment behavior. If training oscillates, first try a lower learning rate (`1e-4`), then
reduce entropy. If the policy never explores useful turns, modestly raise entropy or initial action noise.

## Using a different robot

Replace `DIFFDRIVE_ROBOT_CFG` with an `ArticulationCfg` for the target USD/URDF, then update these as one unit:

1. `wheel_joint_names`, preserving `[left, right]` order;
2. `wheel_radius` and `track_width`;
3. the LiDAR offset;
4. robot start height, footprint radius/clearances, collision threshold, speed, and acceleration limits.

Run the one-environment GUI smoke test again before training. Keep body-frame +X forward, +Y left, +Z up, and positive
yaw counterclockwise, or adapt both simulation and deployment preprocessing consistently.

## Scope and limitations

This is a reactive local navigator. One LiDAR scan cannot guarantee a solution through loops, long cul-de-sacs, or
goals that require global route planning. For real deployments, feed reachable local waypoints from a global planner
and keep an independent emergency-stop/safety supervisor outside the learned policy.
