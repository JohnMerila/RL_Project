"""Direct Isaac Lab environment for LiDAR point-goal navigation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse

from .diffdrive_nav_env_cfg import DiffDriveNavEnvCfg


class DiffDriveNavEnv(DirectRLEnv):
    """Navigate a differential-drive robot to a local goal through randomized clutter."""

    cfg: DiffDriveNavEnvCfg

    def __init__(self, cfg: DiffDriveNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._wheel_joint_ids, wheel_names = self._robot.find_joints(
            list(self.cfg.wheel_joint_names), preserve_order=True
        )
        if tuple(wheel_names) != tuple(self.cfg.wheel_joint_names):
            raise RuntimeError(
                f"Expected wheel joints {self.cfg.wheel_joint_names}, resolved {wheel_names}. "
                "Update wheel_joint_names and the drive geometry together."
            )
        if self.cfg.lidar_num_rays != 72:
            raise RuntimeError(f"Policy contract requires 72 LiDAR rays; configured {self.cfg.lidar_num_rays}.")

        self._actions = torch.zeros((self.num_envs, 2), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._body_commands = torch.zeros_like(self._actions)
        self._wheel_targets = torch.zeros_like(self._actions)

        self._goal_positions_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_goal_distance = torch.zeros_like(self._goal_distance)
        self._lidar_ranges_m = torch.full(
            (self.num_envs, self.cfg.lidar_num_rays), self.cfg.lidar_max_range, device=self.device
        )
        self._minimum_clearance = torch.full((self.num_envs,), self.cfg.lidar_max_range, device=self.device)
        self._goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision = torch.zeros_like(self._goal_reached)
        self._out_of_bounds = torch.zeros_like(self._goal_reached)

        obstacle_radii = [0.5 * math.hypot(size[0], size[1]) for size in self.cfg.obstacle_sizes]
        self._obstacle_radii = torch.tensor(obstacle_radii, device=self.device)
        self._obstacle_positions_w = torch.zeros(
            (self.num_envs, len(self.cfg.obstacle_sizes), 2), device=self.device
        )
        self._obstacle_yaws = torch.zeros((self.num_envs, len(self.cfg.obstacle_sizes)), device=self.device)
        self._obstacle_active = torch.zeros(
            (self.num_envs, len(self.cfg.obstacle_sizes)), dtype=torch.bool, device=self.device
        )
        ray_angles = torch.arange(self.cfg.lidar_num_rays, device=self.device) * (
            2.0 * math.pi / self.cfg.lidar_num_rays
        ) - math.pi
        self._lidar_directions_b = torch.stack((torch.cos(ray_angles), torch.sin(ray_angles)), dim=-1)

        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "progress",
                "goal",
                "collision",
                "near_obstacle",
                "action_rate",
                "angular",
                "alive",
            )
        }

        if self.cfg.debug_vis:
            self.set_debug_vis(True)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._obstacles = RigidObjectCollection(self.cfg.obstacle_collection_cfg)

        ground_cfg = GroundPlaneCfg(
            size=(100.0, 100.0),
            color=(0.12, 0.12, 0.12),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        self._spawn_walls()

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_object_collections["obstacles"] = self._obstacles

        # Create playback markers before simulation startup so Fabric discovers their prototypes cleanly.
        if self.cfg.debug_vis:
            self._goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.82, 0.82))
        light_cfg.func("/World/DomeLight", light_cfg)

    def _spawn_walls(self):
        half = self.cfg.arena_half_extent
        thickness = self.cfg.wall_thickness
        height = 0.60
        wall_cfgs = (
            ("Wall_North", (0.0, half, height / 2), (2 * half + thickness, thickness, height)),
            ("Wall_South", (0.0, -half, height / 2), (2 * half + thickness, thickness, height)),
            ("Wall_East", (half, 0.0, height / 2), (thickness, 2 * half + thickness, height)),
            ("Wall_West", (-half, 0.0, height / 2), (thickness, 2 * half + thickness, height)),
        )
        for name, position, size in wall_cfgs:
            wall_cfg = sim_utils.CuboidCfg(
                size=size,
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.28, 0.30, 0.33)),
            )
            wall_cfg.func(f"/World/envs/env_0/{name}", wall_cfg, translation=position)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-1.0, 1.0))

        target_linear = 0.5 * self.cfg.max_linear_speed * (self._actions[:, 0] + 1.0)
        target_angular = self.cfg.max_angular_speed * self._actions[:, 1]
        target_commands = torch.stack((target_linear, target_angular), dim=-1)

        max_delta = torch.tensor(
            [self.cfg.max_linear_accel, self.cfg.max_angular_accel], device=self.device
        ) * self.step_dt
        delta = (target_commands - self._body_commands).clamp(-max_delta, max_delta)
        self._body_commands.add_(delta)

        linear = self._body_commands[:, 0]
        angular = self._body_commands[:, 1]
        self._wheel_targets[:, 0] = (
            linear - 0.5 * self.cfg.track_width * angular
        ) / self.cfg.wheel_radius
        self._wheel_targets[:, 1] = (
            linear + 0.5 * self.cfg.track_width * angular
        ) / self.cfg.wheel_radius

    def _apply_action(self):
        self._robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_joint_ids)

    def _update_task_state(self):
        robot_pos_w = self._robot.data.root_pos_w
        to_goal_w = self._goal_positions_w - robot_pos_w
        self._goal_distance = torch.linalg.vector_norm(to_goal_w[:, :2], dim=-1)

        self._lidar_ranges_m = self._compute_lidar_ranges()
        self._minimum_clearance = self._lidar_ranges_m.amin(dim=1)

    def _compute_lidar_ranges(self) -> torch.Tensor:
        """Cast planar rays analytically against the same boxes used by PhysX.

        A tensorized 2-D caster is substantially faster than RTX LiDAR for hundreds of training environments and,
        unlike a visual approximation, uses the exact randomized obstacle poses and sizes.
        """
        quat = self._robot.data.root_quat_w
        qw, qx, qy, qz = quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        offset_x, offset_y, _ = self.cfg.lidar_offset
        origin = self._robot.data.root_pos_w[:, :2].clone()
        origin[:, 0] += cos_yaw * offset_x - sin_yaw * offset_y
        origin[:, 1] += sin_yaw * offset_x + cos_yaw * offset_y

        direction_b = self._lidar_directions_b[None, :, :]
        direction_w = torch.empty((self.num_envs, self.cfg.lidar_num_rays, 2), device=self.device)
        direction_w[..., 0] = cos_yaw[:, None] * direction_b[..., 0] - sin_yaw[:, None] * direction_b[..., 1]
        direction_w[..., 1] = sin_yaw[:, None] * direction_b[..., 0] + cos_yaw[:, None] * direction_b[..., 1]

        # Distance to the four inner wall faces for a ray starting inside the arena.
        local_origin = origin - self.scene.env_origins[:, :2]
        wall_inner = self.cfg.arena_half_extent - 0.5 * self.cfg.wall_thickness
        wall_target_x = torch.where(direction_w[..., 0] >= 0.0, wall_inner, -wall_inner)
        wall_target_y = torch.where(direction_w[..., 1] >= 0.0, wall_inner, -wall_inner)
        valid_dx = torch.abs(direction_w[..., 0]) > 1.0e-7
        valid_dy = torch.abs(direction_w[..., 1]) > 1.0e-7
        safe_dx = torch.where(valid_dx, direction_w[..., 0], torch.ones_like(direction_w[..., 0]))
        safe_dy = torch.where(valid_dy, direction_w[..., 1], torch.ones_like(direction_w[..., 1]))
        wall_x = torch.where(
            valid_dx,
            (wall_target_x - local_origin[:, None, 0]) / safe_dx,
            torch.inf,
        )
        wall_y = torch.where(
            valid_dy,
            (wall_target_y - local_origin[:, None, 1]) / safe_dy,
            torch.inf,
        )
        wall_ranges = torch.minimum(wall_x, wall_y).clamp_min(0.0)

        # Transform every ray to every obstacle's local frame and use slab intersection for oriented boxes.
        relative_origin = origin[:, None, :] - self._obstacle_positions_w
        obstacle_cos = torch.cos(self._obstacle_yaws)
        obstacle_sin = torch.sin(self._obstacle_yaws)
        origin_local = torch.empty_like(relative_origin)
        origin_local[..., 0] = obstacle_cos * relative_origin[..., 0] + obstacle_sin * relative_origin[..., 1]
        origin_local[..., 1] = -obstacle_sin * relative_origin[..., 0] + obstacle_cos * relative_origin[..., 1]

        direction = direction_w[:, :, None, :]
        direction_local = torch.empty(
            (self.num_envs, self.cfg.lidar_num_rays, len(self.cfg.obstacle_sizes), 2), device=self.device
        )
        direction_local[..., 0] = (
            obstacle_cos[:, None, :] * direction[..., 0] + obstacle_sin[:, None, :] * direction[..., 1]
        )
        direction_local[..., 1] = (
            -obstacle_sin[:, None, :] * direction[..., 0] + obstacle_cos[:, None, :] * direction[..., 1]
        )

        half_extents = 0.5 * torch.tensor(
            [[size[0], size[1]] for size in self.cfg.obstacle_sizes], device=self.device
        )
        origin_expanded = origin_local[:, None, :, :]
        parallel = torch.abs(direction_local) < 1.0e-7
        safe_direction = torch.where(parallel, torch.ones_like(direction_local), direction_local)
        t1 = (-half_extents[None, None, :, :] - origin_expanded) / safe_direction
        t2 = (half_extents[None, None, :, :] - origin_expanded) / safe_direction
        t_near_axis = torch.minimum(t1, t2)
        t_far_axis = torch.maximum(t1, t2)
        t_near_axis = torch.where(parallel, torch.full_like(t_near_axis, -torch.inf), t_near_axis)
        t_far_axis = torch.where(parallel, torch.full_like(t_far_axis, torch.inf), t_far_axis)
        parallel_outside = parallel & (torch.abs(origin_expanded) > half_extents[None, None, :, :])
        t_near = t_near_axis.amax(dim=-1)
        t_far = t_far_axis.amin(dim=-1)
        valid = (t_far >= torch.clamp_min(t_near, 0.0)) & ~torch.any(parallel_outside, dim=-1)
        valid &= self._obstacle_active[:, None, :]
        obstacle_ranges = torch.where(valid, torch.clamp_min(t_near, 0.0), torch.inf).amin(dim=-1)

        return torch.minimum(wall_ranges, obstacle_ranges).clamp(0.0, self.cfg.lidar_max_range)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_task_state()

        scan = self._lidar_ranges_m.clone()
        if self.cfg.lidar_noise_std > 0.0:
            scan.add_(torch.randn_like(scan) * self.cfg.lidar_noise_std)
        scan.clamp_(0.0, self.cfg.lidar_max_range)
        if self.cfg.lidar_dropout_probability > 0.0:
            dropout = torch.rand_like(scan) < self.cfg.lidar_dropout_probability
            scan.masked_fill_(dropout, self.cfg.lidar_max_range)
        scan.div_(self.cfg.lidar_max_range)

        to_goal_w = self._goal_positions_w - self._robot.data.root_pos_w
        to_goal_b = quat_apply_inverse(self._robot.data.root_quat_w, to_goal_w)
        bearing = torch.atan2(to_goal_b[:, 1], to_goal_b[:, 0])
        goal = torch.stack(
            (
                self._goal_distance.clamp_max(self.cfg.lidar_max_range) / self.cfg.lidar_max_range,
                torch.sin(bearing),
                torch.cos(bearing),
            ),
            dim=-1,
        )
        velocity = torch.stack(
            (
                self._robot.data.root_lin_vel_b[:, 0] / self.cfg.max_linear_speed,
                self._robot.data.root_ang_vel_b[:, 2] / self.cfg.max_angular_speed,
            ),
            dim=-1,
        ).clamp(-2.0, 2.0)

        obs = torch.cat((scan, goal, velocity, self._actions), dim=-1)
        return {"policy": obs}

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_task_state()
        self._goal_reached = self._goal_distance < self.cfg.goal_tolerance

        geometry_collision = self._compute_geometry_collision()
        lidar_collision = self._minimum_clearance < self.cfg.collision_lidar_range
        self._collision = geometry_collision | lidar_collision

        local_pos = self._robot.data.root_pos_w - self.scene.env_origins
        outside = torch.any(torch.abs(local_pos[:, :2]) > self.cfg.arena_half_extent, dim=1)
        tipped = (self._robot.data.projected_gravity_b[:, 2] > -0.5) | (local_pos[:, 2] < -0.05)
        self._out_of_bounds = outside | tipped

        terminated = self._goal_reached | self._collision | self._out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _compute_geometry_collision(self) -> torch.Tensor:
        """Conservative circular-footprint overlap against walls and oriented boxes."""
        robot_pos = self._robot.data.root_pos_w[:, :2]
        relative = robot_pos[:, None, :] - self._obstacle_positions_w
        cos_yaw = torch.cos(self._obstacle_yaws)
        sin_yaw = torch.sin(self._obstacle_yaws)
        local = torch.empty_like(relative)
        local[..., 0] = cos_yaw * relative[..., 0] + sin_yaw * relative[..., 1]
        local[..., 1] = -sin_yaw * relative[..., 0] + cos_yaw * relative[..., 1]
        half_extents = 0.5 * torch.tensor(
            [[size[0], size[1]] for size in self.cfg.obstacle_sizes], device=self.device
        )
        outside_delta = torch.relu(torch.abs(local) - half_extents[None, :, :])
        obstacle_distance = torch.linalg.vector_norm(outside_delta, dim=-1)
        obstacle_collision = torch.any(
            (obstacle_distance < self.cfg.robot_collision_radius) & self._obstacle_active, dim=1
        )

        local_robot = robot_pos - self.scene.env_origins[:, :2]
        wall_inner = self.cfg.arena_half_extent - 0.5 * self.cfg.wall_thickness
        wall_collision = torch.any(
            torch.abs(local_robot) > (wall_inner - self.cfg.robot_collision_radius), dim=1
        )
        return obstacle_collision | wall_collision

    def _get_rewards(self) -> torch.Tensor:
        progress = self._previous_goal_distance - self._goal_distance
        self._previous_goal_distance.copy_(self._goal_distance)

        near_fraction = torch.relu(
            (self.cfg.near_obstacle_distance - self._minimum_clearance) / self.cfg.near_obstacle_distance
        )
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        safe_goal = self._goal_reached & ~self._collision

        terms = {
            "progress": self.cfg.rew_progress * progress,
            "goal": self.cfg.rew_goal * safe_goal.float(),
            "collision": self.cfg.rew_collision * (self._collision | self._out_of_bounds).float(),
            "near_obstacle": self.cfg.rew_near_obstacle * torch.square(near_fraction),
            "action_rate": self.cfg.rew_action_rate * action_rate,
            "angular": self.cfg.rew_angular * torch.abs(self._body_commands[:, 1]),
            "alive": self.cfg.rew_alive * (~(self._collision | self._out_of_bounds)).float(),
        }
        for name, value in terms.items():
            self._episode_sums[name] += value
        return torch.stack(tuple(terms.values()), dim=0).sum(dim=0)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        self._log_episodes(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._body_commands[env_ids] = 0.0
        self._wheel_targets[env_ids] = 0.0

        # The start is fixed at the arena center; yaw and point goal are randomized.
        num_resets = len(env_ids)
        yaw = 2.0 * math.pi * torch.rand(num_resets, device=self.device) - math.pi
        goal_angle = 2.0 * math.pi * torch.rand(num_resets, device=self.device) - math.pi
        goal_radius = torch.empty(num_resets, device=self.device).uniform_(
            self.cfg.min_goal_distance, self.cfg.max_goal_distance
        )
        goal_local = torch.stack((goal_radius * torch.cos(goal_angle), goal_radius * torch.sin(goal_angle)), dim=1)

        self._goal_positions_w[env_ids, :2] = self.scene.env_origins[env_ids, :2] + goal_local
        self._goal_positions_w[env_ids, 2] = 0.02

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 3] = torch.cos(0.5 * yaw)
        root_state[:, 4:6] = 0.0
        root_state[:, 6] = torch.sin(0.5 * yaw)
        root_state[:, 7:] = 0.0
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self._robot.data.default_joint_vel[env_ids])
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._randomize_obstacles(env_ids, goal_local)
        self._previous_goal_distance[env_ids] = goal_radius
        self._goal_distance[env_ids] = goal_radius
        self._goal_reached[env_ids] = False
        self._collision[env_ids] = False
        self._out_of_bounds[env_ids] = False

    def _randomize_obstacles(self, env_ids: torch.Tensor, goal_local: torch.Tensor):
        num_resets = len(env_ids)
        num_obstacles = len(self.cfg.obstacle_sizes)
        device = self.device

        active_counts = torch.randint(
            self.cfg.min_obstacles,
            self.cfg.max_obstacles + 1,
            (num_resets,),
            device=device,
        )
        active = torch.arange(num_obstacles, device=device)[None, :] < active_counts[:, None]
        positions = torch.zeros((num_resets, num_obstacles, 2), device=device)
        placed = torch.zeros_like(active)

        for obstacle_index in range(num_obstacles):
            should_place = active[:, obstacle_index]
            radius = self._obstacle_radii[obstacle_index]
            candidate = torch.zeros((num_resets, 2), device=device)
            valid = ~should_place
            for _ in range(40):
                pending = should_place & ~valid
                if not torch.any(pending):
                    break
                proposal = torch.empty((num_resets, 2), device=device).uniform_(
                    -self.cfg.obstacle_spawn_extent, self.cfg.obstacle_spawn_extent
                )
                far_from_start = torch.linalg.vector_norm(proposal, dim=1) > (self.cfg.robot_clearance + radius)
                far_from_goal = torch.linalg.vector_norm(proposal - goal_local, dim=1) > (
                    self.cfg.goal_clearance + radius
                )
                proposal_valid = far_from_start & far_from_goal
                if obstacle_index > 0:
                    distances = torch.linalg.vector_norm(
                        proposal[:, None, :] - positions[:, :obstacle_index, :], dim=-1
                    )
                    required = (
                        radius + self._obstacle_radii[:obstacle_index] + self.cfg.obstacle_clearance
                    )[None, :]
                    separated = torch.all((distances > required) | ~placed[:, :obstacle_index], dim=1)
                    proposal_valid &= separated
                accept = pending & proposal_valid
                candidate[accept] = proposal[accept]
                valid |= accept
            placed[:, obstacle_index] = should_place & valid
            positions[:, obstacle_index] = candidate

        yaw = 2.0 * math.pi * torch.rand((num_resets, num_obstacles), device=device) - math.pi
        obstacle_pose = torch.zeros((num_resets, num_obstacles, 7), device=device)
        obstacle_pose[..., 3] = torch.cos(0.5 * yaw)
        obstacle_pose[..., 6] = torch.sin(0.5 * yaw)
        obstacle_pose[..., :2] = positions + self.scene.env_origins[env_ids, None, :2]
        heights = torch.tensor([size[2] * 0.5 for size in self.cfg.obstacle_sizes], device=device)
        obstacle_pose[..., 2] = heights[None, :]

        inactive = ~placed
        obstacle_pose[..., 0][inactive] = self.scene.env_origins[env_ids, None, 0].expand_as(inactive)[inactive]
        obstacle_pose[..., 1][inactive] = self.scene.env_origins[env_ids, None, 1].expand_as(inactive)[inactive]
        obstacle_pose[..., 2][inactive] = -2.0
        self._obstacles.write_object_pose_to_sim(obstacle_pose, env_ids=env_ids)
        self._obstacles.write_object_velocity_to_sim(
            torch.zeros((num_resets, num_obstacles, 6), device=device), env_ids=env_ids
        )
        self._obstacle_positions_w[env_ids] = obstacle_pose[..., :2]
        self._obstacle_yaws[env_ids] = yaw
        self._obstacle_active[env_ids] = placed

    def _log_episodes(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        log: dict[str, float] = {}
        for name, values in self._episode_sums.items():
            log[f"Episode_Reward/{name}"] = torch.mean(values[env_ids]).item()
            values[env_ids] = 0.0
        if hasattr(self, "reset_terminated"):
            log["Episode_Outcome/success"] = torch.mean(self._goal_reached[env_ids].float()).item()
            log["Episode_Outcome/collision"] = torch.mean(self._collision[env_ids].float()).item()
            log["Episode_Outcome/out_of_bounds"] = torch.mean(self._out_of_bounds[env_ids].float()).item()
            log["Episode_Outcome/timeout"] = torch.mean(self.reset_time_outs[env_ids].float()).item()
            log["Episode/minimum_clearance_m"] = torch.mean(self._minimum_clearance[env_ids]).item()
            log["Episode/final_goal_distance_m"] = torch.mean(self._goal_distance[env_ids]).item()
        self.extras["log"] = log

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create or toggle the visual-only goal-tolerance marker."""
        if debug_vis:
            if not hasattr(self, "_goal_markers"):
                self._goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
            self._goal_markers.set_visibility(True)
        elif hasattr(self, "_goal_markers"):
            self._goal_markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Move one marker per environment to the current world-frame goal."""
        del event
        if hasattr(self, "_goal_markers") and self._robot.is_initialized:
            self._goal_markers.visualize(translations=self._goal_positions_w)
