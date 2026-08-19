"""Configuration for randomized differential-drive LiDAR navigation."""

from pathlib import Path

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets"

# Fixed obstacle geometry. Pose and active count are randomized independently in every environment at reset.
OBSTACLE_SIZES = (
    (0.35, 0.35, 0.50),
    (0.50, 0.30, 0.50),
    (0.30, 0.65, 0.50),
    (0.55, 0.40, 0.50),
    (0.40, 0.40, 0.50),
    (0.30, 0.75, 0.50),
    (0.65, 0.30, 0.50),
    (0.45, 0.55, 0.50),
)


def make_obstacle_collection_cfg() -> RigidObjectCollectionCfg:
    """Create a batched collection of kinematic collision obstacles."""
    objects: dict[str, RigidObjectCfg] = {}
    for index, size in enumerate(OBSTACLE_SIZES):
        color = (0.65 + 0.03 * (index % 3), 0.16 + 0.04 * (index % 2), 0.10)
        objects[f"obstacle_{index}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/Obstacle_{index}",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
        )
    return RigidObjectCollectionCfg(rigid_objects=objects)


DIFFDRIVE_ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSET_DIR / "differential_drive.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        activate_contact_sensors=False,
        self_collision=False,
        replace_cylinders_with_capsules=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            drive_type="force",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=4.0, damping=0.0),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.005),
        joint_pos={".*wheel_joint": 0.0},
        joint_vel={".*wheel_joint": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            effort_limit_sim=3.0,
            velocity_limit_sim=40.0,
            stiffness=0.0,
            damping=4.0,
        ),
        "passive_caster": ImplicitActuatorCfg(
            joint_names_expr=["caster_swivel_joint", "caster_wheel_joint"],
            effort_limit_sim=0.2,
            velocity_limit_sim=100.0,
            stiffness=0.0,
            damping=0.01,
        ),
    },
)


@configclass
class DiffDriveNavEnvCfg(DirectRLEnvCfg):
    """Training configuration: 120 Hz physics, 20 Hz control, 79 observations."""

    # MDP and timing
    decimation = 6
    episode_length_s = 25.0
    action_space = gym.spaces.Box(
        low=np.full(2, -1.0, dtype=np.float32),
        high=np.full(2, 1.0, dtype=np.float32),
        dtype=np.float32,
    )
    observation_space = 79
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=512,
        env_spacing=8.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # Robot and sensor assets
    robot_cfg: ArticulationCfg = DIFFDRIVE_ROBOT_CFG
    obstacle_collection_cfg: RigidObjectCollectionCfg = make_obstacle_collection_cfg()

    # Robot/controller contract
    wheel_joint_names = ("left_wheel_joint", "right_wheel_joint")
    wheel_radius = 0.055
    track_width = 0.335
    max_linear_speed = 0.8
    max_angular_speed = 1.5
    max_linear_accel = 1.5
    max_angular_accel = 3.0

    # Arena and reset distribution
    arena_half_extent = 3.5
    wall_thickness = 0.15
    min_obstacles = 2
    max_obstacles = len(OBSTACLE_SIZES)
    obstacle_sizes = OBSTACLE_SIZES
    obstacle_spawn_extent = 2.75
    robot_clearance = 0.72
    goal_clearance = 0.58
    obstacle_clearance = 0.16
    min_goal_distance = 1.5
    max_goal_distance = 3.1
    goal_tolerance = 0.30

    # Debug visualization. The marker is visual-only and its radius matches the success tolerance.
    debug_vis = False
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/DiffDriveNavigationGoal",
        markers={
            "goal": sim_utils.CylinderCfg(
                radius=goal_tolerance,
                height=0.025,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 1.0, 0.12),
                    emissive_color=(0.02, 0.45, 0.04),
                    roughness=0.35,
                    opacity=0.80,
                ),
            )
        },
    )

    # LiDAR corruption and collision thresholds
    lidar_num_rays = 72
    lidar_offset = (0.08, 0.0, 0.215)
    lidar_max_range = 8.0
    lidar_noise_std = 0.015
    lidar_dropout_probability = 0.01
    collision_lidar_range = 0.19
    robot_collision_radius = 0.20

    # Per-policy-step reward coefficients
    rew_progress = 8.0
    rew_goal = 20.0
    rew_collision = -20.0
    rew_near_obstacle = -0.35
    rew_action_rate = -0.08
    rew_angular = -0.015
    rew_alive = 0.005
    near_obstacle_distance = 0.42


@configclass
class DiffDriveNavPlayEnvCfg(DiffDriveNavEnvCfg):
    """Small, deterministic-ish configuration for checkpoint playback."""

    def __post_init__(self):
        self.scene.num_envs = 16
        self.lidar_noise_std = 0.0
        self.lidar_dropout_probability = 0.0
        self.debug_vis = True
