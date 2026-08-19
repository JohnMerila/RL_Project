"""Register differential-drive LiDAR navigation environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-DiffDrive-Lidar-Nav-Direct-v0",
    entry_point=f"{__name__}.diffdrive_nav_env:DiffDriveNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.diffdrive_nav_env_cfg:DiffDriveNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DiffDriveNavPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-DiffDrive-Lidar-Nav-Play-v0",
    entry_point=f"{__name__}.diffdrive_nav_env:DiffDriveNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.diffdrive_nav_env_cfg:DiffDriveNavPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DiffDriveNavPPORunnerCfg",
    },
)

