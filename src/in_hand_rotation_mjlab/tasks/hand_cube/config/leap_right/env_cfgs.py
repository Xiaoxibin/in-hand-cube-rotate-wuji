"""LEAP Right hand cube rotation environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg

from in_hand_rotation_mjlab.robots import get_leap_right_hand_cfg
from in_hand_rotation_mjlab.robots.leap_hand.leap_right_constants import (
    LEAP_ACTUATOR_EFFORT_LIMIT_NM,
    LEAP_REFLECTED_ARMATURE,
)
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)


# =============================================================================
# LEAP Right specific initial grasp pose
# =============================================================================

GRASP_INIT_JOINT_POS = {
    "if_mcp": 0.1,
    "if_rot": -0.4,
    "if_pip": 1.3,
    "if_dip": 0.0,

    "mf_mcp": 0.1,
    "mf_rot": 0.0,
    "mf_pip": 1.3,
    "mf_dip": 0.0,

    "rf_mcp": 0.1,
    "rf_rot": 0.4,
    "rf_pip": 1.3,
    "rf_dip": 0.0,

    "th_cmc": 1.45,
    "th_axl": 1.5,
    "th_mcp": 0.579,
    "th_ipl": 1.37,
}


# =============================================================================
# Cube and grasp-cache configuration
# =============================================================================

CUBE_SPAWN_POS = (
    -0.01,
    0.035,
    0.29,
)

GRASP_CACHE_FILE = (
    "tasks/hand_cube/cache/leap_right_grasp_cache.npz"
)


# =============================================================================
# Environment configuration
# =============================================================================

def leap_right_hand_cube_rotate_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create LEAP Right hand cube rotation environment configuration."""

    return make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_leap_right_hand_cfg,
        grasp_init_joint_pos=GRASP_INIT_JOINT_POS,
        cube_spawn_pos=CUBE_SPAWN_POS,

        # LEAP has an existing grasp cache.
        grasp_cache_file=GRASP_CACHE_FILE,

        # Right hand:
        # counter-clockwise yaw is treated as positive.
        negate_yaw_rate=False,

        play=play,

        # LEAP-specific body/geometry names.
        palm_body_name="palm",
        palm_center_geom_expr="palm_collision_.*",

        # LEAP fingertip geometry.
        fingertip_geom_expr=".*_tip",

        # LEAP-specific actuator dynamics.
        robot_reflected_armature=LEAP_REFLECTED_ARMATURE,
        robot_effort_limit=LEAP_ACTUATOR_EFFORT_LIMIT_NM,
        enable_robot_dynamics_dr=True,
    )