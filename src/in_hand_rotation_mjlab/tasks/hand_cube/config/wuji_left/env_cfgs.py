"""Wuji Left hand cube rotation environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg

from in_hand_rotation_mjlab.robots import get_wuji_left_hand_cfg
from in_hand_rotation_mjlab.robots.wuji_hand.wuji_left_constants import (
    WUJI_HOME_JOINT_POS,
)
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)


# =============================================================================
# Wuji specific configuration
# =============================================================================

CUBE_SPAWN_POS = (0.0, 0.0, 0.335,)


# =============================================================================
# Environment configuration
# =============================================================================

def wuji_left_hand_cube_rotate_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Wuji Left hand cube rotation environment configuration."""

    return make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_wuji_left_hand_cfg,

        # No grasp cache is available yet.
        # Start from the validated Wuji home pose.
        grasp_init_joint_pos=WUJI_HOME_JOINT_POS,

        cube_spawn_pos=CUBE_SPAWN_POS,

        # Important:
        # None means that reset_from_grasp_cache is not registered.
        grasp_cache_file="tasks/hand_cube/cache/wuji_left_grasp_cache.npz",
        # grasp_cache_file=None,

        # Left hand yaw convention.
        negate_yaw_rate=True,

        play=play,

        # These names should match the Wuji MJCF.
        palm_body_name="left_palm_link", # left_palm_link
        palm_center_geom_expr="left_palm_collision", # left_palm_collision

        # Do not enable fingertip contact sensor yet.
        #
        # We have not yet confirmed the exact Wuji fingertip geom names.
        # After confirming them, this can be changed to something such as:
        #
        # fingertip_geom_expr=".*_tip"
        #
        fingertip_geom_expr="left_finger[1-5]_tip", # "left_finger.*_tip_collision"

        # Do not reuse LEAP-specific actuator dynamics.
        #
        # Wuji XML already contains its own <position> actuators,
        # including kp, kv, ctrlrange and forcerange.
        robot_reflected_armature=None,
        robot_effort_limit=None,
        enable_robot_dynamics_dr=False,
    )