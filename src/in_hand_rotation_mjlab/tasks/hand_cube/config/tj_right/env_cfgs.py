"""TJ Right hand cube rotation environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from in_hand_rotation_mjlab.robots import get_tj_right_hand_cfg
from in_hand_rotation_mjlab.robots.tj_hand.tj_right_constants import (
    TJ_HOME_JOINT_POS,
)
from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.hand_cube_env_cfg import (
    make_hand_cube_embodiment_env_cfg,
)


# =============================================================================
# TJ specific configuration
# =============================================================================

CUBE_SPAWN_POS = (-0.006, 0.032, 0.08)

GRASP_CACHE_FILE = ("tasks/hand_cube/cache/tj_right_grasp_cache_31.npz")
# =============================================================================
# Environment configuration
# =============================================================================

def tj_right_hand_cube_rotate_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create TJ Right hand cube rotation environment configuration."""

    return make_hand_cube_embodiment_env_cfg(
        robot_cfg_fn=get_tj_right_hand_cfg,

        # Start from the validated TJ home pose.
        grasp_init_joint_pos=TJ_HOME_JOINT_POS,

        cube_spawn_pos=CUBE_SPAWN_POS,

        # No TJ grasp cache is available yet.
        # Keep this disabled until a TJ-specific grasp cache is generated.
        grasp_cache_file=GRASP_CACHE_FILE, # None / GRASP_CACHE_FILE

        # TJ is a right-hand model.
        # Do not use the Wuji left-hand yaw convention.
        negate_yaw_rate=False,

        play=play,

        # These names should match the TJ MJCF.
        palm_body_name="Palm_R",
        palm_center_geom_expr="Palm_R_collision",

        # TJ fingertip collision geoms:
        #
        #   tip_thumb_collision
        #   tip_index_collision
        #   tip_middle_collision
        #   tip_ring_collision
        #   tip_pinky_collision
        #
        # The expression matches the five TJ fingertips only.
        fingertip_geom_expr=r"tip_(?:thumb|index|middle|ring|pinky)_collision",

        # Do not reuse LEAP-specific actuator dynamics.
        #
        # TJ XML already contains its own position actuators.
        robot_reflected_armature=None,
        robot_effort_limit=None,
        enable_robot_dynamics_dr=False,
    )


def _apply_tj_reward_tuning(cfg: ManagerBasedRlEnvCfg) -> None:
    """TJ-specific reward tuning (applied on top of the shared task config).

    1. rotate_finite_diff:
       - negative_yaw_rate_penalty=2.0: penalize backward yaw 2x harder than
         forward yaw, so "rotate forward / slip back" cycles are strictly
         unprofitable (fast slips cost -0.5 vs forward cap +0.25).
       - tumble_penalty_weight=0.5: charge for cube angular velocity not
         aligned with the cube's own z axis (tumbling about x/y), the failure
         mode that wedges the cube into a jammed pose. A 1 rad/s flip costs
         -0.5/step, outweighing the contact reward (0.15).
       - drift_mode="exp" with drift_decay_k_tilt=8.0: continuous decay of the
         rotation reward with tilt instead of the two-level step gate, which
         had a cliff at 0.35 rad the policy could camp just inside.
    2. torque_penalty: normalize by 21 joints at the +/-1.0 Nm actuator limit,
       so the raw term lives in [0, 1] instead of [0, 21]. Without this the
       penalty outweighs the rotation reward once the policy learns to push.
    3. work_penalty: clip the mechanical-power signal at +/-1.0 W-equivalent
       so rare contact spikes cannot dominate the squared term.
    4. fingertip_contact: DISABLED (removed). The proven LEAP policy was
       trained with no contact reward at all (its sensor was never even
       created), and the unconditional contact income (near the 3-finger cap
       every step) is the main revenue of the "locked grasp + micro-rotation"
       local optimum. It also distorts the grasp style: the policy leans on
       mid-phalanx contact instead of fingertip grips. Fall prevention is
       already covered by object_fallen (-10) and the cube_fell termination.
       The contact metrics (Episode_Metrics/fingertip_*) are kept for
       monitoring.
    """
    cfg.rewards["rotate_finite_diff"].params["negative_yaw_rate_penalty"] = 2.0
    cfg.rewards["rotate_finite_diff"].params["low_rate_boost"] = 2.0
    cfg.rewards["rotate_finite_diff"].params["tumble_penalty_weight"] = 0.5
    cfg.rewards["rotate_finite_diff"].params["tumble_dead_zone"] = 0.15
    cfg.rewards["rotate_finite_diff"].params["drift_mode"] = "exp"
    cfg.rewards["rotate_finite_diff"].params["drift_decay_k_tilt"] = 8.0
    cfg.rewards["rotate_finite_diff"].params["drift_decay_k_pos"] = 30.0
    cfg.rewards["rotate_finite_diff"].params["drift_tilt_threshold"] = 0.15

    cfg.rewards["torque_penalty"].params["max_abs_torque"] = 1.0
    cfg.rewards["torque_penalty"].params["normalize"] = True

    cfg.rewards["work_penalty"].params["max_abs_power"] = 1.0

    # Cumulative-yaw reward: pays a bonus per quarter turn actually achieved,
    # so visible rotation progress (rad, not rad/s) earns its own reward.
    # The rate reward above rewards spinning fast; this one rewards the cube
    # actually turning further than before. Signed integral: backward slips
    # cancel progress. Drift-gated: stops paying if the cube leaves the palm.
    cfg.rewards["cumulative_yaw"] = RewardTermCfg(
        func=hand_cube_mdp.object_cumulative_yaw,
        weight=0.5,
        params={
            "object_name": "cube",
            "negate_yaw_rate": False,   # TJ right hand: CCW positive
            "step_angle": 0.7852, #1.5707963,    # pi/4: bonus per quarter turn
            "bonus": 1.0,
            "drift_position_threshold": 0.02,
            "drift_tilt_threshold": 0.35,
        },
    )

    # cfg.rewards["fingertip_contact"] = RewardTermCfg(
    #     func=hand_cube_mdp.fingertip_cube_contact_fraction,
    #     weight=0.08,
    #     params={"sensor_name": "fingertip_cube_contact"},
    # )
    if "fingertip_contact" in cfg.rewards:
        del cfg.rewards["fingertip_contact"]


def tj_right_hand_cube_rotate_env_cfg_tuned(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """TJ env config with TJ-specific reward tuning applied."""
    cfg = tj_right_hand_cube_rotate_env_cfg(play=play)
    _apply_tj_reward_tuning(cfg)
    return cfg
