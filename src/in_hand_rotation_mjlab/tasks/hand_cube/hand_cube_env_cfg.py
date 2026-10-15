from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.entity import EntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from in_hand_rotation_mjlab.tasks.hand_cube import mdp as hand_cube_mdp
from in_hand_rotation_mjlab.tasks.hand_cube.mdp.actions import (
    JointPositionDeltaActionCfg,
)


# =============================================================================
# Actor observation delay randomization
# =============================================================================

# Actor observation delay randomization in environment/control steps.
# This is task-level randomization and is independent of the robot type.
_ACTOR_OBS_DELAY_MIN_LAG = 0
_ACTOR_OBS_DELAY_MAX_LAG = 1
_ACTOR_OBS_DELAY_HOLD_PROB = 0.9
_ACTOR_OBS_DELAY_UPDATE_PERIOD = 10


# =============================================================================
# Generic Hand-Cube task
# =============================================================================

def make_hand_cube_inhand_rotate_env_cfg(
    *,
    palm_body_name: str = "left_palm_link",
    palm_center_geom_expr: str = "palm_collision_.*",
) -> ManagerBasedRlEnvCfg:
    """Create the generic hand-cube in-hand rotation task.

    This function defines the task itself and does not depend on a
    particular robot embodiment.

    Robot-specific information such as:
        - robot EntityCfg
        - number of DoFs
        - initial joint pose
        - palm body name
        - palm center geometry
        - grasp cache
        - actuator limits
        - robot dynamics randomization

    is supplied by make_hand_cube_embodiment_env_cfg().
    """

    # -------------------------------------------------------------------------
    # Actor observations
    # -------------------------------------------------------------------------

    actor_terms = {
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            noise=Unoise(
                n_min=-0.01,
                n_max=0.01,
            ),
            delay_min_lag=_ACTOR_OBS_DELAY_MIN_LAG,
            delay_max_lag=_ACTOR_OBS_DELAY_MAX_LAG,
            delay_hold_prob=_ACTOR_OBS_DELAY_HOLD_PROB,
            delay_update_period=_ACTOR_OBS_DELAY_UPDATE_PERIOD,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
                # Include encoder offset effects in the measured
                # proprioception stream.
                "biased": True,
            },
        ),
        "prev_commanded_joint_pos": ObservationTermCfg(
            func=hand_cube_mdp.joint_pos_commanded,
            params={
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
    }

    # -------------------------------------------------------------------------
    # Critic observations
    # -------------------------------------------------------------------------

    critic_terms = {
        "joint_pos": ObservationTermCfg(
            func=envs_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "joint_vel": ObservationTermCfg(
            func=envs_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "joint_pos_error": ObservationTermCfg(
            func=hand_cube_mdp.joint_pos_command_error,
            params={
                "action_name": "joint_pos",
                "biased": True,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "prev_commanded_joint_pos": ObservationTermCfg(
            func=hand_cube_mdp.joint_pos_commanded,
            params={
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "cube_pose_palm": ObservationTermCfg(
            func=hand_cube_mdp.cube_pose_in_palm_frame,
            params={
                "object_name": "cube",
                "hand_cfg": SceneEntityCfg(
                    "robot",
                    body_names=(palm_body_name,),
                ),
                "palm_center_geom_expr": palm_center_geom_expr,
            },
        ),
        "cube_lin_vel_palm": ObservationTermCfg(
            func=hand_cube_mdp.cube_lin_vel_in_palm_frame,
            params={
                "object_name": "cube",
                "hand_cfg": SceneEntityCfg(
                    "robot",
                    body_names=(palm_body_name,),
                ),
                "palm_center_geom_expr": palm_center_geom_expr,
            },
        ),
        "cube_ang_vel_palm": ObservationTermCfg(
            func=hand_cube_mdp.cube_ang_vel_in_palm_frame,
            params={
                "object_name": "cube",
                "hand_cfg": SceneEntityCfg(
                    "robot",
                    body_names=(palm_body_name,),
                ),
                "palm_center_geom_expr": palm_center_geom_expr,
            },
        ),
        "cube_size": ObservationTermCfg(
            func=hand_cube_mdp.cube_size,
            params={
                "object_name": "cube",
                "geom_name": "cube_geom",
            },
        ),
        "cube_mass": ObservationTermCfg(
            func=hand_cube_mdp.cube_mass,
            params={
                "object_name": "cube",
                "body_name": "cube",
            },
        ),
        "cube_com_offset_b": ObservationTermCfg(
            func=hand_cube_mdp.cube_com_offset_b,
            params={
                "object_name": "cube",
                "body_name": "cube",
            },
        ),
        "cube_friction": ObservationTermCfg(
            func=hand_cube_mdp.cube_friction_coeff,
            params={
                "object_name": "cube",
                "geom_name": "cube_geom",
                "axis": 0,
            },
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            actor_terms,
            enable_corruption=True,
            history_length=10,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            critic_terms,
            enable_corruption=False,
            history_length=1,
            flatten_history_dim=True,
        ),
    }

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionDeltaActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=1.0,
            offset=0.0,
            use_default_offset=False,
            clip_to_joint_limits=True,
            use_soft_joint_pos_limits=True,
            delta_min=-(1 / 24),
            delta_max=(1 / 24),
            interpolate_decimation=True,
        )
    }

    # -------------------------------------------------------------------------
    # Commands
    # -------------------------------------------------------------------------

    commands: dict[str, CommandTermCfg] = {
        "frame_viz": hand_cube_mdp.HandCubeFrameVizCommandCfg(
            hand_name="robot",
            object_name="cube",
            palm_body_name=palm_body_name,
            palm_center_geom_expr=palm_center_geom_expr,
            resampling_time_range=(1e9, 1e9),
            debug_vis=True,
        ),
    }

    # -------------------------------------------------------------------------
    # Events
    # -------------------------------------------------------------------------

    events = {
        # Required for fixed-base mocap robots in vectorized environments:
        # place each hand at its own environment origin on every reset.
        "reset_base": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                # This value is overridden by the embodiment helper.
                # The default is kept for backward compatibility.
                "position_range": (-0.03, 0.03),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "reset_cube_pose": EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube"),
                "pose_range": {
                    "x": (-0.015, 0.015),
                    "y": (-0.015, 0.015),
                    "z": (-0.01, 0.01),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
    }

    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------

    rewards = {
        # LEAPSim-inspired reward with an additional drift-gated rotation term:
        #
        # R_rotate =
        #     clip(omega_fd_z)
        #     * drift_factor(xyz, roll/pitch error from reset)
        #
        "rotate_finite_diff": RewardTermCfg(
            func=hand_cube_mdp.object_yaw_finite_diff_clipped,
            weight=1.25,
            params={
                "object_name": "cube",
                "clip_min": -0.25,
                "clip_max": 0.35,#0.25
                # Smooth finite-difference yaw over recent steps.
                "history_steps": 4,
                # Multiplicative drift gate on non-yaw motion from reset pose.
                "drift_position_threshold": 0.02,
                "drift_tilt_threshold": 0.35,
                "drift_mode": "step",
                "drift_inside_factor": 1.0,
                "drift_outside_factor": 0.1,
            },
        ),

        # Optional fingertip contact reward.
        # It remains disabled in the generic task because not every
        # embodiment necessarily has the same fingertip geometry.
        #
        "fingertip_contact": RewardTermCfg(
            func=hand_cube_mdp.fingertip_cube_contact_fraction,
            weight=0.08,#0.15
            params={"sensor_name": "fingertip_cube_contact"},
        ),

        "object_linvel_penalty": RewardTermCfg(
            func=hand_cube_mdp.object_linvel_l1,
            weight=-0.3,
            params={
                "object_name": "cube",
            },
        ),
        "pose_diff_penalty": RewardTermCfg(
            func=hand_cube_mdp.pose_diff_l2_from_reset,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
                "average_per_joint": False,
                "joint_tolerance": 0.15,
            },
        ),
        "torque_penalty": RewardTermCfg(
            func=hand_cube_mdp.joint_torque_l2,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "work_penalty": RewardTermCfg(
            func=hand_cube_mdp.actuator_work_l2_penalty,
            weight=-0.05,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),
                ),
            },
        ),
        "object_fallen": RewardTermCfg(
            func=hand_cube_mdp.object_fallen,
            weight=-10.0,
            params={
                "object_name": "cube",
                # Cube half-size is 0.03: a cube resting on the ground has its
                # center at z=0.03, so the threshold must sit above that.
                "minimum_height": 0.05, #0.03
            },
        ),
        # Direct cube-pose-drift penalty, active from step 0 (no curriculum).
        #
        # Closes the reward gap that let the policy profit from pushing the
        # cube into the radial-side pocket (MCT_CMC3/MCPI1/finger-root
        # groove): no other term charges for cube position drift while the
        # curriculum penalties are still at weight_min. The dead-zone params
        # exempt normal in-hand rotation jiggle; only genuine drift is charged.
        "cube_pose_drift_penalty": RewardTermCfg(
            func=hand_cube_mdp.object_pose_deviation_l2,
            weight=-1.5,
            params={
                "object_name": "cube",
                "position_weight": 1.0,
                "tilt_weight": 1.0,
                "position_threshold": 0.005,
                "tilt_threshold": 0.05,
            },
        ),
    }

    # -------------------------------------------------------------------------
    # Terminations
    # -------------------------------------------------------------------------

    terminations = {
        "time_out": TerminationTermCfg(
            func=envs_mdp.time_out,
            time_out=True,
        ),
        "cube_fell": TerminationTermCfg(
            func=envs_mdp.root_height_below_minimum,
            params={
                # Must exceed the cube half-size (0.03): a ground-resting cube
                # centers at z=0.03 and must not count as "fell" mid-episode
                # before the reward term does. Kept in sync with object_fallen.
                "minimum_height": 0.05, #0.03
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "cube_too_fast": TerminationTermCfg(
            func=hand_cube_mdp.object_linear_speed_above,
            params={
                "max_linear_speed": 1.0,
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "cube_pose_deviation": TerminationTermCfg(
            func=hand_cube_mdp.object_pose_rp_position_deviation_from_reset,
            params={
                "max_position_error": 0.08,
                "max_tilt_error": 0.8,
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "nan": TerminationTermCfg(
            func=envs_mdp.nan_detection,
        ),
    }

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics = {
        "rotation_progress": MetricsTermCfg(
            func=hand_cube_mdp.object_rotation_progress,
            params={
                "asset_cfg": SceneEntityCfg("cube"),
                "target_yaw_rate": 0.20,
                "position_threshold": 0.02,
                "tilt_threshold": 0.35,
            },
        ),
        "linear_speed": MetricsTermCfg(
            func=hand_cube_mdp.object_linear_speed,
            params={
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "position_error": MetricsTermCfg(
            func=hand_cube_mdp.object_pose_rp_error_from_reset,
            params={
                "component": "position",
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
        "tilt_error": MetricsTermCfg(
            func=hand_cube_mdp.object_pose_rp_error_from_reset,
            params={
                "component": "tilt",
                "asset_cfg": SceneEntityCfg("cube"),
            },
        ),
    }

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------

    curriculum = {
        "object_linvel_penalty_weight": CurriculumTermCfg(
            func=hand_cube_mdp.reward_weight_by_metric_progress,
            params={
                "reward_name": "object_linvel_penalty",
                "metric_name": "rotation_progress",
                "progress_min": 0.05,
                "progress_max": 0.25,
                "weight_min": -0.03,
                "weight_max": -0.3,
                "ema_alpha": 0.08,
                "weight_lerp": 0.15,
            },
        ),
        "pose_diff_penalty_weight": CurriculumTermCfg(
            func=hand_cube_mdp.reward_weight_by_metric_progress,
            params={
                "reward_name": "pose_diff_penalty",
                "metric_name": "rotation_progress",
                "progress_min": 0.05,
                "progress_max": 0.25,
                "weight_min": -0.01,
                "weight_max": -0.1,
                "ema_alpha": 0.08,
                "weight_lerp": 0.15,
            },
        ),
        "torque_penalty_weight": CurriculumTermCfg(
            func=hand_cube_mdp.reward_weight_by_metric_progress,
            params={
                "reward_name": "torque_penalty",
                "metric_name": "rotation_progress",
                "progress_min": 0.05,
                "progress_max": 0.25,
                "weight_min": -0.1,
                "weight_max": -1.0,
                "ema_alpha": 0.08,
                "weight_lerp": 0.15,
            },
        ),
        "work_penalty_weight": CurriculumTermCfg(
            func=hand_cube_mdp.reward_weight_by_metric_progress,
            params={
                "reward_name": "work_penalty",
                "metric_name": "rotation_progress",
                "progress_min": 0.05,
                "progress_max": 0.25,
                "weight_min": -0.01,
                "weight_max": -0.1,
                "ema_alpha": 0.08,
                "weight_lerp": 0.15,
            },
        ),
    }

    # -------------------------------------------------------------------------
    # Environment configuration
    # -------------------------------------------------------------------------

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(
                terrain_type="plane",
            ),
            entities={},
            num_envs=1,
            env_spacing=0.6,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
        curriculum=curriculum,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=palm_body_name,
            distance=0.45,
            elevation=-25,
            azimuth=110,
        ),
        sim=SimulationCfg(
            nconmax=200,
            njmax=600,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                impratio=100,
                cone="elliptic",
            ),
        ),
        decimation=10,
        # 400 control steps at 20 Hz.
        episode_length_s=20.0,
        scale_rewards_by_dt=True,
    )


# =============================================================================
# Embodiment-specific helper
# =============================================================================

def make_hand_cube_embodiment_env_cfg(
    *,
    robot_cfg_fn: Callable[[], EntityCfg],
    grasp_init_joint_pos: dict[str, float],
    cube_spawn_pos: tuple[float, float, float],
    grasp_cache_file: str | None = None,
    negate_yaw_rate: bool = True,
    play: bool = False,
    palm_body_name: str = "palm",
    palm_center_geom_expr: str = "palm_collision_.*",
    fingertip_geom_expr: str | None = None,
    robot_reflected_armature: Any | None = None,
    robot_effort_limit: Any | None = None,
    enable_robot_dynamics_dr: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a hand-cube environment for a concrete hand embodiment.

    This function is shared by different robot embodiments.

    Required robot-specific inputs:
        robot_cfg_fn:
            Function returning the robot EntityCfg.

        grasp_init_joint_pos:
            Initial joint configuration.

        cube_spawn_pos:
            Initial cube position.

    Optional robot-specific inputs:
        grasp_cache_file:
            Path to a grasp cache. If None, grasp-cache reset is disabled.

        palm_body_name:
            Name of the palm body in the robot MJCF.

        palm_center_geom_expr:
            Geometry expression used to define the palm center.

        fingertip_geom_expr:
            Fingertip geometry expression. If None, the fingertip contact
            sensor is not created.

        robot_reflected_armature:
            Robot-specific reflected armature values used by optional
            actuator dynamics randomization.

        robot_effort_limit:
            Robot-specific actuator effort limits.

        enable_robot_dynamics_dr:
            Enable actuator-level domain randomization.

    The important design principle is that this function must not assume
    a particular robot such as LEAP, Wuji, or the future 21-DoF hand.
    """

    import mujoco

    from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    def get_cube_spec(
        cube_size: float = 0.031,
        mass: float = 0.1,
    ) -> mujoco.MjSpec:
        """Create the generic cube MJCF specification."""

        spec = mujoco.MjSpec()

        body = spec.worldbody.add_body(
            name="cube",
        )

        body.add_freejoint(
            name="cube_joint",
        )

        body.add_geom(
            name="cube_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(cube_size,) * 3,
            mass=mass,
            rgba=(0.85, 0.3, 0.15, 1.0),
        )

        return spec

    # -------------------------------------------------------------------------
    # Start from the generic task.
    # -------------------------------------------------------------------------

    cfg = make_hand_cube_inhand_rotate_env_cfg(
        palm_body_name=palm_body_name,
        palm_center_geom_expr=palm_center_geom_expr,
    )

    # -------------------------------------------------------------------------
    # Robot and cube.
    # -------------------------------------------------------------------------

    cfg.rewards["rotate_finite_diff"].params[
        "negate_yaw_rate"
    ] = negate_yaw_rate

    # The rotation-progress metric drives the curriculum terms, so its
    # rotation direction convention must match the reward term's.
    cfg.metrics["rotation_progress"].params["negate_yaw_rate"] = negate_yaw_rate

    robot_cfg = robot_cfg_fn()

    # The initial pose is supplied by the concrete embodiment.
    robot_cfg.init_state.joint_pos = dict(grasp_init_joint_pos)

    cfg.scene.entities = {
        "robot": robot_cfg,
        "cube": EntityCfg(
            init_state=EntityCfg.InitialStateCfg(
                pos=cube_spawn_pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            ),
            spec_fn=get_cube_spec,
        ),
    }

    # -------------------------------------------------------------------------
    # Optional fingertip contact sensor.
    # -------------------------------------------------------------------------

    if fingertip_geom_expr is not None:
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (
            ContactSensorCfg(
                name="fingertip_cube_contact",
                primary=ContactMatch(
                    mode="geom",
                    pattern=fingertip_geom_expr,
                    entity="robot",
                ),
                secondary=ContactMatch(
                    mode="geom",
                    pattern="cube_geom",
                    entity="cube",
                ),
                fields=("found",),
                reduce="none",
                num_slots=1,
            ),
        )

        cfg.sim.contact_sensor_maxmatch = 256

        cfg.metrics["fingertip_contact_count"] = MetricsTermCfg(
            func=hand_cube_mdp.contact_count,
            params={
                "sensor_name": "fingertip_cube_contact",
            },
        )

        cfg.metrics["fingertip_contact_fraction"] = MetricsTermCfg(
            func=hand_cube_mdp.contact_fraction,
            params={
                "sensor_name": "fingertip_cube_contact",
            },
        )

        cfg.metrics["fingertip_found_mean"] = MetricsTermCfg(
            func=hand_cube_mdp.contact_found_mean,
            params={
                "sensor_name": "fingertip_cube_contact",
            },
        )

        cfg.metrics["fingertip_found_max"] = MetricsTermCfg(
            func=hand_cube_mdp.contact_found_max,
            params={
                "sensor_name": "fingertip_cube_contact",
            },
        )

    # -------------------------------------------------------------------------
    # Reset robot joints.
    #
    # The concrete initial pose is already assigned to robot_cfg.init_state.
    # Therefore no random offset is added here.
    # -------------------------------------------------------------------------

    cfg.events["reset_robot_joints"].params["position_range"] = (
        0.0,
        0.0,
    )

    cfg.events["reset_robot_joints"].params["velocity_range"] = (
        0.0,
        0.0,
    )

    # -------------------------------------------------------------------------
    # Action configuration.
    #
    # Keep these values shared across embodiments.
    # The number of actions is determined automatically by the robot's
    # actuated joints.
    # -------------------------------------------------------------------------

    joint_pos_action = cfg.actions["joint_pos"]

    assert isinstance(
        joint_pos_action,
        JointPositionDeltaActionCfg,
    )

    joint_pos_action.scale = 1.0
    joint_pos_action.offset = 0.0
    joint_pos_action.use_default_offset = False

    # -------------------------------------------------------------------------
    # Cube reset pose.
    # -------------------------------------------------------------------------

    cfg.events["reset_cube_pose"].params["pose_range"] = {
        "x": (cube_spawn_pos[0], cube_spawn_pos[0]),
        "y": (cube_spawn_pos[1], cube_spawn_pos[1]),
        "z": (cube_spawn_pos[2], cube_spawn_pos[2]),
        "yaw": (0,0),
    }

    # -------------------------------------------------------------------------
    # Optional grasp-cache reset.
    #
    # LEAP can use its existing cache.
    # Wuji can set grasp_cache_file=None during the initial integration stage.
    # -------------------------------------------------------------------------

    if grasp_cache_file is not None:
        cache_path = Path(grasp_cache_file)

        if not cache_path.is_absolute():
            cache_path = MYMJLAB_SRC_PATH / cache_path

        cfg.events["reset_from_grasp_cache"] = EventTermCfg(
            func=hand_cube_mdp.reset_from_grasp_cache,
            mode="reset",
            params={
                "cache_file": str(cache_path),
                "scale_list": (1.0,),
                "scale_jitter": 0,
                "cube_cfg": SceneEntityCfg(
                    "cube",
                    geom_names=("cube_geom",),
                ),
                "pose_range": {
                    "x": (0, 0),
                    "y": (0, 0),
                    "z": (0, 0),
                    "yaw": (0, 0),
                },
            },
        )

    # -------------------------------------------------------------------------
    # Generic physical domain randomization.
    #
    # These terms are not tied to LEAP and can be shared across embodiments.
    # -------------------------------------------------------------------------

    cfg.events["dr_cube_com"] = EventTermCfg(
        func=envs_mdp.randomize_field,
        mode="reset",
        domain_randomization=True,
        params={
            "field": "body_ipos",
            "ranges": {
                0: (-0.002, 0.002),
                1: (-0.002, 0.002),
                2: (-0.002, 0.002),
            },
            "distribution": "uniform",
            "operation": "add",
            "asset_cfg": SceneEntityCfg(
                "cube",
                body_names=("cube",),
            ),
        },
    )

    cfg.events["dr_shared_contact_friction"] = EventTermCfg(
        func=hand_cube_mdp.randomize_shared_contact_friction,
        mode="reset",
        params={
            "friction_range": (0.6, 1.4),
            "hand_cfg": SceneEntityCfg(
                "robot",
                geom_names=(".*",),
            ),
            "cube_cfg": SceneEntityCfg(
                "cube",
                geom_names=("cube_geom",),
            ),
            "axes": (0,),
        },
    )

    cfg.events["dr_cube_mass"] = EventTermCfg(
        func=hand_cube_mdp.randomize_body_mass,
        mode="reset",
        params={
            "mass_range": (0.7, 1.4),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg(
                "cube",
                body_names=("cube",),
            ),
        },
    )

    cfg.events["dr_robot_link_masses"] = EventTermCfg(
        func=hand_cube_mdp.randomize_body_mass,
        mode="reset",
        params={
            "mass_range": (0.8, 1.2),
            "distribution": "uniform",
            "operation": "scale",
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=(".*",),
            ),
        },
    )

    # -------------------------------------------------------------------------
    # Robot actuator dynamics randomization.
    #
    # This section is optional because different hands have different
    # actuator models.
    #
    # LEAP enables it by passing:
    #     enable_robot_dynamics_dr=True
    #     robot_reflected_armature=LEAP_REFLECTED_ARMATURE
    #     robot_effort_limit=LEAP_ACTUATOR_EFFORT_LIMIT_NM
    #
    # Wuji initially disables it because its XML already contains the
    # validated position actuator dynamics.
    # -------------------------------------------------------------------------

    # if enable_robot_dynamics_dr:
    #     if robot_reflected_armature is None:
    #         raise ValueError(
    #             "robot_reflected_armature must be provided when "
    #             "enable_robot_dynamics_dr=True."
    #         )

    #     if robot_effort_limit is None:
    #         raise ValueError(
    #             "robot_effort_limit must be provided when "
    #             "enable_robot_dynamics_dr=True."
    #         )

    #     cfg.events["dr_motor_friction"] = EventTermCfg(
    #         func=envs_mdp.randomize_field,
    #         mode="reset",
    #         domain_randomization=True,
    #         params={
    #             "field": "dof_frictionloss",
    #             "ranges": (0.5, 1.8),
    #             "distribution": "uniform",
    #             "operation": "scale",
    #             "asset_cfg": SceneEntityCfg(
    #                 "robot",
    #                 joint_names=(".*",),
    #             ),
    #         },
    #     )

    #     cfg.events["dr_motor_damping"] = EventTermCfg(
    #         func=envs_mdp.randomize_field,
    #         mode="reset",
    #         domain_randomization=True,
    #         params={
    #             "field": "dof_damping",
    #             "ranges": (0.6, 1.6),
    #             "distribution": "uniform",
    #             "operation": "scale",
    #             "asset_cfg": SceneEntityCfg(
    #                 "robot",
    #                 joint_names=(".*",),
    #             ),
    #         },
    #     )

    #     cfg.events["dr_reflected_inertia"] = EventTermCfg(
    #         func=envs_mdp.randomize_field,
    #         mode="startup",
    #         domain_randomization=True,
    #         params={
    #             "field": "dof_armature",
    #             "ranges": (
    #                 1 * robot_reflected_armature,  # 0.7
    #                 1 * robot_reflected_armature,  # 1.3
    #             ),
    #             "distribution": "uniform",
    #             "operation": "abs",
    #             "asset_cfg": SceneEntityCfg(
    #                 "robot",
    #                 joint_names=(".*",),
    #             ),
    #         },
    #     )

    #     cfg.events["dr_robot_pd_gains"] = EventTermCfg(
    #         func=envs_mdp.randomize_pd_gains,
    #         mode="startup",
    #         params={
    #             "kp_range": (1, 1),  # (0.9, 1.1)
    #             "kd_range": (1, 1),  # (0.9, 1.1)
    #             "distribution": "uniform",
    #             "operation": "scale",
    #             "asset_cfg": SceneEntityCfg("robot"),
    #         },
    #     )

    #     cfg.events["dr_action_delay"] = EventTermCfg(
    #         func=envs_mdp.sync_actuator_delays,
    #         mode="reset",
    #         params={
    #             "lag_range": (
    #                 int(round(1 * cfg.decimation)),  #0.5
    #                 int(round(1 * cfg.decimation)),  #1.5
    #             ),
    #             "asset_cfg": SceneEntityCfg(
    #                 "robot",
    #                 actuator_names=(".*",),
    #             ),
    #         },
    #     )

    #     cfg.events["dr_effort_limits"] = EventTermCfg(
    #         func=hand_cube_mdp.set_actuator_effort_limits,
    #         mode="reset",
    #         params={
    #             "effort_limit": (
    #                 1 * robot_effort_limit, # 0.9
    #                 1 * robot_effort_limit,  #1.1
    #             ),
    #             "asset_cfg": SceneEntityCfg(
    #                 "robot",
    #                 actuator_names=(".*",),
    #             ),
    #         },
    #     )

    # -------------------------------------------------------------------------
    # Viewer.
    # -------------------------------------------------------------------------

    cfg.viewer.body_name = palm_body_name

    # -------------------------------------------------------------------------
    # Play mode.
    # -------------------------------------------------------------------------

    if play:
        cfg.episode_length_s = 1e9
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}

    return cfg