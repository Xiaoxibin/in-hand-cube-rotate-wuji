from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def actuator_work_l2_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  use_joint_acc: bool = False,
  max_abs_power: float | None = None,
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """Penalize squared actuation work-like signal.

  By default this uses tau*qdot. For LEAP-style behavior, set
  ``use_joint_acc=True`` to use tau*qacc (their finite-difference proxy).
  Optional clipping prevents rare physics spikes from dominating rewards.
  """
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  motion_term = (
    asset.data.joint_acc[:, asset_cfg.joint_ids]
    if use_joint_acc
    else asset.data.joint_vel[:, asset_cfg.joint_ids]
  )
  mech_power = torch.sum(tau * motion_term, dim=-1)
  if max_abs_power is not None:
    mech_power = torch.clamp(mech_power, min=-max_abs_power, max=max_abs_power)
  work = torch.square(mech_power)
  if normalize and max_abs_power is not None and max_abs_power > 0.0:
    work = work / (max_abs_power**2)
  return work


class object_yaw_finite_diff_clipped:
  """Yaw finite-difference reward term with optional temporal averaging.

  This computes signed yaw-rate from wrapped finite differences and can smooth
  it over the latest ``history_steps`` transitions:

    yaw_rate_t = ±mean_t-H+1..t(delta_yaw) / dt

  The sign is controlled by ``negate_yaw_rate``:
  - For left hand (clockwise rotation positive): negate_yaw_rate=True
  - For right hand (counter-clockwise rotation positive): negate_yaw_rate=False

  Optionally, the yaw reward is multiplied by a drift factor based on object
  translation and roll/pitch deviation from reset pose. This suppresses reward
  when the cube drifts via non-target motion.

  Additionally, ``negative_yaw_rate_penalty`` (>1.0) makes backward yaw rates
  more costly than forward rates are rewarding, discouraging the
  "rotate forward / slip back" limit cycle. Default 1.0 keeps the original
  symmetric clipping behavior. The drift gate discounts only positive rewards;
  backward penalties apply at full strength even when the cube drifts.

  ``tumble_penalty_weight`` (>0) charges for the cube's angular velocity
  component NOT aligned with the cube's own z axis (tumbling vs in-hand
  spinning), penalizing flip-into-jam failure modes immediately.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self.history_steps = int(cfg.params.get("history_steps", 1))
    if self.history_steps < 1:
      raise ValueError(f"history_steps must be >= 1, got {self.history_steps}")
    self.prev_yaw = torch.zeros(env.num_envs, device=env.device)
    self._env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    self._delta_hist = torch.zeros(
      (env.num_envs, self.history_steps), device=env.device
    )
    self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    self._env = env
    self.reset(None)

  def _current_yaw(self) -> torch.Tensor:
    cube: Entity = self._env.scene[self.object_name]
    _, _, yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)
    return yaw

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    cube: Entity = self._env.scene[self.object_name]
    yaw = self._current_yaw()
    self.prev_yaw[env_ids] = yaw[env_ids]
    self._delta_hist[env_ids] = 0.0
    self._hist_ptr[env_ids] = 0
    self._hist_count[env_ids] = 0
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = cube.data.root_link_pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    clip_min: float = -0.25,
    clip_max: float = 0.25,
    low_rate_boost: float = 1.0,
    history_steps: int | None = None,
    negate_yaw_rate: bool = True,
    negative_yaw_rate_penalty: float = 1.0,
    tumble_penalty_weight: float = 0.0,
    tumble_dead_zone: float = 0.0,
    drift_position_threshold: float = 0.02,
    drift_tilt_threshold: float = 0.35,
    drift_inside_factor: float = 1.0,
    drift_outside_factor: float = 0.1,
    drift_mode: str = "step",
    drift_decay_k_pos: float = 30.0,
    drift_decay_k_tilt: float = 3.0,
    drift_min_factor: float = 0.1,
  ) -> torch.Tensor:
    del object_name  # Bound in __init__ for consistency across calls.
    if history_steps is not None and history_steps != self.history_steps:
      raise ValueError(
        f"history_steps={history_steps} does not match initialized value "
        f"{self.history_steps}. Set it via reward params."
      )

    yaw = self._current_yaw()
    # On first post-reset step, anchor the finite-diff reference.
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    delta_yaw = wrap_to_pi(yaw - self.prev_yaw)
    self.prev_yaw[:] = yaw

    # Rolling mean over latest history_steps finite differences.
    hist_ptr = self._hist_ptr
    self._delta_hist[self._env_ids, hist_ptr] = delta_yaw
    self._hist_ptr = (hist_ptr + 1) % self.history_steps
    self._hist_count = torch.clamp(self._hist_count + 1, max=self.history_steps)

    hist_len = torch.clamp(self._hist_count, min=1).to(delta_yaw.dtype)
    avg_delta_yaw = self._delta_hist.sum(dim=-1) / hist_len

    # Compute yaw rate with optional negation based on hand embodiment.
    # Left hand: negate_yaw_rate=True (clockwise rotation -> positive reward)
    # Right hand: negate_yaw_rate=False (counter-clockwise rotation -> positive reward)
    yaw_rate = avg_delta_yaw / max(env.step_dt, 1e-6)
    if negate_yaw_rate:
      yaw_rate = -yaw_rate
    # Asymmetric penalty: clamp FIRST, then scale negative rewards up. Doing it
    # in this order keeps fast backward slips penalized beyond the forward
    # saturation level (-0.25 * penalty vs +0.25), so "rotate forward / slip
    # back" cycles are strictly unprofitable. Multiplying before clamping
    # would saturate fast slips back at -clip_min and only affect slow ones.
    # low_rate_boost > 1 raises the reward slope in the slow-spinning regime
    # (below clip_max / low_rate_boost), so early gains in rotation speed pay
    # off immediately instead of being flattened by the saturation. With the
    # default 1.0 this is identical to plain clamping.
    if low_rate_boost != 1.0:
      boosted = torch.clamp(yaw_rate * low_rate_boost, min=0.0, max=clip_max)
      yaw_reward = torch.where(
        yaw_rate >= 0.0,
        boosted,
        torch.clamp(yaw_rate, min=clip_min, max=0.0),
      )
    else:
      yaw_reward = torch.clamp(yaw_rate, min=clip_min, max=clip_max)
    if negative_yaw_rate_penalty != 1.0:
      yaw_reward = torch.where(
        yaw_reward < 0.0, yaw_reward * negative_yaw_rate_penalty, yaw_reward
      )

    cube: Entity = self._env.scene[self.object_name]
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)

    # Tumble penalty: decompose the cube's world-frame angular velocity into
    # the component about the cube's own z axis (in-hand spin, the target
    # motion) and the remainder (tumbling about x/y, the failure mode that
    # wedges the cube into a jammed pose). The reward gains a per-step cost
    # proportional to the tumble magnitude, so a quarter-flip costs from the
    # very first step instead of only after the drift gate notices.
    # ``tumble_dead_zone`` exempts micro-jitter from passive settling
    # (measured ~0.1 rad/s while holding still), so a stationary grasp
    # doesn't pay a constant penalty.
    if tumble_penalty_weight > 0.0:
      cube_ang_vel_w = cube.data.root_link_ang_vel_w
      cube_quat_w = cube.data.root_link_quat_w
      z_axis = torch.zeros_like(cube_ang_vel_w)
      z_axis[:, 2] = 1.0
      cube_z_axis_w = quat_apply(cube_quat_w, z_axis)
      spin_rate = (cube_ang_vel_w * cube_z_axis_w).sum(dim=-1, keepdim=True)
      tumble_rate = cube_ang_vel_w - spin_rate * cube_z_axis_w
      tumble_mag = torch.norm(tumble_rate, dim=-1) - tumble_dead_zone
      tumble_cost = tumble_penalty_weight * torch.clamp(tumble_mag, min=0.0)
      yaw_reward = yaw_reward - tumble_cost

    pos_error = torch.linalg.vector_norm(
      cube.data.root_link_pos_w - self._init_pos_w, dim=-1
    )
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    if drift_mode == "step":
      in_bounds = (pos_error <= drift_position_threshold) & (
        tilt_error <= drift_tilt_threshold
      )
      drift_factor = torch.where(
        in_bounds,
        torch.full_like(yaw_reward, drift_inside_factor),
        torch.full_like(yaw_reward, drift_outside_factor),
      )
    elif drift_mode == "exp":
      pos_excess = torch.clamp(pos_error - drift_position_threshold, min=0.0)
      tilt_excess = torch.clamp(tilt_error - drift_tilt_threshold, min=0.0)
      drift_factor = torch.exp(
        -drift_decay_k_pos * pos_excess - drift_decay_k_tilt * tilt_excess
      )
      drift_factor = torch.clamp(drift_factor, min=drift_min_factor, max=1.0)
    else:
      raise ValueError(
        f"Unknown drift_mode '{drift_mode}'. Expected one of: 'step', 'exp'."
      )

    drift_factor = torch.where(
      self._has_init, drift_factor, torch.ones_like(drift_factor)
    )
    # One-sided drift gate: only discount POSITIVE yaw rewards when the cube
    # drifts. Backward (penalty) rewards stay at full strength regardless of
    # drift, so slipping while drifting cannot be excused by the gate.
    drift_factor = torch.where(
      yaw_reward >= 0.0,
      drift_factor,
      torch.ones_like(drift_factor),
    )
    return yaw_reward * drift_factor


def object_linvel_l1(
  env: ManagerBasedRlEnv,
  object_name: str,
  clip_max: float | None = None,
  normalize: bool = False,
  dead_zone: float = 0.0,
) -> torch.Tensor:
  """LEAPSim linear velocity penalty term: ||v_obj||_1.

  ``dead_zone`` (m/s) exempts small in-hand jiggle from the cost; the
  penalty is then linear in the excess speed above the dead zone, so the
  marginal cost of sliding faster stays constant.
  """
  cube: Entity = env.scene[object_name]
  val = torch.norm(cube.data.root_link_lin_vel_w, p=1, dim=-1)
  if dead_zone > 0.0:
    val = torch.clamp(val - dead_zone, min=0.0)
  if clip_max is not None:
    val = torch.clamp(val, max=clip_max)
  if normalize and clip_max is not None and clip_max > 0.0:
    val = val / clip_max
  return val


def fingertip_cube_contact_fraction(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contacts: int = 3,
  fingertip_indices: tuple[int, ...] | None = None,
) -> torch.Tensor:
  """Contact reward in [0, 1], saturating at max_contacts fingertips.

  Full reward (1.0) is given when max_contacts (default 3) or more fingertips are
  in contact, discouraging the policy from needing all 4 fingertips simultaneously.

  ``fingertip_indices`` optionally restricts the count to a subset of the
  sensor's primary slots. The slot order follows the resolved fingertip
  geom regex (verified at runtime for the TJ hand):
    0=thumb, 1=index, 2=middle, 3=ring, 4=pinky.
  NOTE (10-20_11-11 回退记录): a thumb-only variant of this was tried as
  reward "thumb_contact" (weight 0.03, drift-gated) and reverted — a binary
  per-step contact signal cannot distinguish dynamic pinch from static
  leaning, and the policy learned to keep the thumb glued to the cube.
  Kept as library code (unused); a duty-cycle band reward is the planned
  successor. See env_cfgs.py thumb_contact history block.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  found = sensor.data.found
  if fingertip_indices is not None:
    found = found[:, list(fingertip_indices)]
  in_contact = (found > 0).float()
  count = torch.sum(in_contact, dim=-1)
  return torch.clamp(count, max=max_contacts) / max_contacts


def object_fallen(
  env: ManagerBasedRlEnv,
  object_name: str,
  minimum_height: float,
) -> torch.Tensor:
  """Indicator that object center fell below minimum height."""
  cube: Entity = env.scene[object_name]
  return (cube.data.root_link_pos_w[:, 2] < minimum_height).float()


def joint_torque_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(".*",)),
  max_abs_torque: float | None = None,
  normalize: bool = False,
) -> torch.Tensor:
  """LEAPSim torque penalty term: sum(tau^2)."""
  asset: Entity = env.scene[asset_cfg.name]
  tau = asset.data.actuator_force[:, asset_cfg.joint_ids]
  if max_abs_torque is not None:
    tau = torch.clamp(tau, min=-max_abs_torque, max=max_abs_torque)
  val = torch.sum(torch.square(tau), dim=-1)
  if normalize:
    if max_abs_torque is not None and max_abs_torque > 0.0:
      denom = tau.shape[1] * (max_abs_torque**2)
    else:
      denom = max(tau.shape[1], 1)
    val = val / denom
  return val


class pose_diff_l2_from_reset:
  """Pose-diff penalty from reset with optional per-joint dead-zone."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self.asset: Entity = env.scene[self.asset_cfg.name]
    num_joints = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids].shape[1]
    self.init_joint_pos = torch.zeros(
      (env.num_envs, num_joints),
      device=env.device,
      dtype=self.asset.data.joint_pos.dtype,
    )
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self.init_joint_pos[env_ids] = self.asset.data.joint_pos[env_ids][
      :, self.asset_cfg.joint_ids
    ]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    average_per_joint: bool = True,
    joint_tolerance: float = 0.0,
    clip_max: float | None = None,
    normalize: bool = False,
  ) -> torch.Tensor:
    del env, asset_cfg
    q = self.asset.data.joint_pos[:, self.asset_cfg.joint_ids]
    joint_diff = torch.abs(q - self.init_joint_pos)
    if joint_tolerance > 0.0:
      # Ignore small deviations around reset; only penalize excess beyond tolerance.
      joint_diff = torch.clamp(joint_diff - joint_tolerance, min=0.0)
    val = torch.sum(torch.square(joint_diff), dim=-1)
    if average_per_joint:
      val = val / max(q.shape[1], 1)
    if clip_max is not None:
      val = torch.clamp(val, max=clip_max)
    if normalize and clip_max is not None and clip_max > 0.0:
      val = val / clip_max
    return val


class drift_gated_contact_fraction:
  """Fingertip contact fraction, paid ONLY while the cube stays near reset pose.

  STATUS (10-20_11-11 回退记录): tried as reward "thumb_contact" and reverted.
  The drift gate closed the "press + farm income" loophole but a binary
  per-step contact signal still cannot distinguish dynamic pinch from static
  leaning — the policy glued the thumb to the cube and collected the subsidy
  in quiet gaps (play-verified on run 2026-10-20_11-11-15). Kept as library
  code (unused); the successor should reward the contact-duty-cycle RHYTHM
  (contact-release-recontact band), not contact itself.

  An unconditional contact subsidy is dangerous: pressing a finger into the
  cube pushes it out, and if the contact income exceeds the drift penalty the
  "press + freeze" failure mode becomes profitable again (the jam the
  full-contact reward caused at weight 0.08). Gating on cube pose deviation
  closes that loophole: drift past the knee → subsidy stops, so pressing
  fingers into the cube to farm contact income is strictly unprofitable.

  Same per-slot indexing contract as fingertip_cube_contact_fraction: slot
  order follows the resolved fingertip geom regex (verified for the TJ hand:
  0=thumb, 1=index, 2=middle, 3=ring, 4=pinky).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._sensor_name: str = cfg.params["sensor_name"]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    cube: Entity = self._env.scene["cube"]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_contacts: int = 3,
    fingertip_indices: tuple[int, ...] | None = None,
    drift_position_threshold: float = 0.012,
    drift_tilt_threshold: float = 0.35,
  ) -> torch.Tensor:
    del sensor_name  # Bound via self._sensor_name from cfg.
    sensor: ContactSensor = self._env.scene[self._sensor_name]
    assert sensor.data.found is not None
    found = sensor.data.found
    if fingertip_indices is not None:
      found = found[:, list(fingertip_indices)]
    in_contact = (found > 0).float()
    count = torch.sum(in_contact, dim=-1)
    fraction = torch.clamp(count, max=max_contacts) / max_contacts

    # Drift gate: identical in-bounds test to object_cumulative_yaw.
    cube: Entity = self._env.scene["cube"]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )
    in_bounds = (pos_error <= drift_position_threshold) & (
      tilt_error <= drift_tilt_threshold
    )
    gated = torch.where(in_bounds, fraction, torch.zeros_like(fraction))
    return torch.where(self._has_init, gated, torch.zeros_like(gated))


class thumb_duty_cycle_band:
  """Reward the thumb's contact-RELEASE-recontact RHYTHM, not contact itself.

  History (see also fingertip_cube_contact_fraction / drift_gated_contact_
  fraction docstrings):
    - 18-52 run (no thumb reward): thumb detached entirely late in episodes
      → pinch lost → slip → cube_stalled.
    - 11-11 run (binary contact subsidy): thumb glued to the cube — a binary
      per-step signal cannot tell dynamic pinch from static leaning.

  Both failure modes are the two ENDS of one duty-cycle axis. The healthy
  in-hand rotation pattern is a periodic pinCH-release-recontact cycle:
  thumb contact rate over a sliding window sits in a mid band (~0.3-0.8),
  glued is ~1.0, detached is ~0.0. So the reward is a BAND function of the
  windowed contact rate: full pay inside [lo, hi], zero outside. It punishes
  both extremes symmetrically and encodes the visible rhythm directly.

  Details:
    - The band uses soft edges (cosine ramp of width ``edge_width``) so the
      gradient exists at the boundary — a hard 0/1 band gives no learning
      signal to a policy sitting just outside.
    - Drift gate identical to cumulative_yaw / drift_gated_contact_fraction:
      no pay while the cube is out of bounds, so leaning on the cube to farm
      the reward is unprofitable.
    - Warmup: the window is invalid until ``window_steps`` samples exist
      (early steps pay zero, not the ramp value of a partially-filled window).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._sensor_name: str = cfg.params["sensor_name"]
    window_steps = int(cfg.params.get("window_steps", 10))
    if window_steps < 2:
      raise ValueError(f"window_steps must be >= 2, got {window_steps}")
    self._window_steps = window_steps
    self._contact_hist = torch.zeros(
      (env.num_envs, window_steps), device=env.device
    )
    self._hist_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._hist_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    self._contact_hist[env_ids] = 0.0
    self._hist_ptr[env_ids] = 0
    self._hist_count[env_ids] = 0
    cube: Entity = self._env.scene["cube"]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = "fingertip_cube_contact",
    fingertip_indices: tuple[int, ...] = (0,),
    window_steps: int | None = None,
    band_lo: float = 0.2,
    band_hi: float = 0.8,
    edge_width: float = 0.15,
    drift_position_threshold: float = 0.012,
    drift_tilt_threshold: float = 0.35,
  ) -> torch.Tensor:
    del sensor_name, window_steps  # Bound at __init__ from cfg params.
    if self._window_steps < 2:
      raise ValueError("window_steps must be >= 2")

    sensor: ContactSensor = self._env.scene[self._sensor_name]
    assert sensor.data.found is not None
    found = sensor.data.found
    if fingertip_indices is not None:
      found = found[:, list(fingertip_indices)]
    in_contact = (found > 0).float()

    # Push current contact into the sliding window.
    ptr = self._hist_ptr
    self._contact_hist[torch.arange(env.num_envs, device=env.device), ptr] = (
      in_contact.squeeze(-1)
    )
    self._hist_ptr = (ptr + 1) % self._window_steps
    self._hist_count = torch.clamp(self._hist_count + 1, max=self._window_steps)

    # Warmup: pay nothing until the window is full (a partially filled
    # window biases the rate toward its initial zeros).
    full = self._hist_count >= self._window_steps
    contact_rate = self._contact_hist.mean(dim=-1)

    # Cosine soft band: 1 inside [lo, hi], cosine-ramped to 0 within
    # edge_width outside each edge. Gradient exists at the boundary.
    edge = max(edge_width, 1e-6)
    below = torch.clamp((contact_rate - (band_lo - edge)) / edge, 0.0, 1.0)
    above = torch.clamp(((band_hi + edge) - contact_rate) / edge, 0.0, 1.0)
    shape = 0.5 * (1.0 - torch.cos(math.pi * below)) * 0.5 * (
      1.0 - torch.cos(math.pi * above)
    )

    # Drift gate: identical in-bounds test to cumulative_yaw.
    cube: Entity = self._env.scene["cube"]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )
    in_bounds = (pos_error <= drift_position_threshold) & (
      tilt_error <= drift_tilt_threshold
    )

    reward = shape * in_bounds.float() * full.float()
    return torch.where(self._has_init, reward, torch.zeros_like(reward))


class object_pose_deviation_l2:
  """Penalize cube pose drift (position + roll/pitch tilt) from episode reset.

  This closes the reward gap that let the policy profit from pushing the cube
  into the radial-side pocket (MCT_CMC3/MCPI1/finger-root groove): no existing
  reward term directly charges for cube position drift, and the curriculum
  penalties (which only touch joints/object velocity) stay at weight_min
  until the rotation-progress curriculum activates. This term is active from
  step 0 and has no curriculum, so drifting into the pocket carries a
  continuous per-step cost at every stage of training.

  The penalty is linear in each error component (not squared) so that the
  marginal cost of drifting deeper into the pocket stays constant instead of
  saturating the gradient far away from reset.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    cube: Entity = self._env.scene["cube"]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str = "cube",
    position_weight: float = 1.0,
    tilt_weight: float = 0.2,
    position_threshold: float = 0.005,
    tilt_threshold: float = 0.05,
    position_knee: float | None = None,
    position_far_weight: float = 1.0,
  ) -> torch.Tensor:
    del object_name  # Cube is bound in reset(); kept for cfg symmetry.

    cube: Entity = self._env.scene["cube"]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)

    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )

    # Dead-zone below the thresholds: small in-hand jiggle while rotating is
    # free, only genuine drift toward the pocket is charged.
    pos_excess = torch.clamp(pos_error - position_threshold, min=0.0)
    tilt_excess = torch.clamp(tilt_error - tilt_threshold, min=0.0)

    # Optional two-zone position cost: within the knee the slope is
    # ``position_weight`` (gentle, tolerates rotation jitter); beyond the
    # knee the slope steepens to ``position_far_weight`` (pocket-approach
    # zone becomes expensive). With position_knee=None this is a plain
    # linear cost, identical to the original behavior.
    if position_knee is not None:
      knee_excess = max(position_knee - position_threshold, 0.0)
      near = torch.clamp(pos_excess, max=knee_excess)
      far = torch.clamp(pos_excess - knee_excess, min=0.0)
      pos_cost = position_weight * near + position_far_weight * far
    else:
      pos_cost = position_weight * pos_excess

    cost = pos_cost + tilt_weight * tilt_excess
    return torch.where(self._has_init, cost, torch.zeros_like(cost))


class object_cumulative_yaw:
  """Reward for ACCUMULATED rotation angle, not instantaneous rate.

  The rate-based ``object_yaw_finite_diff_clipped`` pays for "spinning fast
  right now" but has no memory: a policy that wiggles or spins very slowly
  (0.02 rad/s, invisible over a 20 s episode) earns the same as one that
  truly turns the cube. This term pays for progress that a human can SEE:
  the signed yaw integral since episode start, unwrapped, with a configurable
  dead-zone so that pure jiggle integrates to ~0.

  Design properties:
    - The integral is SIGNED: backward rotation cancels earlier progress, so
      "rotate forward / slip back" cycles earn nothing here either.
    - Monotone in true rotation: the only way to accumulate score is to
      actually turn the cube further than ever before in this episode.
      There is no rate to game (unlike low-rate boosting of a clipped rate
      reward, which let slow spins collect the same as fast ones).
    - Optional drift gate: accumulated angle only pays while the cube stays
      near its initial pose; drifting while turning stops the meter.
    - Optional saturation per "lap" (2*pi) keeps the term bounded and makes
      each additional full turn worth the same.

  Reward shape: a STEP bonus each time the unwrapped integral crosses
  ``step_angle`` (default quarter turn), scaled by the current drift gate.
  Step bonuses (instead of a continuous payoff) make the reward robust to
  the sampling rate of yaw crossing and give a crisp, countable event:
  the policy literally gets paid per quarter-turn milestone reached.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.object_name = cfg.params["object_name"]
    self._env = env
    self._cum_yaw = torch.zeros(env.num_envs, device=env.device)
    self._last_paid = torch.zeros(env.num_envs, device=env.device)
    # self._stall_steps = torch.zeros(env.num_envs, device=env.device)   # ← 新增
    self._yaw_prev = torch.zeros(env.num_envs, device=env.device)
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    cube: Entity = self._env.scene[self.object_name]
    yaw = self._current_yaw()
    self._cum_yaw[env_ids] = 0.0
    self._last_paid[env_ids] = 0.0
    # self._stall_steps[env_ids] = 0.0                                    # ← 新增
    self._yaw_prev[env_ids] = yaw[env_ids]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def _current_yaw(self) -> torch.Tensor:
    cube: Entity = self._env.scene[self.object_name]
    _, _, yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)
    return yaw

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    object_name: str,
    negate_yaw_rate: bool = True,
    step_angle: float = math.pi / 2,
    bonus: float = 1.0,
    drift_position_threshold: float = 0.02,
    drift_tilt_threshold: float = 0.35,
    max_bonus_per_step: float = 1.5,
    # stall_penalty: float = 0.05,          # ← 新增:停滞时每步罚分
    # stall_steps_threshold: int = 25,      # ← 新增:连续多少步无进展开始罚(25 步 = 1.25s)
  ) -> torch.Tensor:
    del object_name  # Bound in __init__.

    yaw = self._current_yaw()
    # Re-anchor on the first post-reset step.
    reset_mask = env.episode_length_buf <= 1
    if torch.any(reset_mask):
      self.reset(reset_mask.nonzero(as_tuple=False).squeeze(-1))

    delta = wrap_to_pi(yaw - self._yaw_prev)
    self._yaw_prev[:] = yaw
    if negate_yaw_rate:
      delta = -delta
    self._cum_yaw += delta
    # Pay a bonus for each new quarter-turn milestone beyond the last paid one.
    progress = self._cum_yaw - self._last_paid
    steps_paid = torch.clamp(
      torch.floor(progress / step_angle), min=0.0
    )
    reward = steps_paid * bonus

    # ---- 新增:stall penalty 块(插在 reward = steps_paid * bonus 之后)----
    # Progress this step = |signed delta| (any genuine rotation counts, either
    # direction, since backward already cancels the integral itself).
    # progressed = delta.abs() > 1e-4
    # # Reset the stall counter on steps with real rotation; increment otherwise.
    # self._stall_steps = torch.where(
    #   progressed,
    #   torch.zeros_like(self._stall_steps),
    #   torch.clamp(self._stall_steps + 1, max=stall_steps_threshold * 2),
    # )
    # stall = (self._stall_steps >= stall_steps_threshold).to(reward.dtype)
    # reward = reward - stall_penalty * stall
    # ---- 新增块结束 ----

    # Drift gate: only pay while the cube stays near its initial pose.
    cube: Entity = self._env.scene[self.object_name]
    pos_w = cube.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(cube.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )
    in_bounds = (pos_error <= drift_position_threshold) & (
      tilt_error <= drift_tilt_threshold
    )
    reward = torch.where(in_bounds, reward, torch.zeros_like(reward))
    reward = torch.where(self._has_init, reward, torch.zeros_like(reward))

    # Mark milestones as paid and cap the per-step payout.
    self._last_paid += steps_paid * step_angle
    return torch.clamp(reward, max=max_bonus_per_step)
