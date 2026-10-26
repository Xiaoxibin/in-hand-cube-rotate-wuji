from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_CUBE_CFG = SceneEntityCfg("cube")


def object_linear_speed_above(
  env: ManagerBasedRlEnv,
  max_linear_speed: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
) -> torch.Tensor:
  """Terminate when object linear speed exceeds a threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
  return speed > max_linear_speed


class object_yaw_stalled:
  """Terminate when the cube has not genuinely rotated for a while.

  The frozen-grasp local optimum (thumb pressed on top, ring/pinky wedged at
  the side) pays only mild per-step penalties, so a stalled episode is only
  mildly negative and the policy keeps occupying it. Terminating stalled
  episodes early makes the frozen state expensive (no future reward left in
  it), which no amount of reward shaping achieves as cleanly.

  Design:
    - Uses a dead-zoned CUMULATIVE |yaw| delta over a sliding window, not
      instantaneous angular velocity: slow-but-real rotation (a legal
      intermediate behavior, cf. low_rate_boost) must not be killed. Only
      window_steps of near-zero net motion trigger termination.
    - The dead-zone (per-step epsilon) exempts measurement/settling jitter:
      jitter integrates to ~0 over the window and still counts as stalled.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_CUBE_CFG)
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._yaw_prev = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    # Params that only shape internal buffers are read once here.
    self._window_steps = int(cfg.params.get("window_steps", 50))
    self._yaw_buffer = torch.zeros(
      (env.num_envs, self._window_steps), device=env.device
    )
    self._buf_ptr = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # Only judge stall once the episode is older than ``warmup_steps`` (by
    # default one full window, 2.5 s); before that the buffer is only
    # partially filled and sums small by construction.
    self._warmup_steps = int(cfg.params.get("warmup_steps", self._window_steps))
    self._age = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._session = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self._session_id = 0
    self.reset(None)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)
    yaw = self._current_yaw()
    self._yaw_prev[env_ids] = yaw[env_ids]
    self._yaw_buffer[env_ids] = 0.0
    self._buf_ptr[env_ids] = 0
    self._has_init[env_ids] = True
    self._age[env_ids] = 0
    self._session_id += 1
    self._session[env_ids] = self._session_id

  def _current_yaw(self) -> torch.Tensor:
    _, _, yaw = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    return yaw

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_steps: int = 50,
    min_yaw_delta: float = 0.05,
    per_step_epsilon: float = 0.0005,
    asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
  ) -> torch.Tensor:
    del env, asset_cfg  # Bound once in __init__.
    del window_steps  # Fixed at __init__ time via cfg params.

    # The manager re-evaluates every term right after an auto-reset, so this
    # is the first call of a new episode for the envs that just reset. Do not
    # feed that call into the window: it would inject the cross-reset yaw jump
    # as fake motion and attribute the reset to whichever term the fresh state
    # happens to trigger (the stall flag was observed on time_out resets).
    if self._has_init.all():
      self.reset(self._session != self._episode_buf)

    yaw = self._current_yaw()
    delta = wrap_to_pi(yaw - self._yaw_prev)
    self._yaw_prev[:] = yaw

    # Dead-zone: sub-epsilon jitter does not count as progress.
    real_delta = torch.where(
      delta.abs() > per_step_epsilon, delta, torch.zeros_like(delta)
    )
    # Rolling window of real yaw deltas.
    self._yaw_buffer[self._env_ids_buf, self._buf_ptr] = real_delta
    self._buf_ptr = (self._buf_ptr + 1) % self._window_steps
    # Net motion over the window: signed sum so that rotate-forward/slip-back
    # cycles (which a human would not see as progress) count as stalled too.
    net_motion = self._yaw_buffer.sum(dim=-1).abs()
    # Judge only once the window is full for this episode. A partially filled
    # window sums small by construction, so `_has_init` alone let a policy
    # that rotates just under the threshold stay "stalled" forever and earn a
    # free reset every warmup_steps. `_age` therefore persists across the
    # re-anchor of the episode-creating call (do not reset it there).
    self._age += 1
    stalled = self._has_init & (self._age > self._warmup_steps) & (
      net_motion < min_yaw_delta
    )
    return stalled

  @property
  def _episode_buf(self) -> torch.Tensor:
    return self._env.episode_length_buf

  # arange buffer reused each call; created lazily to keep __init__ simple.
  @property
  def _env_ids_buf(self):
    if not hasattr(self, "_env_ids_buf_val"):
      self._env_ids_buf_val = torch.arange(
        self._yaw_buffer.shape[0], device=self._yaw_buffer.device
      )
    return self._env_ids_buf_val


class object_pose_rp_position_deviation_from_reset:
  """Terminate when object drifts from episode-initial pose (pos + roll/pitch)."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_CUBE_CFG)
    self._asset: Entity = env.scene[self._asset_cfg.name]
    self._init_pos_w = torch.zeros((env.num_envs, 3), device=env.device)
    self._init_roll = torch.zeros(env.num_envs, device=env.device)
    self._init_pitch = torch.zeros(env.num_envs, device=env.device)
    self._has_init = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None):
    if env_ids is None:
      env_ids = slice(None)

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    self._init_pos_w[env_ids] = pos_w[env_ids]
    self._init_roll[env_ids] = roll[env_ids]
    self._init_pitch[env_ids] = pitch[env_ids]
    self._has_init[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    max_position_error: float,
    max_tilt_error: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_CUBE_CFG,
  ) -> torch.Tensor:
    del env, asset_cfg  # Bound once in __init__.

    pos_w = self._asset.data.root_link_pos_w
    roll, pitch, _ = euler_xyz_from_quat(self._asset.data.root_link_quat_w)
    pos_error = torch.linalg.vector_norm(pos_w - self._init_pos_w, dim=-1)
    roll_error = wrap_to_pi(roll - self._init_roll).abs()
    pitch_error = wrap_to_pi(pitch - self._init_pitch).abs()
    tilt_error = torch.linalg.vector_norm(
      torch.stack([roll_error, pitch_error], dim=-1), dim=-1
    )
    return self._has_init & (
      (pos_error > max_position_error) | (tilt_error > max_tilt_error)
    )
