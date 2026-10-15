"""Collect size-conditioned hand-cube grasp cache by passive settling.

For each cube half-size on a regular grid, place the cube at the default
spawn pose with the hand at its home keyframe, simulate passive settling,
and record:

    (cube_size, joint_pos, cube_pose_rel)

for stable grasps.

The output cache is used by ``hand_cube_mdp.reset_from_grasp_cache``.

This script is hand-agnostic. Robot-specific information such as the palm
geometry expression and output cache path can be provided through command
line arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class CollectConfig:
  """Configuration for grasp-cache collection."""

  # ---------------------------------------------------------------------------
  # Output
  # ---------------------------------------------------------------------------

  out_file: str = "src/in_hand_rotation_mjlab/tasks/hand_cube/cache/wuji_left_grasp_cache.npz"

  # ---------------------------------------------------------------------------
  # Simulation
  # ---------------------------------------------------------------------------

  num_envs: int = 256
  settle_steps: int = 240
  device: str | None = None
  seed: int = 1

  # ---------------------------------------------------------------------------
  # Cube size grid
  #
  # ``size_min`` and ``size_max`` are cube HALF-SIZES.
  # For example:
  #
  #   half_size = 0.0375
  #
  # corresponds to a cube with side length:
  #
  #   2 * 0.0375 = 0.075 m
  # ---------------------------------------------------------------------------

  size_min: float = 0.020
  size_max: float = 0.0276
  num_sizes: int = 32

  # ---------------------------------------------------------------------------
  # Stability filters
  # ---------------------------------------------------------------------------

  lin_vel_threshold: float = 0.08
  ang_vel_threshold: float = 1.2

  # Palm geometry used for the distance check.
  #
  # This must be a MuJoCo geom-name regular expression.
  #
  # For Wuji, for example, this may be:
  #
  #   "palm_collision_.*"
  #
  # If no palm geometry matches, the script falls back to the palm body.
  #
  palm_center_geom_expr: str = "left_palm_collision"

  max_palm_distance: float = 0.14
  min_cube_height: float = 0.05

  # ---------------------------------------------------------------------------
  # Robot-specific geometry names
  # ---------------------------------------------------------------------------

  palm_body_name: str = "left_palm_link"
  cube_geom_name: str = "cube_geom"


def run_collect(task_id: str, cfg: CollectConfig) -> None:
  """Collect stable grasp states for one hand/cube task."""

  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # ---------------------------------------------------------------------------
  # Load environment configuration
  # ---------------------------------------------------------------------------

  env_cfg = load_env_cfg(task_id, play=False)

  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs

  # ---------------------------------------------------------------------------
  # Disable terminations during passive settling.
  #
  # We do NOT want the environment to reset before the stability test.
  # ---------------------------------------------------------------------------

  if env_cfg.terminations is not None:
    for key in ("cube_fell", "cube_far_from_palm"):
      if key in env_cfg.terminations:
        env_cfg.terminations[key] = None

  # ---------------------------------------------------------------------------
  # Disable reset events related to grasp cache and cube-size randomization.
  #
  # We explicitly control cube size below.
  # ---------------------------------------------------------------------------

  if env_cfg.events is not None:
    for key in ("reset_from_grasp_cache", "randomize_cube_size_once"):
      if key in env_cfg.events:
        env_cfg.events[key] = None

  # ---------------------------------------------------------------------------
  # Make sure the episode is long enough for passive settling.
  # ---------------------------------------------------------------------------

  step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
  required_ep_len_s = (cfg.settle_steps + 2) * step_dt
  env_cfg.episode_length_s = max(env_cfg.episode_length_s, required_ep_len_s)

  # ---------------------------------------------------------------------------
  # Build environment
  # ---------------------------------------------------------------------------

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

  robot = env.scene["robot"]
  cube = env.scene["cube"]

  action_dim = env.action_manager.total_action_dim

  # ---------------------------------------------------------------------------
  # Resolve palm geometry/body
  # ---------------------------------------------------------------------------

  palm_geom_ids, _ = robot.find_geoms(cfg.palm_center_geom_expr, preserve_order=True)
  palm_body_ids, _ = robot.find_bodies(cfg.palm_body_name, preserve_order=True)

  if len(palm_body_ids) != 1:
    raise RuntimeError(
      f"Could not uniquely resolve palm body '{cfg.palm_body_name}'. "
      f"Found {len(palm_body_ids)} bodies."
    )

  palm_body_id = palm_body_ids[0]

  use_palm_geom_center = len(palm_geom_ids) > 0

  if use_palm_geom_center:
    print(
      f"[collect] using palm geometries: '{cfg.palm_center_geom_expr}' "
      f"({len(palm_geom_ids)} geoms)"
    )
  else:
    print(
      f"[collect] no palm geometry matched '{cfg.palm_center_geom_expr}', "
      f"falling back to body '{cfg.palm_body_name}'"
    )

  # ---------------------------------------------------------------------------
  # Resolve cube geom
  # ---------------------------------------------------------------------------

  cube_geom_local_ids, _ = cube.find_geoms(cfg.cube_geom_name, preserve_order=True)

  if len(cube_geom_local_ids) != 1:
    raise RuntimeError(
      f"Could not uniquely resolve cube geom '{cfg.cube_geom_name}'. "
      f"Found {len(cube_geom_local_ids)} geoms."
    )

  cube_geom_world_id = int(cube.indexing.geom_ids[cube_geom_local_ids[0]].item())

  # ---------------------------------------------------------------------------
  # Build cube size grid
  # ---------------------------------------------------------------------------

  sizes = np.linspace(cfg.size_min, cfg.size_max, cfg.num_sizes, dtype=np.float32)

  # ---------------------------------------------------------------------------
  # Output buffers
  # ---------------------------------------------------------------------------

  size_buf: list[np.ndarray] = []
  joint_buf: list[np.ndarray] = []
  pose_rel_buf: list[np.ndarray] = []

  # ---------------------------------------------------------------------------
  # Zero action
  #
  # During grasp-cache collection the hand is not actively controlled.
  # The hand starts from its normal environment reset pose/home pose and
  # the cube is allowed to passively settle.
  # ---------------------------------------------------------------------------

  zero_action = torch.zeros((cfg.num_envs, action_dim), device=env.device)
  all_env_ids = torch.arange(cfg.num_envs, device=env.device)

  total_kept = 0

  # ---------------------------------------------------------------------------
  # Iterate over cube sizes
  # ---------------------------------------------------------------------------

  for si, half_size in enumerate(sizes):

    # ---------------------------------------------------------------
    # Set cube size for all environments.
    #
    # MuJoCo box geom size is half-size.
    # ---------------------------------------------------------------

    env.sim.model.geom_size[all_env_ids, cube_geom_world_id, :] = float(half_size)

    # ---------------------------------------------------------------
    # Reset environment.
    #
    # The normal environment reset determines:
    #
    #   - hand home pose
    #   - initial joint configuration
    #   - cube spawn position
    #   - any reset-time noise
    #
    # We intentionally do not use grasp-cache reset here.
    # ---------------------------------------------------------------

    env.reset()

    # ---------------------------------------------------------------
    # Passive settling.
    #
    # Zero action means that the hand follows its current reset state
    # without an additional commanded joint displacement.
    # ---------------------------------------------------------------

    for _ in range(cfg.settle_steps):
      env.step(zero_action)

    # -----------------------------------------------------------------
    # Read cube state
    # -----------------------------------------------------------------

    cube_pos_w = cube.data.root_link_pos_w
    cube_quat_w = cube.data.root_link_quat_w

    cube_lin_vel_w = cube.data.root_link_lin_vel_w
    cube_ang_vel_w = cube.data.root_link_ang_vel_w

    # -----------------------------------------------------------------
    # Calculate palm center
    # -----------------------------------------------------------------

    if use_palm_geom_center:
      palm_center_w = robot.data.geom_pos_w[:, palm_geom_ids].mean(dim=1)

    else:
      palm_center_w = robot.data.body_link_pos_w[:, palm_body_id]

    # -----------------------------------------------------------------
    # Cube-to-palm distance
    # -----------------------------------------------------------------

    cube_dist = torch.norm(cube_pos_w - palm_center_w, dim=-1)

    # -----------------------------------------------------------------
    # Stability test
    # -----------------------------------------------------------------

    stable = torch.ones(cfg.num_envs, dtype=torch.bool, device=env.device)
    stable &= env.episode_length_buf >= cfg.settle_steps
    stable &= torch.norm(cube_lin_vel_w, dim=-1) <= cfg.lin_vel_threshold
    stable &= torch.norm(cube_ang_vel_w, dim=-1) <= cfg.ang_vel_threshold
    stable &= cube_dist <= cfg.max_palm_distance
    stable &= cube_pos_w[:, 2] >= cfg.min_cube_height

    # -----------------------------------------------------------------
    # Fingertip contact
    # -----------------------------------------------------------------
    fingertip_sensor = env.scene["fingertip_cube_contact"]
    assert fingertip_sensor.data.found is not None
    fingertip_found = fingertip_sensor.data.found > 0
    fingertip_contact_count = fingertip_found.sum(dim=-1)
    stable &= fingertip_contact_count >= 4

    # palm cube contact
    # palm_sensor = env.scene["palm_cube_contact"]
    # assert palm_sensor.data.found is not None
    # palm_in_contact = palm_sensor.data.found > 0
    # stable &= ~palm_in_contact.any(dim=-1)
    
    # -----------------------------------------------------------------
    # Extract stable environments
    # -----------------------------------------------------------------

    stable_ids = stable.nonzero(as_tuple=False).squeeze(-1)
    kept = int(stable_ids.numel())

    if kept > 0:

      # ---------------------------------------------------------------
      # Cube pose relative to the environment origin.
      #
      # Stored as:
      #
      #   [x, y, z, qw, qx, qy, qz]
      #
      # assuming MuJoCo's quaternion convention used by this project.
      # ---------------------------------------------------------------

      cube_pose_rel = torch.cat(
        [
          cube_pos_w[stable_ids] - env.scene.env_origins[stable_ids],
          cube_quat_w[stable_ids],
        ],
        dim=-1,
      )

      # ---------------------------------------------------------------
      # Store:
      #
      #   cube_size
      #   joint_pos
      #   cube_pose_rel
      # ---------------------------------------------------------------

      size_buf.append(np.full(kept, half_size, dtype=np.float32))
      joint_buf.append(robot.data.joint_pos[stable_ids].cpu().numpy())
      pose_rel_buf.append(cube_pose_rel.cpu().numpy())

    total_kept += kept

    print(
      f"[collect] size {si + 1}/{cfg.num_sizes}  "
      f"half_size={half_size:.5f}  kept={kept}/{cfg.num_envs}  total={total_kept}"
    )

  # ---------------------------------------------------------------------------
  # Close environment
  # ---------------------------------------------------------------------------

  env.close()

  # ---------------------------------------------------------------------------
  # Check result
  # ---------------------------------------------------------------------------

  if total_kept == 0:
    raise RuntimeError(
      "No stable samples collected. "
      "Check the following parameters:\n"
      "  - hand home pose\n"
      "  - cube spawn position\n"
      "  - cube size range\n"
      "  - settle_steps\n"
      "  - palm_center_geom_expr\n"
      "  - max_palm_distance\n"
      "  - min_cube_height"
    )

  # ---------------------------------------------------------------------------
  # Save cache
  # ---------------------------------------------------------------------------

  out_path = Path(cfg.out_file)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  cube_sizes = np.concatenate(size_buf).astype(np.float32)
  joint_positions = np.concatenate(joint_buf).astype(np.float32)
  cube_poses = np.concatenate(pose_rel_buf).astype(np.float32)

  np.savez_compressed(
    out_path,
    cube_size=cube_sizes,
    joint_pos=joint_positions,
    cube_pose_rel=cube_poses,
    joint_names=np.asarray(robot.joint_names),
  )

  print()
  print("=" * 72)
  print("[collect] grasp cache generation finished")
  print(f"[collect] samples    : {total_kept}")
  print(f"[collect] cube sizes : {cfg.num_sizes}")
  print(f"[collect] joint dim  : {joint_positions.shape[1]}")
  print(f"[collect] output     : {out_path}")
  print("=" * 72)


def main() -> None:
  import mjlab.tasks  # noqa: F401
  import in_hand_rotation_mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  cfg = tyro.cli(
    CollectConfig,
    args=remaining_args,
    default=CollectConfig(),
    prog=f"scripts/collect_hand_cube_grasp_cache_wuji.py {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )

  run_collect(chosen_task, cfg)


if __name__ == "__main__":
  main()