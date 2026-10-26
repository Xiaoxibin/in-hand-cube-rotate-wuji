"""Headless rollout eval for one or more checkpoints.

Usage:
  uv run python scripts/eval_ckpt.py --steps 600 --num-envs 16 \
      label=2026-10-22_16-55-22/model_24699.pt [more label=path ...]

Reports, per checkpoint: cumulative cube yaw since reset, per-env min/max,
and a termination breakdown (which term fired, how many times).
"""

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401
import in_hand_rotation_mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi
from mjlab.utils.torch import configure_torch_backends

TASK = "Mjlab-TJ-Right-HandCube-Rotate"
LOG_ROOT = "logs/rsl_rl/tj_right_hand_cube_rotate/"


def _dbg_counts(term_mgr, idx: torch.Tensor) -> dict[str, int]:
  out: dict[str, int] = {}
  for name, flags in (getattr(term_mgr, "_term_dones", None) or {}).items():
    if torch.is_tensor(flags) and flags.shape[:1] == idx.shape:
      c = int((flags[idx] != 0).sum().item())
      if c:
        out[str(name)] = c
  return out


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoints", nargs="+", help="label=relpath")
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--num-envs", type=int, default=16)
  args = parser.parse_args()

  configure_torch_backends()
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = args.num_envs
  cfg.sim.nconmax = 300
  cfg.sim.njmax = 1500
  env_raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  agent_cfg = load_rl_cfg(TASK)
  env = RslRlVecEnvWrapper(env_raw, clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cuda:0")

  # Probe termination flags during a short rollout to learn their layout.
  term_mgr = env_raw.termination_manager
  n = args.num_envs
  cube = env_raw.scene["cube"]

  for spec in args.checkpoints:
    label, path = spec.split("=", 1)
    env.reset()
    runner.load(LOG_ROOT + path, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")

    cum = torch.zeros(n, device="cuda:0")
    prev = None
    episode_len = torch.zeros(n, device="cuda:0")
    episode_yaw = torch.zeros(n, device="cuda:0")
    done_counts: dict[str, int] = {}
    logged_lens: list[float] = []
    logged_yaw: list[float] = []

    for t in range(args.steps):
      obs = env.get_observations()
      with torch.no_grad():
        act = policy(obs)
      # mjlab re-evaluates termination terms on the post-reset step, so the
      # flags the TerminationsManager holds at a reset describe the fresh
      # state. To attribute a reset we only distinguish time_out (truncation)
      # from everything else; the stalled check below tells the rest apart.
      _, _, done, _ = env.step(act)
      yaw = euler_xyz_from_quat(cube.data.root_link_quat_w)[2]
      if prev is not None:
        d = wrap_to_pi(yaw - prev)
        cum += d
        episode_yaw += d
      prev = yaw
      episode_len += 1

      if (done > 0).any():
        idx = done.bool()
        counts = _dbg_counts(term_mgr, idx)
        for name, c in counts.items():
          done_counts[name] = done_counts.get(name, 0) + c
        logged_lens.extend(episode_len[idx].tolist())
        logged_yaw.extend(episode_yaw[idx].tolist())
        episode_len[idx] = 0
        episode_yaw[idx] = 0

      if (t + 1) % 100 == 0:
        print(
          f"  {label} t={(t + 1) * 0.05:6.1f}s cum {cum.mean():+.2f} "
          f"min {cum.min():+.2f} max {cum.max():+.2f} "
          f"episodes={len(logged_lens)}",
          flush=True,
        )

    lens = torch.tensor(logged_lens) if logged_lens else torch.zeros(1)
    yaws = torch.tensor(logged_yaw) if logged_yaw else torch.zeros(1)
    rate = yaws.sum() / max(lens.sum().item() * env_raw.step_dt, 1e-6)
    print(
      f"EVAL {label}: cum {cum.mean():+.2f} rad | episodes {len(lens)} | "
      f"mean_len {lens.mean() * env_raw.step_dt:.1f}s | "
      f"yaw/episode {yaws.mean():+.2f} rad | rate {rate:.3f} rad/s | "
      f"term counts {done_counts}",
      flush=True,
    )


if __name__ == "__main__":
  main()
