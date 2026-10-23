#!/usr/bin/env python
"""diff_obs.py -- compare per-term observations: training env vs native sim2sim.

Builds the same checkpoint twice -- once through the training environment and
once through the native sim2sim client + policy server -- then compares the
actor observation term by term and history frame by history frame. Use it to
locate sim2sim parity bugs (sign, offset, history order, stale defaults).

Run from the repo root:
    MUJOCO_GL=egl uv run python tools/diff_obs.py \
        --checkpoint logs/rsl_rl/<exp>/<run>/model_XXXX.pt \
        --task Mjlab-TJ-Right-HandCube-Rotate
"""
from __future__ import annotations

import argparse

import numpy as np


def build_env_obs(task: str, ckpt: str, seed: int = 0):
    from dataclasses import asdict

    import in_hand_rotation_mjlab.tasks  # noqa: F401  (populate task registry)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from rsl_rl.runners import OnPolicyRunner

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = seed
    acfg = load_rl_cfg(task)
    env = RslRlVecEnvWrapper(
        ManagerBasedRlEnv(cfg=cfg, device="cuda:0"), clip_actions=acfg.clip_actions
    )
    agent = asdict(acfg)
    for key in ("actor", "critic"):
        if isinstance(agent.get(key), dict) and agent[key].get(
            "class_name", "MLPModel"
        ) != "CNNModel":
            agent[key].pop("cnn_cfg", None)
    runner = OnPolicyRunner(env, agent, "cuda:0")
    runner.load(ckpt, map_location="cuda:0")
    env.reset()
    return env, runner, env.get_observations()["actor"].cpu().numpy().ravel()


def build_s2s_obs(task: str, ckpt: str):
    from in_hand_rotation_mjlab.policy_server import (
        MjlabPolicyServer,
        PolicyServerConfig,
    )
    from in_hand_rotation_mjlab.sim2sim.native.client import NativeMujocoClient
    from in_hand_rotation_mjlab.sim2sim.native.config import NativeSim2SimConfig

    cfg = NativeSim2SimConfig(checkpoint_file=ckpt, device="cuda:0")
    client = NativeMujocoClient(task, cfg)
    server = MjlabPolicyServer(
        task,
        PolicyServerConfig(
            checkpoint_file=ckpt,
            device="cuda:0",
            play=True,
            # Must match NativeMujocoSim2SimRunner: the native client sends
            # joint_pos_rel, so the server must not subtract default_joint_pos.
            raw_joint_pos_obs=False,
        ),
        action_term_metadata=client.build_server_action_metadata(),
        step_dt=client.step_dt,
        max_episode_steps=client.max_episode_steps,
    )
    packet = client.reset_rollout()
    server.reset(packet)
    obs = server.obs_processor.compute(packet.term_values).cpu().numpy().ravel()
    return client, server, packet, obs


def split_terms(
    arr: np.ndarray, names, dims, history_length: int
) -> dict[str, np.ndarray]:
    """Split a flattened actor obs into per-term (history, dim) views.

    ``group_obs_term_dim`` already folds the history dimension into each
    term's shape (e.g. (10, 21) for a 21-DoF term with history_length=10), so
    ``dims`` here are per-term TOTAL sizes. Flattened history is laid out per
    term from oldest to newest.
    """
    out, offset = {}, 0
    for name, total in zip(names, dims):
        per_frame = total // history_length
        out[name] = arr[offset : offset + total].reshape(history_length, per_frame)
        offset += total
    if offset != arr.size:
        print(f"[WARN] consumed {offset} of {arr.size} observation dims")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Mjlab-TJ-Right-HandCube-Rotate")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env, _, env_obs = build_env_obs(args.task, args.checkpoint, args.seed)
    client, _server, packet, s2s_obs = build_s2s_obs(args.task, args.checkpoint)

    obs_manager = env.unwrapped.observation_manager
    names = list(obs_manager.active_terms["actor"])
    # group_obs_term_dim maps group -> [per-term TOTAL dim tuples]. History is
    # already folded in (e.g. (10, 21)), and the batch dim is excluded.
    dims = [int(np.prod(d)) for d in obs_manager.group_obs_term_dim["actor"]]
    hist_len = int(env.unwrapped.cfg.observations["actor"].history_length)

    print(f"terms: {list(zip(names, dims))}  history_length={hist_len}")
    print(
        f"env actor obs dim={env_obs.size}  s2s actor obs dim={s2s_obs.size}"
        f"  per-frame={[d // hist_len for d in dims]}"
    )

    env_terms = split_terms(env_obs, names, dims, hist_len)
    s2s_terms = split_terms(s2s_obs, names, dims, hist_len)

    print(f"\n{'term':<28}{'max|env|':>10}{'max|s2s|':>10}{'max diff':>10}")
    for name in names:
        diff = np.abs(env_terms[name] - s2s_terms[name]).max()
        print(
            f"{name:<28}{np.abs(env_terms[name]).max():>10.4f}"
            f"{np.abs(s2s_terms[name]).max():>10.4f}{diff:>10.4f}"
        )

    print("\n=== per-history-frame (oldest -> newest), first 3 joints ===")
    for name in names:
        print(f"[{name}]")
        for i in range(hist_len):
            print(
                f"  hist[{i}] env={np.round(env_terms[name][i][:3], 4)}"
                f"  s2s={np.round(s2s_terms[name][i][:3], 4)}"
            )

    print("\n=== raw sim2sim terms (before scale/history) ===")
    for key, value in packet.term_values.items():
        print(f"  {key}: {np.round(value.cpu().numpy().ravel()[:5], 4)}")

    print("\n=== state identity check ===")
    entity = client.state.entity("robot")
    ids = entity.joint_qpos_adr.cpu().numpy()
    qpos = client.data.qpos[ids]
    default_pos = entity.default_joint_pos.cpu().numpy().ravel()
    print("  qpos[hand]               :", np.round(qpos[:5], 4))
    print("  entity.default_joint_pos :", np.round(default_pos[:5], 4))
    print(
        "  state.joint_pos          :",
        np.round(client.state.joint_pos("robot").cpu().numpy().ravel()[:5], 4),
    )
    print("  qpos - default           :", np.round(qpos[:5] - default_pos[:5], 4))
    if client._init_key_id is not None:
        key_pos = client.model.key_qpos[client._init_key_id, ids]
        print("  model.key_qpos[init]     :", np.round(key_pos[:5], 4))
    else:
        print("  model.key_qpos[init]     : <no keyframe>")

    probe_s2s_pipeline(_server, client)


def probe_s2s_pipeline(server, client) -> None:
    """Trace each sim2sim observation term from packet value to policy input.

    The server processor applies, in order: ``obs - offset``, clip, scale,
    delay, history. ``offset`` is set for terms whose name matches an action
    term (see policy_server/observation.py). This trace shows which stage
    introduces any divergence from the training environment.
    """
    print("\n=== sim2sim per-term pipeline trace ===")
    adapter = server.obs_processor
    terms = getattr(adapter, "_terms", None) or getattr(adapter, "terms", None)
    if terms is None:
        print("  <cannot introspect obs processor>")
        return

    for term in terms:
        cfg = term.cfg
        print(f"[{term.name}]")
        print(
            f"  cfg.clip={cfg.clip} cfg.scale={cfg.scale}"
            f" history_length={cfg.history_length}"
            f" flatten_history_dim={cfg.flatten_history_dim}"
        )

        # 1) The offset the server applies. This is the suspect: the training
        #    env's joint_pos term is ALREADY relative to default_joint_pos, so
        #    subtracting default_joint_pos again here would sign-flip it.
        if term.offset is not None:
            print(f"  >>> term.offset        = {np.round(term.offset.cpu().numpy().ravel()[:5], 4)}")
        else:
            print("  >>> term.offset        = None")

        # 2) Pick the evaluator that produced the packet value for this term.
        evaluator = None
        for t in getattr(client.obs_adapter, "_terms", []):
            if t.name == term.name:
                evaluator = t.evaluator
                break
        if evaluator is not None:
            raw = evaluator().detach().cpu().numpy().ravel()
            print(f"  client evaluator()     = {np.round(raw[:5], 4)}  (what the packet carries)")

        # 3) History contents actually handed to the policy.
        buf = term.history_buffer
        if buf is not None:
            pushes = buf._num_pushes.cpu().numpy()
            hist = buf.buffer.detach().cpu().numpy()[0]
            flat = hist.reshape(-1, hist.shape[-1]) if hist.ndim > 1 else hist[None]
            print(f"  history num_pushes={pushes} shape={hist.shape}")
            for i in range(flat.shape[0]):
                print(f"    hist[{i}] = {np.round(flat[i][:5], 4)}")

    # 4) What the server actually produced in the composed actor vector.
    print("\n=== composed actor vector, joint_pos block, 3 joints ===")
    print("  (see the per-history-frame table above)")


if __name__ == "__main__":
    main()
