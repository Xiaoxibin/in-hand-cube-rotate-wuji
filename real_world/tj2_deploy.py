#!/usr/bin/env python
"""Deploy a ZMQ policy server policy on TJ Hand 2 hardware.

No ROS / Docker: talks to the hand via the Gento SDK (Ethernet) and to the
policy server via ZMQ REQ/REP.

Topology:
    [GPU machine]  scripts/policy_server_zmq.py  listens  tcp://0.0.0.0:5555
         ^                                   ^
         | ZMQ (obs -> action, 20 Hz)        |
    [hardware machine]  this script -- Gento SDK -- controller 6.6.7.190 -- hand

Observation contract (matches deploy.py / policy_server):
    Client sends RAW sim-frame joint positions (radians). The server subtracts
    default_joint_pos internally for the `joint_pos` term.

Usage (from real_world/):
    python tj2_deploy.py --server-endpoint tcp://<GPU_MACHINE_IP>:5555 \
        --policy-hz 20 --init-duration-s 5.0

Emergency stop: Ctrl+C disables the hand motors.
"""

from __future__ import annotations

import argparse
import base64
import time
from typing import Any

import numpy as np
import zmq

from tj2_interface import (
    NUM_JOINTS,
    TJ2Interface,
)


# ---------------------------------------------------------------------------
# Wire helpers (same encoding as real_world/deploy.py)
# ---------------------------------------------------------------------------

def _encode_array(array: np.ndarray) -> dict[str, Any]:
    x = np.ascontiguousarray(array, dtype=np.float32)
    return {
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "data_b64": base64.b64encode(x.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(str(payload["dtype"]))
    shape = tuple(int(v) for v in payload["shape"])
    raw = base64.b64decode(str(payload["data_b64"]))
    return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)


def _encode_obs(
    *,
    q: np.ndarray,
    prev_cmd: np.ndarray,
    action_term_name: str,
) -> dict[str, Any]:
    """Observation packet in the server's sim frame (radians, raw values)."""
    row = lambda x: np.asarray(x[None, :], dtype=np.float32)  # noqa: E731
    return {
        "term_values": {
            "joint_pos":                _encode_array(row(q)),
            "prev_commanded_joint_pos": _encode_array(row(prev_cmd)),
        },
        "action_term_joint_pos": {
            action_term_name: _encode_array(row(q)),
        },
    }


# ---------------------------------------------------------------------------
# ZMQ RPC client (same protocol as real_world/deploy.py)
# ---------------------------------------------------------------------------

class _ZmqRpcClient:
    def __init__(self, *, endpoint: str, timeout_ms: int):
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, int(timeout_ms))
        self._socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
        self._socket.connect(endpoint)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            self._socket.send_json(payload)
            response = self._socket.recv_json()
        except zmq.Again as exc:
            raise TimeoutError(
                f"Timed out waiting for policy server at '{self.endpoint}'."
            ) from exc
        if not bool(response.get("ok", False)):
            raise RuntimeError(str(response.get("error", "Unknown server error.")))
        return response

    def ping(self) -> None:
        self._request({"type": "ping"})

    def init(self, *, step_dt: float, max_episode_steps: int) -> dict[str, Any]:
        return self._request({
            "type": "init",
            "step_dt": float(step_dt),
            "max_episode_steps": int(max_episode_steps),
        })

    def reset(self, obs: dict[str, Any]) -> None:
        self._request({"type": "reset", "observation": obs})

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        response = self._request({"type": "infer", "observation": obs})
        return {
            k: _decode_array(v)
            for k, v in response["action"]["joint_position_commands"].items()
        }

    def close(self, *, shutdown_server: bool = False) -> None:
        if shutdown_server:
            try:
                self._request({"type": "close"})
            except Exception:
                pass
        self._socket.close(linger=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy a ZMQ policy server on TJ Hand 2 hardware."
    )
    p.add_argument("--server-endpoint",   type=str, required=True,
                   help="tcp://<GPU-machine-ip>:5555")
    p.add_argument("--server-timeout-ms", type=int, default=30_000)
    p.add_argument("--policy-hz",         type=float, default=20.0)
    p.add_argument("--max-steps",         type=int, default=0,
                   help="0 = run until Ctrl+C.")
    p.add_argument("--max-episode-steps", type=int, default=10_000_000)
    p.add_argument("--action-term-name",  type=str, default="joint_pos")
    p.add_argument("--init-duration-s",   type=float, default=5.0,
                   help="Seconds to interpolate from current pose to server default pose.")
    p.add_argument("--shutdown-server-on-exit", action="store_true")
    # Hand-side options (forwarded to TJ2Interface).
    p.add_argument("--controller-ip",     type=str, default="6.6.7.190")
    p.add_argument("--handedness",        choices=["left", "right"], default="right")
    p.add_argument("--mode",              choices=["mit", "pos"], default="mit")
    p.add_argument("--kp",                type=int, default=50)
    p.add_argument("--kd",                type=int, default=20)
    p.add_argument("--sdk-path",          type=str, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    hand = TJ2Interface(
        sdk_path=args.sdk_path,
        controller_ip=tuple(int(v) for v in args.controller_ip.split(".")),
        handedness=args.handedness,
        ctrl_mode=args.mode,
        kp=args.kp,
        kd=args.kd,
    )
    print(f"[deploy] connecting to hand controller {args.controller_ip} ...")
    hand.connect()

    try:
        q0 = hand.get_joint_positions_numpy()
        if q0.shape[0] != NUM_JOINTS:
            raise RuntimeError(f"Expected {NUM_JOINTS} joints, got {q0.shape[0]}.")

        # --- Connect to policy server and fetch default pose -----------------
        rpc = _ZmqRpcClient(endpoint=args.server_endpoint,
                            timeout_ms=args.server_timeout_ms)
        policy_dt = 1.0 / float(args.policy_hz)

        print(f"[deploy] connecting to policy server: {args.server_endpoint}")
        rpc.ping()
        init_ack = rpc.init(step_dt=policy_dt, max_episode_steps=args.max_episode_steps)

        term_md = init_ack["action_term_metadata"][args.action_term_name]
        # default_joint_pos: [1, 21] in server (sim) frame, radians.
        default_q = _decode_array(term_md["default_joint_pos"]).reshape(-1).astype(np.float64)
        print(f"[deploy] server default pose (deg): "
              f"{np.degrees(default_q).round(1).tolist()}")

        # --- Move hand to server default pose --------------------------------
        print(f"[deploy] moving to default pose over {args.init_duration_s:.1f}s ...")
        hand.move_to(default_q, duration_s=args.init_duration_s, hz=args.policy_hz)
        print("[deploy] default pose reached.")

        q_start = hand.get_joint_positions_numpy()
        prev_cmd = q_start.copy()
        rpc.reset(_encode_obs(q=q_start, prev_cmd=prev_cmd,
                              action_term_name=args.action_term_name))

        # --- Policy loop ------------------------------------------------------
        step = 0
        total_ms = 0.0
        infer_ms = 0.0
        LOG_EVERY = 50
        dt = 1.0 / args.policy_hz
        print(f"[deploy] policy loop started at {args.policy_hz:.2f} Hz "
              f"(Ctrl+C to stop)")

        t_next = time.monotonic()
        try:
            while True:
                t0 = time.perf_counter()

                q = hand.get_joint_positions_numpy()
                obs = _encode_obs(q=q, prev_cmd=prev_cmd,
                                  action_term_name=args.action_term_name)

                t1 = time.perf_counter()
                cmd_map = rpc.infer(obs)
                t2 = time.perf_counter()

                cmd = cmd_map[args.action_term_name].reshape(-1).astype(np.float64)
                hand.send_command(cmd)

                t3 = time.perf_counter()
                total_ms += (t3 - t0) * 1000.0
                infer_ms += (t2 - t1) * 1000.0
                prev_cmd = cmd
                step += 1

                if step % LOG_EVERY == 0:
                    avg = total_ms / step
                    print(f"[deploy] step={step}  iter_ms={(t3 - t0) * 1000:.2f}  "
                          f"infer_ms={(t2 - t1) * 1000:.2f}  "
                          f"avg_iter_ms={avg:.2f}  avg_hz={1000.0 / avg:.1f}")

                if args.max_steps > 0 and step >= args.max_steps:
                    print(f"[deploy] reached max_steps={args.max_steps}.")
                    break

                t_next += dt
                sleep_dt = t_next - time.monotonic()
                if sleep_dt > 0:
                    time.sleep(sleep_dt)
                else:
                    t_next = time.monotonic()  # fell behind; resync
        except KeyboardInterrupt:
            print("\n[deploy] interrupted by user.")
        finally:
            rpc.close(shutdown_server=args.shutdown_server_on_exit)
    finally:
        hand.disable_and_close()


if __name__ == "__main__":
    main()
