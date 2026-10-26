#!/usr/bin/env python3
"""sysid_collect.py -- sysid data collector for the TJ Hand 2.

Same excitation recipe and NPZ format as sysid_collect.py (LEAP), but driven
by the Gento SDK instead of ROS. Output arrays:

    time       (T,)   float64  seconds
    hand_cmd   (T,21) float32  commanded positions, sim frame, radians
    hand_state (T,21) float32  measured positions,   sim frame, radians

Feed the saved file to the fitting script:

    uv run python scripts/sysid_fit_from_log.py Mjlab-TJ-Right-HandCube-Rotate \
        --log-file data/tj2_sysid/<finger>_<ts>.npz ...

Usage:
    python tj2_sysid_collect.py --fingers index middle --chirp-s 20
    python tj2_sysid_collect.py --sine-only          # gentle single-pass option
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from tj2_interface import (
    JOINT_LIMITS_RAD,
    NUM_JOINTS,
    TJ2Interface,
)

# ---------------------------------------------------------------------------
# Finger grouping (sim joint indices).
#
# TJ Hand has 21 DoF, no separate wrist joint:
#   thumb 5 DoF: CMC1, CMC2, CMC3, MCPT, DIPT  -> indices 0-4
#   each other finger 4 DoF (MCP2, MCP1, PIP, DIP)
# Index 0 (CMC1) belongs to the THUMB chain -- the vendor sometimes calls it
# "wrist" (see JOINT_NAMES in tj2_interface.py), but it must be excited with
# the thumb: excluding it left the thumb-base actuator uncalibrated.
# Order matches TJ_JOINT_ORDER / SDK Hand2JointIndex (identity mapping).
# ---------------------------------------------------------------------------

FINGER_JOINTS: dict[str, list[int]] = {
    "thumb":  [0, 1, 2, 3, 4],      # 21-DoF TJ: thumb has 5 joints (0=CMC1)
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# Neutral = midrange of each sim joint limit; excitation amplitude = 60% of
# half-range (conservative: unknown real payload on the fingers).
NEUTRAL_Q = ((JOINT_LIMITS_RAD[:, 0] + JOINT_LIMITS_RAD[:, 1]) / 2.0)
AMPLITUDES = (JOINT_LIMITS_RAD[:, 1] - JOINT_LIMITS_RAD[:, 0]) / 2.0 * 0.60


# ---------------------------------------------------------------------------
# Recorder (same format as sysid_collect.py)
# ---------------------------------------------------------------------------

class Recorder:
    def __init__(self):
        self._times: list[float] = []
        self._cmds: list[np.ndarray] = []
        self._states: list[np.ndarray] = []

    def record(self, t: float, cmd: np.ndarray, state: np.ndarray) -> None:
        self._times.append(t)
        self._cmds.append(cmd.copy())
        self._states.append(state.copy())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            time=np.array(self._times, dtype=np.float64),
            hand_cmd=np.stack(self._cmds, axis=0).astype(np.float32),
            hand_state=np.stack(self._states, axis=0).astype(np.float32),
        )
        print(f"[sysid] saved {len(self._times)} samples -> {path}")

    def clear(self) -> None:
        self._times.clear()
        self._cmds.clear()
        self._states.clear()

    def __len__(self) -> int:
        return len(self._times)


# ---------------------------------------------------------------------------
# Loop helpers
# ---------------------------------------------------------------------------

class Loop:
    """Fixed-rate loop with measured-position readback and recording."""

    def __init__(self, hand: TJ2Interface, recorder: Recorder, hz: float):
        self.hand = hand
        self.recorder = recorder
        self.dt = 1.0 / hz

    def step(self, q_cmd: np.ndarray) -> None:
        self.hand.send_command(q_cmd)
        time.sleep(self.dt)  # give the hand a cycle to move before readback
        q_state = self.hand.get_joint_positions_numpy()
        self.recorder.record(time.monotonic(), q_cmd, q_state)

    def step_quiet(self, q_cmd: np.ndarray) -> None:
        """Send command without recording (used for interpolation transitions)."""
        self.hand.send_command(q_cmd)
        time.sleep(self.dt)


def interpolate_to(loop: Loop, recorder: Recorder,
                   q_target: np.ndarray, duration_s: float = 2.0) -> None:
    q_start = loop.hand.get_joint_positions_numpy()
    n = max(1, int(duration_s * (1.0 / loop.dt)))
    for i in range(n):
        alpha = (i + 1) / n
        loop.step_quiet((1 - alpha) * q_start + alpha * q_target)
    recorder.clear()  # transitions are never useful sysid data


def settle(loop: Loop, recorder: Recorder, settle_s: float,
           q_hold: np.ndarray) -> None:
    n = int(settle_s / loop.dt)
    for _ in range(n):
        loop.step(q_hold)


def _make_q(joint_ids: list[int], offsets: np.ndarray) -> np.ndarray:
    """Full command: other joints stay at their neutral pose."""
    q = NEUTRAL_Q.copy()
    for k, jid in enumerate(joint_ids):
        q[jid] = float(np.clip(
            NEUTRAL_Q[jid] + offsets[k],
            JOINT_LIMITS_RAD[jid, 0], JOINT_LIMITS_RAD[jid, 1],
        ))
    return q


# ---------------------------------------------------------------------------
# Excitation patterns (same recipe as sysid_collect.py)
# ---------------------------------------------------------------------------

def step_ladder(loop: Loop, recorder: Recorder, joint_ids: list[int],
                amplitudes: np.ndarray, hold_s: float) -> None:
    print(f"[sysid]   step_ladder: {len(joint_ids)} joints, hold={hold_s:.2f}s")
    hold_steps = max(1, int(hold_s / loop.dt))
    n = len(joint_ids)
    for k in range(n):
        for scale in [0.5, 0.75, 1.0]:
            a = amplitudes[k] * scale
            for target in [a, 0.0, -a, 0.0]:
                offsets = np.zeros(n, dtype=np.float64)
                offsets[k] = target
                q = _make_q(joint_ids, offsets)
                for _ in range(hold_steps):
                    loop.step(q)
    for sign in [1.0, 0.0, -1.0, 0.0]:
        q = _make_q(joint_ids, amplitudes * sign)
        for _ in range(hold_steps):
            loop.step(q)


def chirp(loop: Loop, recorder: Recorder, joint_ids: list[int],
          amplitudes: np.ndarray, duration_s: float,
          f_start: float = 0.1, f_end: float = 3.0) -> None:
    print(f"[sysid]   chirp: {f_start}->{f_end} Hz, {duration_s:.0f}s/pass")
    n_steps = int(duration_s / loop.dt)
    n = len(joint_ids)
    for k in range(n):
        t = 0.0
        for _ in range(n_steps):
            phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / duration_s)
            offsets = np.zeros(n, dtype=np.float64)
            offsets[k] = amplitudes[k] * np.sin(phase)
            loop.step(_make_q(joint_ids, offsets))
            t += loop.dt
    t = 0.0
    for _ in range(n_steps):
        phase = 2 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t**2 / duration_s)
        offsets = amplitudes * float(np.sin(phase))
        loop.step(_make_q(joint_ids, offsets))
        t += loop.dt


def prbs(loop: Loop, recorder: Recorder, joint_ids: list[int],
         amplitudes: np.ndarray, duration_s: float,
         min_dwell: int = 3, max_dwell: int = 8) -> None:
    print(f"[sysid]   prbs: {duration_s:.0f}s")
    rng = np.random.default_rng()
    n_steps = int(duration_s / loop.dt)
    n = len(joint_ids)
    offsets = np.zeros(n, dtype=np.float64)
    dwell = 0
    for _ in range(n_steps):
        if dwell <= 0:
            levels = rng.choice([-1.0, 0.0, 1.0], size=n)
            offsets = amplitudes * levels
            dwell = int(rng.integers(min_dwell, max_dwell + 1))
        loop.step(_make_q(joint_ids, offsets))
        dwell -= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated sysid data collection for TJ Hand 2."
    )
    p.add_argument("--fingers", nargs="+",
                   # TJ 21-DoF: include pinky too (5 fingers x 4 + thumb 5).
                   default=["thumb", "index", "middle", "ring", "pinky"],
                   choices=list(FINGER_JOINTS.keys()),
                   help="Fingers to excite in order (default: all five).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output directory (default: <repo>/data/tj2-sysid/).")
    p.add_argument("--controller-ip", type=str, default="6.6.7.190")
    p.add_argument("--handedness", choices=["left", "right"], default="right")
    p.add_argument("--kp", type=int, default=50)
    p.add_argument("--kd", type=int, default=20)
    p.add_argument("--sdk-path", type=str, default=None)
    p.add_argument("--hz", type=float, default=20.0,
                   help="Control rate for excitation (Hz).")
    p.add_argument("--settle-s", type=float, default=3.0)
    p.add_argument("--step-hold-s", type=float, default=0.8)
    p.add_argument("--chirp-s", type=float, default=30.0)
    p.add_argument("--prbs-s", type=float, default=60.0)
    p.add_argument("--init-duration-s", type=float, default=3.0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else Path(__file__).parent.parent / "data" / "tj2-sysid"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sysid] output directory: {out_dir}")

    hand = TJ2Interface(
        sdk_path=args.sdk_path,
        controller_ip=tuple(int(v) for v in args.controller_ip.split(".")),
        handedness=args.handedness,
        ctrl_mode="mit",
        kp=args.kp,
        kd=args.kd,
    )
    print(f"[sysid] connecting to hand controller {args.controller_ip} ...")
    hand.connect()

    try:
        recorder = Recorder()
        loop = Loop(hand, recorder, args.hz)

        # Move to global neutral and settle; discard warm-up.
        interpolate_to(loop, recorder, NEUTRAL_Q, args.init_duration_s)
        settle(loop, recorder, args.settle_s, NEUTRAL_Q)
        recorder.clear()

        for finger in args.fingers:
            joint_ids = FINGER_JOINTS[finger]
            amplitudes = AMPLITUDES[joint_ids]
            finger_neutral = _make_q(joint_ids, np.zeros(len(joint_ids)))

            print("=" * 50)
            print(f"[sysid] FINGER: {finger.upper()} (joints {joint_ids})")
            print(f"[sysid]   neutral (deg): {np.degrees(finger_neutral[joint_ids]).round(1)}")
            print(f"[sysid]   amplitude (deg): {np.degrees(amplitudes).round(1)}")
            print("=" * 50)

            recorder.clear()
            interpolate_to(loop, recorder, finger_neutral, 2.0)
            settle(loop, recorder, args.settle_s, finger_neutral)
            recorder.clear()

            step_ladder(loop, recorder, joint_ids, amplitudes, args.step_hold_s)
            settle(loop, recorder, args.settle_s, finger_neutral)
            chirp(loop, recorder, joint_ids, amplitudes, args.chirp_s)
            settle(loop, recorder, args.settle_s, finger_neutral)
            prbs(loop, recorder, joint_ids, amplitudes, args.prbs_s)

            interpolate_to(loop, recorder, NEUTRAL_Q, 2.0)
            settle(loop, recorder, args.settle_s, NEUTRAL_Q)

            ts = datetime.now().strftime("%H%M%S")
            recorder.save(out_dir / f"{finger}_{ts}.npz")

        print(f"[sysid] complete. Files in: {out_dir}")
        print("[sysid] next: fit with scripts/sysid_fit_from_log.py --finger <name>")
    finally:
        hand.disable_and_close()


if __name__ == "__main__":
    main()
