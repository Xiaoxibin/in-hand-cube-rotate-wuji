#!/usr/bin/env python
"""TJ Hand 2 (TJDH21R, 21-DoF) hardware interface backed by the Gento Python SDK.

The SDK (PYTHON_SDK/GentoRobot.py + libGentoSDKPY.so) talks Ethernet to the
hand controller; no ROS / Docker required.

Joint correspondence (verified against sim):
    Sim order (TJ_JOINT_ORDER, radians) == SDK Hand2JointIndex order.
        sim CMC1  -> SDK WRIST
        sim CMC2  -> SDK THUMB_CMC
        sim CMC3  -> SDK THUMB_IP ... etc. (identity, index 0..20)
    Only the names differ; no permutation needed.

Units:
    Sim / policy frame : radians.
    SDK frame          : int16, firmware raw = angle_deg * 100.

Calibration:
    JOINT_SIGN and JOINT_ZERO_OFFSET_RAD below must be validated on hardware
    before running the policy (see `python tj2_interface.py --calibrate`).
      raw_int16 = (sign * q_rad + zero_offset_rad) * RAD2HW
      q_rad     = sign * (raw_int16 / RAD2HW - zero_offset_rad)

CLI smoke tests (no policy server needed):
    python tj2_interface.py --sine            # gentle per-finger sine tracking
    python tj2_interface.py --calibrate       # interactive per-joint sign check
    python tj2_interface.py --read            # raw/sim dump (zero-offset helper)

Sign calibration cannot be inferred from readback alone: command and
feedback share JOINT_SIGN, so a wrong sign is self-consistent. The operator
must compare the real finger against the MuJoCo model (that is the y/n prompt
in --calibrate). Zero offset is measured by posing the hand at sim q=0 and
reading raw values with --read.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SDK_PATH = Path(__file__).parent.parent.parent / "TJ_HAND"
if not DEFAULT_SDK_PATH.exists():
    DEFAULT_SDK_PATH = Path.home() / "下载" / "TJ_HAND"

NUM_JOINTS = 21
RAD2HW = 180.0 / math.pi * 100.0  # radians -> SDK int16 (angle_deg * 100)
HW2RAD = 1.0 / RAD2HW

# Sim joint order (== SDK Hand2JointIndex order, identity mapping).
# Index: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
JOINT_NAMES = [
    "CMC1/wrist", "CMC2/th_cmc", "CMC3/th_mcp", "MCPT/th_ip", "DIPT/th_tip",
    "MCPI2", "MCPI1", "PIPI", "DIPI",
    "MCPM2", "MCPM1", "PIPM", "DIPM",
    "MCPR2", "MCPR1", "PIPR", "DIPR",
    "MCPS2", "MCPS1", "PIPS", "DIPS",
]

# ---------------------------------------------------------------------------
# Calibration tables -- FILL IN after hardware validation.
# ---------------------------------------------------------------------------

# Per-joint sign: +1 if a positive sim command moves the same direction as a
# positive SDK value. Validate with --calibrate.
JOINT_SIGN = np.ones(NUM_JOINTS, dtype=np.float64)

# Per-joint zero offset in radians: SDK zero pose minus sim zero pose.
# q_hw = sign * q_sim + zero_offset. All zeros assumes firmware zero == sim zero.
JOINT_ZERO_OFFSET_RAD = np.zeros(NUM_JOINTS, dtype=np.float64)

# Safety clamps in sim radians. Source: ctrlrange of every <position> actuator
# in src/in_hand_rotation_mjlab/robots/tj_hand/xmls/right.xml. The policy
# server already clips to joint limits; this is a second, hardware-side net.
JOINT_LIMITS_RAD = np.array([
    [-1.58,  0.01],   # 0  CMC1   (wrist)
    [-0.01,  0.72],   # 1  CMC2
    [-0.96,  0.27],   # 2  CMC3
    [-1.52,  0.25],   # 3  MCPT
    [-1.68,  0.01],   # 4  DIPT
    [-0.44,  0.44],   # 5  MCPI2
    [-1.58,  0.35],   # 6  MCPI1
    [-1.75,  0.01],   # 7  PIPI
    [-1.49,  0.09],   # 8  DIPI
    [-0.44,  0.44],   # 9  MCPM2
    [-1.58,  0.35],   # 10 MCPM1
    [-1.75,  0.01],   # 11 PIPM
    [-1.49,  0.09],   # 12 DIPM
    [-0.44,  0.27],   # 13 MCPR2
    [-1.58,  0.35],   # 14 MCPR1
    [-1.75,  0.01],   # 15 PIPR
    [-1.49,  0.09],   # 16 DIPR
    [-0.44,  0.27],   # 17 MCPS2
    [-1.58,  0.35],   # 18 MCPS1
    [-1.75,  0.01],   # 19 PIPS
    [-1.49,  0.09],   # 20 DIPS
], dtype=np.float64)


def sim_to_hw(q_sim: np.ndarray) -> list[int]:
    """Sim radians -> SDK int16 list, with sign/offset calibration applied.

    Clip FIRST, in the sim frame: JOINT_LIMITS_RAD is expressed in sim
    radians. Clipping after applying sign/offset compares a hardware-frame
    value against sim-frame limits, which is wrong whenever a sign is -1 or
    an offset is non-zero.
    """
    q_sim = np.clip(
        np.asarray(q_sim, dtype=np.float64),
        JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1],
    )
    q = JOINT_SIGN * q_sim + JOINT_ZERO_OFFSET_RAD
    return [int(round(v * RAD2HW)) for v in q]


def hw_to_sim(q_hw: np.ndarray) -> np.ndarray:
    """SDK int16 readings -> sim radians."""
    q = np.asarray(q_hw, dtype=np.float64) * HW2RAD - JOINT_ZERO_OFFSET_RAD
    return JOINT_SIGN * q


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class TJ2Interface:
    """Thin, safety-first wrapper around the Gento SDK for TJ Hand 2."""

    def __init__(
        self,
        *,
        sdk_path: Path | str | None = None,
        controller_ip: tuple[int, int, int, int] = (6, 6, 7, 190),
        handedness: str = "right",
        ctrl_mode: str = "mit",
        kp: int = 50,
        kd: int = 20,
        log_level: str = "info",
    ):
        sdk_path = Path(sdk_path) if sdk_path else DEFAULT_SDK_PATH
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from PYTHON_SDK.GentoRobot import (  # noqa: PLC0415 (runtime path dep)
            FXLogMask,
            FXObjType,
            GentoRobot,
            TJHandedness,
            TJSendMode2,
        )

        self._TJSendMode2 = TJSendMode2
        self._handedness = (
            TJHandedness.TJ_HANDEDNESS_RIGHT if handedness == "right"
            else TJHandedness.TJ_HANDEDNESS_LEFT
        )
        self._obj = (
            FXObjType.OBJ_ARM1 if handedness == "right" else FXObjType.OBJ_ARM0
        )
        self._log_mask = getattr(FXLogMask, f"FX_LOG_{log_level.upper()}_FLAG")

        if ctrl_mode not in ("mit", "pos"):
            raise ValueError(f"Unknown ctrl_mode '{ctrl_mode}' (use 'mit' or 'pos').")
        self.ctrl_mode = ctrl_mode
        self.kp = int(kp)
        self.kd = int(kd)

        self.robot = GentoRobot()
        self._controller_ip = controller_ip
        self._connected = False
        self._enabled = False

    # ----- lifecycle -------------------------------------------------------

    def connect(self) -> None:
        ip = self._controller_ip
        ret = self.robot.link(*ip, log_level=self._log_mask)
        if not self.robot._connected:
            raise RuntimeError(f"Cannot link controller {ip}: ret={ret}")
        state = self.robot.check_link_state()
        if state == -1:
            self.robot.unlink()
            raise RuntimeError("Link up but no data received (cable/firewall?).")
        self._connected = True

        ret, hand_type = self.robot.hand_get_type(self._obj)
        print(f"[tj2] controller {ip} linked. hand type: {hand_type}")

        mode = (
            self._TJSendMode2.TJ_SEND_MODE2_MIT if self.ctrl_mode == "mit"
            else self._TJSendMode2.TJ_SEND_MODE2_POS
        )
        ret = self.robot.tj2_send_ctrl_mode(self._handedness, [mode] * NUM_JOINTS)
        if ret != 0:
            self.robot.unlink()
            raise RuntimeError(f"tj2_send_ctrl_mode failed: ret={ret}")
        self.robot.tj2_clear_err_code(self._handedness)

        ret = self.robot.tj2_motor_enable(self._handedness)
        if ret != 0:
            self.robot.unlink()
            raise RuntimeError(f"tj2_motor_enable failed: ret={ret}")
        time.sleep(0.5)
        self._enabled = True
        print(f"[tj2] enabled in {self.ctrl_mode.upper()} mode "
              f"(kp={self.kp}, kd={self.kd})")

    def disable_and_close(self) -> None:
        if self._enabled:
            self.robot.tj2_motor_disable(self._handedness)
            self._enabled = False
        if self._connected:
            self.robot.unlink()
            self._connected = False
        print("[tj2] motors disabled, link closed.")

    def __enter__(self) -> "TJ2Interface":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disable_and_close()

    # ----- state -----------------------------------------------------------

    def get_joint_positions_numpy(self) -> np.ndarray:
        """Measured joint positions, sim frame, radians, shape (21,)."""
        ret, pos_now = self.robot.tj2_get_pos_now(self._handedness)
        if ret != 0:
            raise RuntimeError(f"tj2_get_pos_now failed: ret={ret}")
        return hw_to_sim(np.asarray(pos_now, dtype=np.float64))

    def get_status(self) -> dict:
        """Raw diagnostics: error codes and per-joint online mask."""
        _, errs = self.robot.tj2_getErrCode(self._handedness)
        ret, online = self.robot.tj2_get_online(self._handedness)
        return {"error_codes": list(errs), "online": online if ret == 0 else None}

    # ----- command ---------------------------------------------------------

    def send_command(self, q_sim: np.ndarray) -> None:
        """Send one 21-joint position command (sim radians).

        MIT mode: tau = kp*(pos_t - pos) + kd*(v_t - v) + tff, with v_t=0,
        tff=0 -- matches the MuJoCo <position> actuators used in training.
        POS mode: firmware-fixed-gain position servo (used for basic tests).
        """
        q = np.clip(
            np.asarray(q_sim, dtype=np.float64),
            JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1],
        )
        if self.ctrl_mode == "mit":
            ret = self.robot.tj2_send_mit_data(
                self._handedness,
                pos_target=sim_to_hw(q),
                v_target=[0] * NUM_JOINTS,
                tff_target=[0] * NUM_JOINTS,
                kp=[self.kp] * NUM_JOINTS,
                kd=[self.kd] * NUM_JOINTS,
            )
        else:
            ret = self.robot.tj2_send_pos_data(self._handedness, sim_to_hw(q))
        if ret != 0:
            raise RuntimeError(f"send failed: ret={ret}")

    def move_to(
        self,
        q_target_sim: np.ndarray,
        duration_s: float,
        hz: float = 20.0,
    ) -> None:
        """Linearly interpolate from the current measured pose to q_target."""
        q_start = self.get_joint_positions_numpy()
        q_target = np.clip(
            np.asarray(q_target_sim, dtype=np.float64),
            JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1],
        )
        n = max(1, int(duration_s * hz))
        dt = 1.0 / hz
        t_next = time.monotonic()
        for i in range(n):
            alpha = (i + 1) / n
            self.send_command((1.0 - alpha) * q_start + alpha * q_target)
            t_next += dt
            sleep_dt = t_next - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

def _calibrate(args) -> None:
    """Interactive per-joint SIGN calibration.

    Why the operator must judge the direction:
        The command path uses JOINT_SIGN (sim_to_hw) and the feedback path
        uses the same JOINT_SIGN (hw_to_sim). If the sign is wrong, both
        paths are wrong TOGETHER, so "commanded +X, measured +X" remains
        true and self-consistent. The error is only visible by comparing the
        REAL finger's motion against the MuJoCo model (or the known anatomy:
        e.g. positive PIPI in the MJCF flexes the finger). Hence the y/n
        prompt: it is the ground truth, not the readback delta.

    The script still automates the parts a human cannot do reliably:
        - command one joint at a time from its current pose;
        - pick a direction that has room inside the joint limits;
        - detect "no motion" (disabled motor, jam, wrong limit table);
        - flip JOINT_SIGN[j] when the operator says the direction is wrong;
        - print a copy-pasteable JOINT_SIGN array at the end.
    """
    ids = [int(v) for v in args.joints] if args.joints else list(range(NUM_JOINTS))
    signs = JOINT_SIGN.copy()
    step = math.radians(args.step_deg)

    print("=" * 70)
    print("SIGN CALIBRATION")
    print("For each joint the script commands a single-joint step in the SIM")
    print("frame. Watch the REAL hand and compare with the MuJoCo model for")
    print("the same command. Answer 'n' if the real finger moved the other way.")
    print("=" * 70)

    with TJ2Interface(
        sdk_path=args.sdk_path, controller_ip=args.ip_tuple(),
        handedness=args.handedness, ctrl_mode="pos",
    ) as hand:
        for j in ids:
            if not 0 <= j < NUM_JOINTS:
                print(f"[skip] joint {j} out of range 0..{NUM_JOINTS - 1}")
                continue
            input(f"\nJoint {j} ({JOINT_NAMES[j]}) -- keep finger clear, press Enter")
            q0 = hand.get_joint_positions_numpy()
            lo, hi = JOINT_LIMITS_RAD[j]

            # Pick the direction with room inside the (sim-frame) limits.
            direction = 1.0 if q0[j] + step <= hi else -1.0
            if q0[j] + direction * step < lo:
                print(f"  no room to move inside limits [{lo:+.2f}, {hi:+.2f}] "
                      f"(current {q0[j]:+.2f}) -- skipping")
                continue

            q_cmd = q0.copy()
            q_cmd[j] = q0[j] + direction * step
            hand.send_command(q_cmd)
            time.sleep(args.hold_s)
            q1 = hand.get_joint_positions_numpy()
            measured_deg = math.degrees(q1[j] - q0[j])

            print(f"  commanded {direction * args.step_deg:+.0f} deg (sim frame); "
                  f"reported delta {measured_deg:+.1f} deg")

            # Return to the start pose before asking, so the operator can
            # watch a clean back-and-forth and the next joint starts clean.
            hand.send_command(q0)
            time.sleep(args.hold_s)

            if abs(measured_deg) < 3.0:
                print("  -> NO MOTION: motor disabled, joint jammed, or the limit "
                      "table is wrong. Sign is not the issue; fix this first.")
                continue
            if direction * measured_deg < 0.0:
                print("  -> command/feedback DISAGREE: raw sign convention is "
                      "inconsistent (unexpected). Do not trust this joint.")

            answer = input(
                "  Did the REAL finger move in the SAME direction as the "
                "MuJoCo model for this positive sim command? [y/n/skip] "
            ).strip().lower()
            if answer == "n":
                signs[j] = -signs[j]
                print(f"  -> JOINT_SIGN[{j}] flipped to {signs[j]:+.0f}. "
                      "Re-run this joint to confirm.")
            elif answer == "y":
                print(f"  -> JOINT_SIGN[{j}] = {signs[j]:+.0f} confirmed.")
            else:
                print(f"  -> skipped; JOINT_SIGN[{j}] stays {signs[j]:+.0f}.")

    print("\n" + "=" * 70)
    print("Result -- paste into tj2_interface.py:")
    print("JOINT_SIGN = np.array([")
    for row_start in range(0, NUM_JOINTS, 5):
        row = signs[row_start:row_start + 5]
        print("    " + ", ".join(f"{v:+.0f}" for v in row) + ",")
    print("], dtype=np.float64)")
    print("=" * 70)


def _sine(args) -> None:
    """Gentle coupled flex/extend sine on one finger group at a time."""
    groups = {
        "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20], "wrist": [0],
    }
    with TJ2Interface(
        sdk_path=args.sdk_path, controller_ip=args.ip_tuple(),
        handedness=args.handedness, ctrl_mode=args.mode, kp=args.kp, kd=args.kd,
    ) as hand:
        q_mid = (JOINT_LIMITS_RAD[:, 0] + JOINT_LIMITS_RAD[:, 1]) / 2.0
        amp = (JOINT_LIMITS_RAD[:, 1] - JOINT_LIMITS_RAD[:, 0]) / 2.0 * 0.5
        print("[tj2] sine test: ctrl+c to stop")
        try:
            for name, ids in groups.items():
                if name not in args.fingers:
                    continue
                print(f"[tj2] group: {name} {ids}")
                t0, dt = time.monotonic(), 1.0 / args.hz
                while time.monotonic() - t0 < args.seconds:
                    t = time.monotonic() - t0
                    q = q_mid.copy()
                    for j in ids:
                        q[j] = q_mid[j] + amp[j] * math.sin(2 * math.pi * 0.5 * t)
                    hand.send_command(q)
                    time.sleep(dt)
        except KeyboardInterrupt:
            pass
        print("[tj2] returning to mid pose")
        hand.move_to(q_mid, duration_s=2.0, hz=args.hz)


def _read(args) -> None:
    """Dump raw and sim-frame joint readings -- helper for ZERO calibration.

    Zero-offset definition:
        q_hw = JOINT_SIGN * q_sim + JOINT_ZERO_OFFSET_RAD
    so at the mechanical pose that corresponds to sim q = 0:
        JOINT_ZERO_OFFSET_RAD[j] = raw_int16[j] * HW2RAD
    (independent of the sign). Put the hand in that pose by hand with motors
    disabled, then run this command.
    """
    with TJ2Interface(
        sdk_path=args.sdk_path, controller_ip=args.ip_tuple(),
        handedness=args.handedness, ctrl_mode="pos",
    ) as hand:
        ret, raw = hand.robot.tj2_get_pos_now(hand._handedness)
        if ret != 0:
            raise RuntimeError(f"tj2_get_pos_now failed: ret={ret}")
        raw = np.asarray(raw, dtype=np.float64)
        sim = hw_to_sim(raw)
        print(f"{'idx':>3} {'name':<12} {'raw':>8} {'raw_rad':>9} "
              f"{'sim_rad':>9} {'sim_deg':>9}")
        for j in range(NUM_JOINTS):
            print(f"{j:>3} {JOINT_NAMES[j]:<12} {int(raw[j]):>8} "
                  f"{raw[j] * HW2RAD:>9.4f} {sim[j]:>9.4f} "
                  f"{math.degrees(sim[j]):>9.1f}")
        print("\nZero offsets (if the hand is at the sim q=0 pose):")
        print("JOINT_ZERO_OFFSET_RAD = np.array([")
        for start in range(0, NUM_JOINTS, 5):
            row = raw[start:start + 5] * HW2RAD
            print("    " + ", ".join(f"{v:+.4f}" for v in row) + ",")
        print("], dtype=np.float64)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TJ Hand 2 smoke tests.")
    p.add_argument("--sdk-path", type=str, default=None)
    p.add_argument("--controller-ip", type=str, default="6.6.7.190")
    p.add_argument("--handedness", choices=["left", "right"], default="right")
    p.add_argument("--mode", choices=["mit", "pos"], default="mit")
    p.add_argument("--kp", type=int, default=50)
    p.add_argument("--kd", type=int, default=20)
    p.add_argument("--hz", type=float, default=20.0)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--fingers", nargs="+",
                   default=["index"], choices=["thumb", "index", "middle",
                                               "ring", "pinky", "wrist"])
    p.add_argument("--sine", action="store_true", help="Run sine tracking test.")
    p.add_argument("--calibrate", action="store_true",
                   help="Interactive per-joint SIGN calibration (you judge "
                        "direction against the MuJoCo model).")
    p.add_argument("--read", action="store_true",
                   help="Dump raw/sim joint readings (zero-offset helper).")
    p.add_argument("--joints", type=int, nargs="+", default=None,
                   help="Subset of joint indices for --calibrate "
                        "(default: all 21; e.g. --joints 0 1 2 3 4 for thumb).")
    p.add_argument("--step-deg", type=float, default=15.0,
                   help="Calibration step size in degrees (default 15).")
    p.add_argument("--hold-s", type=float, default=1.0,
                   help="Hold time per calibration step (default 1.0 s).")
    args = p.parse_args()

    octets = tuple(int(v) for v in args.controller_ip.split("."))
    if len(octets) != 4:
        p.error(f"--controller-ip must be a.b.c.d, got {args.controller_ip}")
    args.ip_tuple = lambda: octets  # noqa: E731
    return args


def main() -> None:
    args = _parse_args()
    if args.calibrate:
        _calibrate(args)
    elif args.sine:
        _sine(args)
    elif args.read:
        _read(args)
    else:
        print("Nothing to do: pass --sine, --calibrate, or --read.")


if __name__ == "__main__":
    main()
