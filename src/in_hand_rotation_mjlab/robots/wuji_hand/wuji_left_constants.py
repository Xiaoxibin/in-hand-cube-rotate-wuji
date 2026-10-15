"""
Wuji Hand 1 Left constants and EntityCfg for mjlab.

This file provides the robot binding for the 20-DoF Wuji Hand 1 Left.

The MJCF file already defines 20 position actuators. Their joint limits,
control ranges, proportional gains, damping gains, and force limits are
kept in the MJCF model.

This module mainly provides:

1. MJCF loading.
2. Mesh asset embedding.
3. Collision configuration.
4. Initial joint configuration.
5. XML position actuator registration.
6. mjlab EntityCfg construction.

The actuator control path is:

    Policy action
        ↓
    JointPositionDeltaAction
        ↓
    joint position target
        ↓
    XML-defined <position> actuator
        ↓
    MuJoCo

The XML actuator parameters are NOT removed or recreated here.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from in_hand_rotation_mjlab import MYMJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg


# =============================================================================
# MJCF + assets
# =============================================================================

WUJI_LEFT_HAND_XML: Path = (
    MYMJLAB_SRC_PATH
    / "robots"
    / "wuji_hand"
    / "xmls"
    / "left_hand.xml"
)

assert WUJI_LEFT_HAND_XML.exists(), (
    f"Missing Wuji left-hand MJCF: {WUJI_LEFT_HAND_XML}"
)


def get_assets(meshdir: str) -> dict[str, bytes]:
    """
    Embed mesh assets into MjSpec.assets.

    The Wuji MJCF uses mesh assets relative to its asset directory.
    update_assets() packages these files into the MjSpec so that the
    robot can be loaded correctly by mjlab and MuJoCo-based viewers.
    """
    assets: dict[str, bytes] = {}

    update_assets(assets,WUJI_LEFT_HAND_XML.parent / "assets",meshdir,)
    return assets


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> None:
    """
    Assign stable names to unnamed visual/collision geoms.

    mjlab collision configuration uses regular-expression matching on
    geom names. Therefore unnamed geoms are given deterministic names.
    """
    counter: dict[str, int] = {}

    for geom in spec.geoms:
        if geom.name:
            continue

        body_name = geom.parent.name if geom.parent else "world"

        is_visual = geom.contype == 0 and geom.conaffinity == 0
        suffix = "visual" if is_visual else "collision"

        key = f"{body_name}_{suffix}"

        idx = counter.get(key, 0)
        counter[key] = idx + 1

        geom.name = f"{key}_{idx}" if idx > 0 else key


def get_spec() -> mujoco.MjSpec:
    """
    Create MjSpec from the Wuji left-hand MJCF and attach mesh assets.

    IMPORTANT:
    The XML-defined actuators are intentionally kept.

    The XML contains 20 <position> actuators with their own:
        - kp
        - kv
        - ctrlrange
        - forcerange

    XmlPositionActuatorCfg below tells mjlab to use those existing
    XML actuators.
    """
    spec = mujoco.MjSpec.from_file(str(WUJI_LEFT_HAND_XML))

    _name_unnamed_geoms(spec)

    spec.assets = get_assets(spec.meshdir)

    return spec


# =============================================================================
# Collision configuration
# =============================================================================

# Current Wuji naming convention:
#
#   left_finger1_joint1 ... joint4
#   left_finger2_joint1 ... joint4
#   left_finger3_joint1 ... joint4
#   left_finger4_joint1 ... joint4
#   left_finger5_joint1 ... joint4
#
# The distal/contact link is assumed to be link4.
#
# The exact geom names depend on left_hand.xml. The broad expression below
# is intentionally used first and can be refined after inspecting the
# compiled geom names.

WUJI_KEYPOINT_GEOMS = (r"left_finger[1-5]_(?:tip|link4_collision)")
# WUJI_KEYPOINT_GEOMS = r"left_finger.*_link4_collision"

WUJI_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision.*",r"left_finger[1-5]_tip",),

    # General collision:
    #   condim=3 -> normal + tangential contact forces.
    #
    # Fingertip/contact geoms:
    #   condim=6 -> includes torsional/rolling contact dimensions.
    condim={
        WUJI_KEYPOINT_GEOMS: 6,
        ".*_collision.*": 3,
    },

    # Fingertip friction.
    #
    # The distal fingertip value is intentionally higher than the
    # general collision value because fingertip-object interaction is
    # important for dexterous manipulation.
    friction={
        WUJI_KEYPOINT_GEOMS: (1.0, 5e-3, 5e-4),
        ".*_collision.*": (0.6,),
    },

    # Fingertip contact solver parameters.
    solref={
        WUJI_KEYPOINT_GEOMS: (0.01, 1.0),
    },

    # Higher priority for fingertip contacts.
    priority={
        WUJI_KEYPOINT_GEOMS: 1,
    },
)


# =============================================================================
# Initial state
# =============================================================================

# Fixed-base hand.
#
# The final base position may later be controlled by the task/scene
# configuration. This is only the default robot initial position.

WUJI_HOME_POS: tuple[float, float, float] = (0.06, 0.00, 0.30)
WUJI_HOME_ROT: tuple[float, float, float, float] = (0.7071068, 0.0, -0.7071068, 0.0)
WUJI_HOME_JOINT_POS: dict[str, float] = {
    # Thumb - finger 1
    "left_finger1_joint1": 1.10,
    "left_finger1_joint2": 0.00,
    "left_finger1_joint3": 0.95,
    "left_finger1_joint4": 0.80,

    # Index - finger 2
    "left_finger2_joint1": 0.60,
    "left_finger2_joint2": 0.00,
    "left_finger2_joint3": 1.00,
    "left_finger2_joint4": 0.80,

    # Middle - finger 3
    "left_finger3_joint1": 0.80,
    "left_finger3_joint2": 0.00,
    "left_finger3_joint3": 1.00,
    "left_finger3_joint4": 0.80,

    # Ring - finger 4
    "left_finger4_joint1": 0.65,
    "left_finger4_joint2": 0.00,
    "left_finger4_joint3": 1.00,
    "left_finger4_joint4": 0.80,

    # Pinky - finger 5
    "left_finger5_joint1": 0.60,
    "left_finger5_joint2": 0.00,
    "left_finger5_joint3": 0.95,
    "left_finger5_joint4": 0.75,
}


def make_home_state(
    pos: tuple[float, float, float] = WUJI_HOME_POS,
    rot: tuple[float, float, float, float] = WUJI_HOME_ROT,
) -> EntityCfg.InitialStateCfg:
    """
    Return a fresh initial-state configuration.

    A new joint-position dictionary is created for each configuration
    to avoid accidental mutation between environments.
    """
    return EntityCfg.InitialStateCfg(
        pos=pos,
        rot=rot,
        joint_pos=dict(WUJI_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    )


# =============================================================================
# XML actuator configuration
# =============================================================================

# The Wuji MJCF already defines 20 <position> actuators:
#
#   left_finger1_joint1_actuator
#   ...
#   left_finger5_joint4_actuator
#
# Each actuator contains its own XML parameters:
#
#   kp
#   kv
#   ctrlrange
#   forcerange
#
# Example:
#
#   <position
#       name="left_finger1_joint1_actuator"
#       joint="left_finger1_joint1"
#       kp="..."
#       kv="..."
#       ctrlrange="..."
#       forcerange="..."
#   />
#
# We therefore DO NOT:
#
#   1. delete the XML actuators;
#   2. recreate PD gains in Python;
#   3. use IdealPdActuatorCfg.
#
# XmlPositionActuatorCfg simply tells mjlab to use the existing XML
# position actuators.

WUJI_XML_ACTUATORS = XmlPositionActuatorCfg(
    target_names_expr=(
        r"left_finger[1-5]_joint[1-4]",
    ),
)


ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        WUJI_XML_ACTUATORS,
    ),

    # Keep a safety margin between the commanded position and the hard
    # MJCF joint limits.
    soft_joint_pos_limit_factor=0.9,
)


# =============================================================================
# Final EntityCfg
# =============================================================================

def get_wuji_left_hand_cfg() -> EntityCfg:
    """
    Return mjlab EntityCfg for Wuji Hand 1 Left.

    This is the main entry point used by tasks.

    The returned EntityCfg contains:
        - initial state
        - collision configuration
        - MJCF specification
        - XML-defined position actuators
        - articulation configuration
    """
    return EntityCfg(
        init_state=make_home_state(),
        collisions=(
            WUJI_COLLISION,
        ),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


# =============================================================================
# Debug entrypoint
# =============================================================================

def main():
    import mujoco

    print("=" * 80)
    print("WUJI FINAL DIAGNOSTIC")
    print("Home Pose + Collision ON + Actuator ON")
    print("=" * 80)

    # ==================================================================
    # 1. Load Wuji configuration
    # ==================================================================

    wuji = get_wuji_left_hand_cfg()

    print("\n" + "=" * 80)
    print("WUJI CONFIG")
    print("=" * 80)

    print("Init joint positions:")

    for joint_name, joint_pos in WUJI_HOME_JOINT_POS.items():
        print(
            f"  {joint_name:36s}: "
            f"{joint_pos:+.6f}"
        )

    # ==================================================================
    # 2. Compile MuJoCo model
    # ==================================================================

    spec = wuji.spec_fn()
    model = spec.compile()
    data = mujoco.MjData(model)

    print("\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)

    print(f"nq    = {model.nq}")
    print(f"nv    = {model.nv}")
    print(f"njnt  = {model.njnt}")
    print(f"nbody = {model.nbody}")
    print(f"ngeom = {model.ngeom}")
    print(f"nu    = {model.nu}")

    # ==================================================================
    # 3. Collision status
    # ==================================================================

    print("\n" + "=" * 80)
    print("COLLISION STATUS")
    print("=" * 80)

    collision_enabled_count = 0

    for geom_id in range(model.ngeom):
        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])

        if contype != 0 and conaffinity != 0:
            collision_enabled_count += 1

    print(
        f"Collision-enabled geoms: "
        f"{collision_enabled_count} / {model.ngeom}"
    )

    # ==================================================================
    # 4. Print collision geometry information
    # ==================================================================

    print("\n" + "=" * 80)
    print("COLLISION GEOMETRY")
    print("=" * 80)

    for geom_id in range(model.ngeom):

        geom_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )

        body_id = int(model.geom_bodyid[geom_id])

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
        )

        contype = int(model.geom_contype[geom_id])
        conaffinity = int(model.geom_conaffinity[geom_id])

        print(
            f"[{geom_id:02d}] "
            f"geom={str(geom_name):42s} "
            f"body={str(body_name):30s} "
            f"contype={contype} "
            f"conaffinity={conaffinity}"
        )

    # ==================================================================
    # 5. Build joint-name -> qpos index mapping
    # ==================================================================

    joint_qpos_indices = {}

    for joint_id in range(model.njnt):

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        qpos_index = int(model.jnt_qposadr[joint_id])

        joint_qpos_indices[joint_name] = qpos_index

    # ==================================================================
    # 6. Set Wuji Home Pose
    # ==================================================================

    print("\n" + "=" * 80)
    print("SET WUJI HOME POSE")
    print("=" * 80)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    for joint_name, joint_pos in WUJI_HOME_JOINT_POS.items():

        if joint_name not in joint_qpos_indices:
            print(
                f"WARNING: Joint '{joint_name}' "
                f"not found in compiled model."
            )
            continue

        qpos_index = joint_qpos_indices[joint_name]

        data.qpos[qpos_index] = joint_pos

        print(
            f"  {joint_name:36s} "
            f"qpos={joint_pos:+.6f}"
        )

    # ==================================================================
    # 7. Set position actuator targets
    #
    # XML <position> actuators use data.ctrl as desired joint position.
    # ==================================================================

    print("\n" + "=" * 80)
    print("SET POSITION ACTUATOR TARGETS")
    print("=" * 80)

    for actuator_id in range(model.nu):

        actuator_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        joint_id = int(
            model.actuator_trnid[actuator_id, 0]
        )

        if joint_id < 0:
            print(
                f"[{actuator_id:02d}] "
                f"{str(actuator_name):42s} "
                f"joint=N/A"
            )
            continue

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        target = WUJI_HOME_JOINT_POS.get(
            joint_name,
            0.0,
        )

        data.ctrl[actuator_id] = target

        print(
            f"[{actuator_id:02d}] "
            f"{str(actuator_name):42s} "
            f"joint={str(joint_name):36s} "
            f"ctrl={target:+.6f}"
        )

    # ==================================================================
    # 8. Forward dynamics
    # ==================================================================

    mujoco.mj_forward(model, data)

    # ==================================================================
    # 9. Initial contact check
    # ==================================================================

    print("\n" + "=" * 80)
    print("INITIAL CONTACTS AT HOME POSE")
    print("=" * 80)

    print(f"Number of contacts: {data.ncon}")

    if data.ncon == 0:

        print("No contacts at Home Pose.")

    else:

        for contact_id in range(data.ncon):

            contact = data.contact[contact_id]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            geom1_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom1,
            )

            geom2_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom2,
            )

            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])

            body1_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body1,
            )

            body2_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body2,
            )

            print(f"\nCONTACT {contact_id}")

            print(
                f"  geom1 : [{geom1}] "
                f"{geom1_name}"
            )

            print(
                f"  body1 : [{body1}] "
                f"{body1_name}"
            )

            print(
                f"  geom2 : [{geom2}] "
                f"{geom2_name}"
            )

            print(
                f"  body2 : [{body2}] "
                f"{body2_name}"
            )

            print(
                f"  dist  : "
                f"{contact.dist:+.8f}"
            )

    # ==================================================================
    # 10. Initial joint state
    # ==================================================================

    print("\n" + "=" * 80)
    print("INITIAL JOINT STATE")
    print("=" * 80)

    print(
        f"{'ID':>3s} "
        f"{'Joint':36s} "
        f"{'qpos':>12s} "
        f"{'qvel':>12s}"
    )

    print("-" * 70)

    for joint_id in range(model.njnt):

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        qpos_index = int(
            model.jnt_qposadr[joint_id]
        )

        qvel_index = int(
            model.jnt_dofadr[joint_id]
        )

        print(
            f"{joint_id:3d} "
            f"{str(joint_name):36s} "
            f"{data.qpos[qpos_index]:+12.6f} "
            f"{data.qvel[qvel_index]:+12.6f}"
        )

    # ==================================================================
    # 11. Dynamic stability test
    # ==================================================================

    print("\n" + "=" * 80)
    print("DYNAMIC STABILITY TEST")
    print("=" * 80)

    initial_qpos = data.qpos.copy()

    max_qpos_drift = 0.0
    max_qvel = 0.0
    max_contacts = 0

    # Store every collision pair that appears.
    collision_pair_counts = {}

    # Also record the first step at which each pair appears.
    collision_pair_first_step = {}

    num_steps = 500

    for step in range(num_steps):

        mujoco.mj_step(model, data)

        # --------------------------------------------------------------
        # Joint state statistics
        # --------------------------------------------------------------

        qpos_drift = float(
            abs(data.qpos - initial_qpos).max()
        )

        qvel_max = float(
            abs(data.qvel).max()
        )

        max_qpos_drift = max(
            max_qpos_drift,
            qpos_drift,
        )

        max_qvel = max(
            max_qvel,
            qvel_max,
        )

        # --------------------------------------------------------------
        # Contact statistics
        # --------------------------------------------------------------

        current_contacts = int(data.ncon)

        max_contacts = max(
            max_contacts,
            current_contacts,
        )

        for contact_id in range(current_contacts):

            contact = data.contact[contact_id]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            # Make pair order-independent.
            pair = tuple(
                sorted((geom1, geom2))
            )

            collision_pair_counts[pair] = (
                collision_pair_counts.get(pair, 0)
                + 1
            )

            if pair not in collision_pair_first_step:
                collision_pair_first_step[pair] = step

    print(f"Simulation steps = {num_steps}")
    print(
        f"Max qpos drift   = "
        f"{max_qpos_drift:.8e}"
    )
    print(
        f"Max |qvel|       = "
        f"{max_qvel:.8e}"
    )
    print(
        f"Max contacts     = "
        f"{max_contacts}"
    )

    # ==================================================================
    # 12. Collision pair statistics
    # ==================================================================

    print("\n" + "=" * 80)
    print("COLLISION PAIR STATISTICS")
    print("=" * 80)

    if not collision_pair_counts:

        print(
            "No collision pairs detected "
            "during the entire simulation."
        )

    else:

        # Sort by occurrence count.
        sorted_pairs = sorted(
            collision_pair_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        print(
            f"Unique collision pairs: "
            f"{len(sorted_pairs)}"
        )

        for rank, (pair, count) in enumerate(
            sorted_pairs,
            start=1,
        ):

            geom1, geom2 = pair

            geom1_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom1,
            )

            geom2_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom2,
            )

            body1 = int(
                model.geom_bodyid[geom1]
            )

            body2 = int(
                model.geom_bodyid[geom2]
            )

            body1_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body1,
            )

            body2_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body2,
            )

            first_step = collision_pair_first_step[
                pair
            ]

            print(
                f"\nPair #{rank}"
            )

            print(
                f"  Occurrences : {count}"
            )

            print(
                f"  First step  : {first_step}"
            )

            print(
                f"  geom1       : "
                f"[{geom1}] {geom1_name}"
            )

            print(
                f"  body1       : "
                f"[{body1}] {body1_name}"
            )

            print(
                f"  geom2       : "
                f"[{geom2}] {geom2_name}"
            )

            print(
                f"  body2       : "
                f"[{body2}] {body2_name}"
            )

    # ==================================================================
    # 13. Final contact state
    # ==================================================================

    print("\n" + "=" * 80)
    print("FINAL CONTACTS")
    print("=" * 80)

    print(f"Number of contacts: {data.ncon}")

    if data.ncon == 0:

        print("No contacts at final simulation step.")

    else:

        for contact_id in range(data.ncon):

            contact = data.contact[contact_id]

            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            geom1_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom1,
            )

            geom2_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom2,
            )

            body1 = int(model.geom_bodyid[geom1])
            body2 = int(model.geom_bodyid[geom2])

            body1_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body1,
            )

            body2_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body2,
            )

            print(
                f"[{contact_id:02d}] "
                f"{geom1_name} ({body1_name})"
                f"  <->  "
                f"{geom2_name} ({body2_name})"
                f"  dist={contact.dist:+.8f}"
            )

    # ==================================================================
    # 14. Final joint state
    # ==================================================================

    print("\n" + "=" * 80)
    print("FINAL JOINT STATE")
    print("=" * 80)

    print(
        f"{'ID':>3s} "
        f"{'Joint':36s} "
        f"{'Initial':>12s} "
        f"{'Final':>12s} "
        f"{'Delta':>12s} "
        f"{'Velocity':>12s}"
    )

    print("-" * 95)

    for joint_id in range(model.njnt):

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        qpos_index = int(
            model.jnt_qposadr[joint_id]
        )

        qvel_index = int(
            model.jnt_dofadr[joint_id]
        )

        initial = float(
            initial_qpos[qpos_index]
        )

        final = float(
            data.qpos[qpos_index]
        )

        delta = final - initial

        velocity = float(
            data.qvel[qvel_index]
        )

        print(
            f"{joint_id:3d} "
            f"{str(joint_name):36s} "
            f"{initial:+12.6f} "
            f"{final:+12.6f} "
            f"{delta:+12.6f} "
            f"{velocity:+12.6f}"
        )

    # ==================================================================
    # 15. Summary
    # ==================================================================

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    if max_contacts == 0:

        print(
            "RESULT: No collision was detected "
            "during the Home Pose test."
        )

        print(
            "The Home Pose itself does not appear "
            "to cause self-collision."
        )

    else:

        print(
            "RESULT: Collision was detected "
            "during the Home Pose test."
        )

        print(
            f"Maximum simultaneous contacts: "
            f"{max_contacts}"
        )

        print(
            f"Unique collision pairs: "
            f"{len(collision_pair_counts)}"
        )

        print(
            "Please inspect COLLISION PAIR STATISTICS "
            "above."
        )

    print(
        "\nViewer will now start with:"
    )

    print("  Collision : ON")
    print("  Actuator  : ON")
    print("  qpos      : WUJI_HOME_JOINT_POS")
    print("  ctrl      : WUJI_HOME_JOINT_POS")

    # ==================================================================
    # 16. Reset to Home Pose before viewer
    # ==================================================================

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    for joint_name, joint_pos in WUJI_HOME_JOINT_POS.items():

        if joint_name in joint_qpos_indices:

            qpos_index = joint_qpos_indices[
                joint_name
            ]

            data.qpos[qpos_index] = joint_pos

    for actuator_id in range(model.nu):

        joint_id = int(
            model.actuator_trnid[actuator_id, 0]
        )

        if joint_id < 0:
            continue

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_id,
        )

        data.ctrl[actuator_id] = (
            WUJI_HOME_JOINT_POS.get(
                joint_name,
                0.0,
            )
        )

    mujoco.mj_forward(model, data)

    # ==================================================================
    # 17. Launch viewer
    # ==================================================================

    print("\n" + "=" * 80)
    print("LAUNCH VIEWER")
    print("=" * 80)

    try:

        import mujoco.viewer

        with mujoco.viewer.launch_passive(
            model,
            data,
        ) as viewer:

            while viewer.is_running():

                mujoco.mj_step(
                    model,
                    data,
                )

                viewer.sync()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
