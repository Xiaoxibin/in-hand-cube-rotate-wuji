from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyServerConfig:
  checkpoint_file: str
  device: str | None = None
  play: bool = True
  raw_joint_pos_obs: bool = True
  """Whether the client sends ABSOLUTE joint positions for observation terms
  whose name matches an action term.

  When True (legacy hardware convention, e.g. real_world/deploy.py), the
  server subtracts that action term's ``default_joint_pos`` before feeding the
  term to the policy.

  The native sim2sim client already evaluates ``joint_pos_rel`` (relative to
  the default pose) and sends that, so it must set this to False. Otherwise
  the default pose is subtracted twice and every ``joint_pos`` history frame
  comes out sign-flipped, which drives the policy far outside its training
  distribution.
  """

