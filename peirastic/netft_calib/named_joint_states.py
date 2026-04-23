"""Named joint targets for NetFT calibration (replaces MoveIt named states)."""

from __future__ import annotations

from typing import Dict, List, Sequence

import yaml


_DEFAULT_NAMED_STATES: Dict[str, List[float]] = {
    "ftcalib_jgroup": [0.0, 0.0, 0.0, -2.0, 0.0, 2.0, 0.0],
    "ready": [
        0.09162008114028396,
        -0.19826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.30396583422025,
        0.8480939705504309,
    ],
}


def load_named_joint_states(path: str | None) -> Dict[str, List[float]]:
    states = dict(_DEFAULT_NAMED_STATES)
    if not path:
        return states
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Named joint states YAML must be a mapping: {name: [7 floats]}")
    for name, joints in data.items():
        if not isinstance(joints, (list, tuple)) or len(joints) != 7:
            raise ValueError(f"Invalid joint vector for '{name}': expected length 7")
        states[str(name)] = [float(x) for x in joints]
    return states


def get_named_joints(states: Dict[str, List[float]], name: str) -> Sequence[float]:
    if name not in states:
        known = ", ".join(sorted(states.keys()))
        raise KeyError(f"Unknown named joint group '{name}'. Known: {known}")
    return states[name]
