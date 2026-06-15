#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("gello")
@dataclass
class GelloTeleopConfig(TeleoperatorConfig):
    """Configuration for a GELLO leader device.

    The GELLO hardware returns joint positions, usually 7 values per arm:
    six UR-style arm joints plus one optional gripper signal.
    """

    # Single-arm mode uses single_port. Bimanual mode uses left_port/right_port.
    single_port: str | None = None
    left_port: str | None = None
    right_port: str | None = None
    bimanual: bool = False

    # If the GELLO software is not installed as a package, point this to the
    # folder that contains the `gello/` package, for example .../utils/gello_software.
    gello_software_path: Path | None = None
    auto_discover_ports: bool = False

    num_joints_per_arm: int = 6
    include_gripper: bool = False
    gripper_name: str = "gripper"
    left_prefix: str = "left_"
    right_prefix: str = "right_"

    # The normal UR5e action names used by src/lerobot/robots/ur5e/ur5e.py.
    joint_names: list[str] | None = None

    def __post_init__(self):
        if self.joint_names is None:
            self.joint_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow",
                "wrist_1",
                "wrist_2",
                "wrist_3",
            ]

        if len(self.joint_names) != self.num_joints_per_arm:
            raise ValueError(
                f"joint_names length {len(self.joint_names)} must match "
                f"num_joints_per_arm {self.num_joints_per_arm}"
            )

        if self.bimanual:
            if not self.auto_discover_ports and (self.left_port is None or self.right_port is None):
                raise ValueError("left_port and right_port are required for bimanual GELLO unless auto_discover_ports=True")
        else:
            if not self.auto_discover_ports and self.single_port is None:
                raise ValueError("single_port is required for single-arm GELLO unless auto_discover_ports=True")
