#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("tj")
@dataclass
class TJRobotConfig(RobotConfig):
    """TJ/Marvin arm controlled through the Tianji Python SDK.

    Joint values are expressed in degrees, matching the vendor SDK.
    """

    ip: str = "192.168.1.190"
    arm: str = "B"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    sdk_python_dir: Path | None = None
    joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]
    )
    joint_limits: list[list[float]] | None = None
    max_relative_target: float | None = 30.0

    velocity_percent: int = 40
    acceleration_percent: int = 40
    connect_frame_checks: int = 5
    connect_frame_check_dt_s: float = 0.01
    check_error_on_connect: bool = True
    log_switch: bool = False
    local_log_switch: bool = False
    disable_on_disconnect: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.arm not in {"A", "B"}:
            raise ValueError(f"arm must be 'A' or 'B', got {self.arm!r}")
        if len(self.joint_names) != 7:
            raise ValueError("TJ joint_names must contain exactly 7 joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"joint_names must be unique, got {self.joint_names}")
        if self.joint_limits is not None:
            if len(self.joint_limits) != 2 or any(len(row) != 7 for row in self.joint_limits):
                raise ValueError("joint_limits must have shape (2, 7)")
        if self.max_relative_target is not None and self.max_relative_target <= 0:
            raise ValueError("max_relative_target must be positive when set")
        for name, value in {
            "velocity_percent": self.velocity_percent,
            "acceleration_percent": self.acceleration_percent,
        }.items():
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100], got {value}")
        if self.connect_frame_checks < 0:
            raise ValueError("connect_frame_checks must be non-negative")
        if self.connect_frame_check_dt_s < 0:
            raise ValueError("connect_frame_check_dt_s must be non-negative")
