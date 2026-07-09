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
    enable_gripper_action: bool = True
    gripper_name: str = "gripper"
    gripper_backend: str = "tj_channel"
    gripper_command_channel: int | None = None
    gripper_open_hex: str | None = None
    gripper_close_hex: str | None = None
    gripper_close_threshold: float = 0.5
    gripper_send_deadband: float = 0.02
    robotiq_usb_port: str = "auto"
    robotiq_usb_device_id: int = 9
    robotiq_usb_speed: int = 255
    robotiq_usb_force: int = 255
    robotiq_usb_activate_on_connect: bool = True
    robotiq_usb_open_on_connect: bool = False

    velocity_percent: int = 40
    acceleration_percent: int = 40
    connect_frame_checks: int = 0
    connect_frame_check_dt_s: float = 0.01
    check_error_on_connect: bool = True
    wait_response_on_action: bool = False
    action_response_timeout_ms: int = 100
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
        if self.enable_gripper_action:
            if not self.gripper_name:
                raise ValueError("gripper_name must be non-empty when enable_gripper_action is True")
            if self.gripper_backend not in {"tj_channel", "robotiq_usb"}:
                raise ValueError("gripper_backend must be 'tj_channel' or 'robotiq_usb'")
            if (
                self.gripper_command_channel is not None
                and self.gripper_command_channel not in {1, 2, 3}
            ):
                raise ValueError("gripper_command_channel must be 1 (CAN/CANFD), 2 (COM1), or 3 (COM2)")
            if not 0.0 <= self.gripper_close_threshold <= 1.0:
                raise ValueError("gripper_close_threshold must be in [0, 1]")
            if self.gripper_send_deadband < 0:
                raise ValueError("gripper_send_deadband must be non-negative")
            if not 1 <= self.robotiq_usb_device_id <= 247:
                raise ValueError("robotiq_usb_device_id must be in [1, 247]")
            for name, value in {
                "robotiq_usb_speed": self.robotiq_usb_speed,
                "robotiq_usb_force": self.robotiq_usb_force,
            }.items():
                if not 0 <= value <= 255:
                    raise ValueError(f"{name} must be in [0, 255], got {value}")
        for name, value in {
            "velocity_percent": self.velocity_percent,
            "acceleration_percent": self.acceleration_percent,
        }.items():
            if not 0 <= value <= 100:
                raise ValueError(f"{name} must be in [0, 100], got {value}")
        if self.connect_frame_checks < 0:
            raise ValueError("connect_frame_checks must be non-negative")
        if self.action_response_timeout_ms <= 0:
            raise ValueError("action_response_timeout_ms must be positive")
        if self.connect_frame_check_dt_s < 0:
            raise ValueError("connect_frame_check_dt_s must be non-negative")
