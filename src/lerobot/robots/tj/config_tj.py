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
    gripper_backend: str = "robotiq_usb"
    gripper_send_deadband: float = 0.004
    robotiq_usb_port: str = "auto"
    robotiq_usb_device_id: int = 9
    robotiq_usb_speed: int = 80
    robotiq_usb_force: int = 255
    robotiq_usb_activate_on_connect: bool = True
    robotiq_usb_open_on_connect: bool = False
    enable_dmtac_images: bool = False
    enable_dmtac_force: bool = False
    dmtac_dev_ids: list[int | str] = field(default_factory=lambda: ["auto"])
    dmtac_auto_count: int = 2
    dmtac_sdk_dir: Path | None = Path("/home/robot/daimeng/DM-Tac-SDK/SDK_Publish_V1.2.13.1")
    dmtac_backend: str = "cpu"
    dmtac_mode: str = "high"
    dmtac_max_fps: int = 120
    dmtac_show_fps: bool = False
    dmtac_cpu_fallback: bool = True
    # DM-Tac raw camera frames are 640 x 480 (width x height).
    dmtac_image_height: int = 480
    dmtac_image_width: int = 640
    dmtac_image_channels: int = 3
    dmtac_force_dim: int = 6
    dmtac_wait_timeout_ms: int = 500
    dmtac_remote_addr: str = "192.168.127.10:50051"
    dmtac_pc_host: str = "0.0.0.0"
    dmtac_pc_port: int = 60000

    velocity_percent: int = 40
    acceleration_percent: int = 40
    connect_frame_checks: int = 0
    connect_frame_check_dt_s: float = 0.01
    check_error_on_connect: bool = True
    wait_response_on_action: bool = False
    action_response_timeout_ms: int = 100
    log_switch: bool = False
    silent_sdk: bool = True
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
            if self.gripper_backend != "robotiq_usb":
                raise ValueError("gripper_backend must be 'robotiq_usb'")
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
        if self.enable_dmtac_images or self.enable_dmtac_force:
            if not self.dmtac_dev_ids:
                raise ValueError(
                    "dmtac_dev_ids must contain at least one sensor when enable_dmtac_images or enable_dmtac_force is True"
                )
            if self.dmtac_auto_count <= 0:
                raise ValueError("dmtac_auto_count must be positive")
            if self.dmtac_mode.lower() not in {"standard", "high"}:
                raise ValueError("dmtac_mode must be 'standard' or 'high'")
            if self.dmtac_max_fps <= 0:
                raise ValueError("dmtac_max_fps must be positive")
            if self.dmtac_wait_timeout_ms <= 0:
                raise ValueError("dmtac_wait_timeout_ms must be positive")
            if self.dmtac_force_dim <= 0:
                raise ValueError("dmtac_force_dim must be positive")
            for name, value in {
                "dmtac_image_height": self.dmtac_image_height,
                "dmtac_image_width": self.dmtac_image_width,
                "dmtac_image_channels": self.dmtac_image_channels,
            }.items():
                if value <= 0:
                    raise ValueError(f"{name} must be positive")
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
