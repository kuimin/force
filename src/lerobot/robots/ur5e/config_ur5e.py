#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ur5e")
@dataclass
class UR5eRobotConfig(RobotConfig):
    """UR5e robot controlled through Universal Robots RTDE.

    Joint values are in radians, matching ``ur_rtde`` and the Embodied-RL UR5EJP controller.
    """

    ip: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # RTDE / UR motion parameters copied from the Embodied-RL UR5EJP controller defaults.
    lookahead_time: float = 0.2
    gain: int = 100
    max_joint_speed: float = 0.5
    max_joint_acc: float = 0.5
    control_mode: str = "servoJ"  # "servoJ" for streaming control, "moveJ" for point-to-point motion.
    servo_time: float = 0.02

    tcp_offset_pose: list[float] | None = None
    payload_mass: float | None = None
    payload_cog: list[float] | None = None
    joints_init: list[float] | None = None
    joints_init_speed: float = 1.05
    reset_joints: list[float] | None = None
    joint_limits: list[list[float]] | None = None

    # Safety: clamp target joints relative to current joints before sending.
    max_relative_target: float | None = 0.25

    def __post_init__(self):
        super().__post_init__()
        if self.control_mode not in {"servoJ", "moveJ"}:
            raise ValueError(f"control_mode must be 'servoJ' or 'moveJ', got {self.control_mode!r}")
        for name, value in {
            "tcp_offset_pose": self.tcp_offset_pose,
            "payload_cog": self.payload_cog,
            "joints_init": self.joints_init,
            "reset_joints": self.reset_joints,
        }.items():
            if value is not None and len(value) != (3 if name == "payload_cog" else 6):
                raise ValueError(f"{name} has invalid length: {len(value)}")
        if self.joint_limits is not None:
            if len(self.joint_limits) != 2 or any(len(row) != 6 for row in self.joint_limits):
                raise ValueError("joint_limits must have shape (2, 6)")
