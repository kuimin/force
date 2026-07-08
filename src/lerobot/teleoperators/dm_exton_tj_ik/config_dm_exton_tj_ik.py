#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("dm_exton_tj_ik")
@TeleoperatorConfig.register_subclass("DM")
@dataclass
class DMExtonTJIKTeleopConfig(TeleoperatorConfig):
    """Use a DM-EXton2 end-effector pose stream as a TJ joint teleoperator.

    The teleoperator subscribes to a ROS 2 ``geometry_msgs/PoseStamped`` topic
    from DM-EXton2 and solves Tianji/Marvin IK to produce ``joint_*.pos``
    actions for ``TJRobot``.
    """

    arm: str = "B"
    pose_topic: str | None = None
    pose_message_type: str = "pose_stamped"
    pose_array_start_index: int = 7
    sdk_python_dir: Path | None = None
    kine_config_path: Path | None = None
    use_clutch: bool = True
    clutch_topic: str | None = None
    publish_state: bool = True
    state_topic: str | None = None

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
    reference_joints: list[float] = field(
        default_factory=lambda: [-21.8, -41.0, 4.75, -63.67, -10.15, 14.72, -7.68]
    )
    tj_initial_joints: list[float] | None = None
    tj_initial_pose_4x4: list[list[float]] | None = None
    dm_reference_pose: list[float] | None = None
    mapping_mode: str = "position_relative"
    zsp_type: int = 0
    zsp_para: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    zsp_angle: float = 0.0
    dgr1: float = 0.05
    dgr2: float = 0.05

    position_scale: float = 500.0  # DM PoseStamped target is meters; TJ IK expects millimeters.
    state_position_scale: float = 0.001  # TJ FK returns millimeters; PoseStamped state is published in meters.
    position_offset_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_position_offset_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_deadband_mm: float = 3.0
    rotation_deadband_rad: float = 0.003
    max_position_step_mm: float = 10000.0
    max_rotation_step_rad: float = 1.6
    max_accumulated_position_mm: float = 120.0
    reanchor_on_ik_failure: bool = False
    align_master_axes: bool = True
    master_align_z_deg: float = 0.0
    master_align_x_deg: float = 0.0
    master_post_axis_map: list[list[float]] | None = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ]
    )
    mirror_x: bool = False
    mirror_y: bool = False
    mirror_z: bool = False
    hold_last_action_on_ik_failure: bool = True
    fallback_to_position_only_on_ik_failure: bool = False

    ros_node_name: str = "dm_exton_tj_ik_teleop"
    spin_period_s: float = 0.001
    first_pose_timeout_s: float = 10.0
    incoming_command_timeout_s: float = 0.25
    soft_start_duration_s: float = 0.5
    filter_frequency_hz: float = 1000.0
    filter_mincutoff: float = 2.5
    filter_beta: float = 0.2
    filter_dcutoff: float = 1.0

    def __post_init__(self):
        if self.arm not in {"A", "B"}:
            raise ValueError(f"arm must be 'A' or 'B', got {self.arm!r}")
        side = "left" if self.arm == "A" else "right"
        if self.pose_topic is None:
            self.pose_topic = f"/target_robot/{side}_ee/target"
        if self.publish_state and self.state_topic is None:
            self.state_topic = f"/target_robot/{side}_ee/state"
        if self.use_clutch and self.clutch_topic is None:
            self.clutch_topic = f"/clutch/{side}"
        if self.pose_message_type not in {"auto", "pose_stamped", "float_array"}:
            raise ValueError("pose_message_type must be 'auto', 'pose_stamped', or 'float_array'")
        if self.pose_array_start_index < 0:
            raise ValueError("pose_array_start_index must be non-negative")
        if len(self.joint_names) != 7:
            raise ValueError("joint_names must contain exactly 7 joints")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"joint_names must be unique, got {self.joint_names}")
        if len(self.reference_joints) != 7:
            raise ValueError("reference_joints must contain exactly 7 values")
        if self.tj_initial_joints is not None and len(self.tj_initial_joints) != 7:
            raise ValueError("tj_initial_joints must contain exactly 7 values")
        if self.tj_initial_pose_4x4 is not None:
            if len(self.tj_initial_pose_4x4) != 4 or any(len(row) != 4 for row in self.tj_initial_pose_4x4):
                raise ValueError("tj_initial_pose_4x4 must be a 4x4 matrix")
        if self.dm_reference_pose is not None and len(self.dm_reference_pose) != 7:
            raise ValueError("dm_reference_pose must contain [x, y, z, qx, qy, qz, qw]")
        if self.mapping_mode not in {
            "absolute",
            "absolute_position",
            "position_increment",
            "pose_increment",
            "relative",
            "position_relative",
        }:
            raise ValueError(
                "mapping_mode must be 'absolute', 'absolute_position', 'position_increment', "
                "'pose_increment', 'relative', or 'position_relative'"
            )
        if len(self.zsp_para) != 6:
            raise ValueError("zsp_para must contain exactly 6 values")
        if len(self.position_offset_mm) != 3:
            raise ValueError("position_offset_mm must contain exactly 3 values")
        if len(self.target_position_offset_mm) != 3:
            raise ValueError("target_position_offset_mm must contain exactly 3 values")
        if self.master_post_axis_map is not None:
            if len(self.master_post_axis_map) != 3 or any(len(row) != 3 for row in self.master_post_axis_map):
                raise ValueError("master_post_axis_map must be None or a 3x3 matrix")
        if self.position_scale <= 0:
            raise ValueError("position_scale must be positive")
        if self.state_position_scale <= 0:
            raise ValueError("state_position_scale must be positive")
        if self.position_deadband_mm < 0:
            raise ValueError("position_deadband_mm must be non-negative")
        if self.rotation_deadband_rad < 0:
            raise ValueError("rotation_deadband_rad must be non-negative")
        if self.max_position_step_mm <= 0:
            raise ValueError("max_position_step_mm must be positive")
        if self.max_rotation_step_rad <= 0:
            raise ValueError("max_rotation_step_rad must be positive")
        if self.max_accumulated_position_mm <= 0:
            raise ValueError("max_accumulated_position_mm must be positive")
        if self.spin_period_s <= 0:
            raise ValueError("spin_period_s must be positive")
        if self.first_pose_timeout_s <= 0:
            raise ValueError("first_pose_timeout_s must be positive")
        if self.incoming_command_timeout_s <= 0:
            raise ValueError("incoming_command_timeout_s must be positive")
        if self.soft_start_duration_s < 0:
            raise ValueError("soft_start_duration_s must be non-negative")
        if self.filter_frequency_hz <= 0:
            raise ValueError("filter_frequency_hz must be positive")
        if self.filter_mincutoff <= 0:
            raise ValueError("filter_mincutoff must be positive")
        if self.filter_dcutoff <= 0:
            raise ValueError("filter_dcutoff must be positive")
