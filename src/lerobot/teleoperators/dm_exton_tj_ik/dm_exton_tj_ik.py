#!/usr/bin/env python

import importlib.util
import logging
import math
import threading
import time
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_dm_exton_tj_ik import DMExtonTJIKTeleopConfig

logger = logging.getLogger(__name__)


def _third_party_tj_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "third_party" / "tj_marvin_sdk"


def _default_sdk_python_dir() -> Path:
    return _third_party_tj_dir() / "SDK_PYTHON"


def _default_kine_config_path() -> Path:
    return _third_party_tj_dir() / "CommonConfig" / "ccs_m6_40.MvKDCfg"


def _load_fx_kine(sdk_python_dir: Path) -> ModuleType:
    sdk_file = sdk_python_dir / "fx_kine.py"
    if not sdk_file.is_file():
        raise ImportError(f"TJ kinematics SDK file not found: {sdk_file}")
    spec = importlib.util.spec_from_file_location("lerobot_tj_fx_kine", sdk_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TJ kinematics SDK from {sdk_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _LowPassFilter:
    def __init__(self, alpha: float):
        self._set_alpha(alpha)
        self.y = None
        self.s = None

    def _set_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if alpha <= 0 or alpha > 1.0:
            raise ValueError(f"alpha ({alpha}) should be in (0.0, 1.0]")
        self.alpha = alpha

    def filter(self, value: np.ndarray, alpha: float | None = None) -> np.ndarray:
        if alpha is not None:
            self._set_alpha(alpha)
        if self.y is None:
            s = value
        else:
            s = self.alpha * value + (1.0 - self.alpha) * self.s
        self.y = value
        self.s = s
        return s

    def last_value(self) -> np.ndarray | None:
        return self.y


class _OneEuroFilter:
    def __init__(self, freq: float, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        if freq <= 0:
            raise ValueError("freq should be > 0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be > 0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be > 0")
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x = _LowPassFilter(self._alpha(self.mincutoff))
        self.dx = _LowPassFilter(self._alpha(self.dcutoff))
        self.last_time = None

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, value: np.ndarray, timestamp: float | None = None) -> np.ndarray:
        if self.last_time is not None and timestamp is not None and timestamp > self.last_time:
            rate = 1.0 / (timestamp - self.last_time)
        else:
            rate = self.freq
        self.last_time = timestamp

        last_value = self.x.last_value()
        dx = np.zeros_like(value) if last_value is None else (value - last_value) * rate
        edx = self.dx.filter(dx, alpha=self._alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.linalg.norm(edx)
        return self.x.filter(value, alpha=self._alpha(cutoff))


class _PoseFilter:
    def __init__(self, freq: float, mincutoff: float, beta: float, dcutoff: float):
        self.pos_filter = _OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.quat_filter = _OneEuroFilter(freq, mincutoff, beta, dcutoff)

    def process(
        self, position_xyz: list[float] | np.ndarray, quat_xyzw: list[float] | np.ndarray, timestamp: float | None
    ) -> tuple[list[float], list[float]]:
        position = np.asarray(position_xyz, dtype=np.float64)
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        filtered_position = self.pos_filter.filter(position, timestamp)
        filtered_quat = self.quat_filter.filter(quat, timestamp)
        norm = np.linalg.norm(filtered_quat)
        if norm > 1e-9:
            filtered_quat = filtered_quat / norm
        else:
            filtered_quat = quat
        return filtered_position.tolist(), filtered_quat.tolist()


def _quat_xyzw_to_matrix(quat_xyzw: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-9:
        raise ValueError("Quaternion norm is zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s

    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 1e-9:
        quat /= norm
    return quat


def _rotation_angle_rad(rotation: np.ndarray) -> float:
    cos_angle = (float(np.trace(rotation)) - 1.0) * 0.5
    return float(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def _rotation_axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / axis_norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def _rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _master_to_tj_axes_rotation(
    z_deg: float,
    x_deg: float,
    post_axis_map: list[list[float]] | None = None,
) -> np.ndarray:
    rotation = _rotation_x(math.radians(x_deg)) @ _rotation_z(math.radians(z_deg))
    if post_axis_map is None:
        return rotation
    post_rotation = np.asarray(post_axis_map, dtype=np.float64)
    if post_rotation.shape != (3, 3):
        raise ValueError(f"post_axis_map must be a 3x3 matrix, got {post_rotation.shape}")
    return post_rotation @ rotation


def _change_increment_frame(
    matrix: np.ndarray,
    frame_rotation: np.ndarray,
    orientation_frame_rotation: np.ndarray | None = None,
) -> np.ndarray:
    aligned = matrix.copy()
    aligned[:3, 3] = frame_rotation @ matrix[:3, 3]
    rotation = frame_rotation if orientation_frame_rotation is None else orientation_frame_rotation
    aligned[:3, :3] = rotation @ matrix[:3, :3] @ rotation.T
    return aligned


def _clip_rotation_step(rotation: np.ndarray, max_angle_rad: float) -> np.ndarray:
    angle = _rotation_angle_rad(rotation)
    if angle <= max_angle_rad:
        return rotation
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    return _rotation_axis_angle_to_matrix(axis, max_angle_rad)


def _pose_to_tj_matrix_1x16(
    position_xyz: list[float] | np.ndarray,
    quat_xyzw: list[float] | np.ndarray,
    position_scale: float,
    position_offset_mm: list[float],
    mirror_xyz: tuple[bool, bool, bool],
) -> list[float]:
    position = np.asarray(position_xyz, dtype=np.float64) * position_scale
    mirror = np.asarray([-1.0 if flag else 1.0 for flag in mirror_xyz], dtype=np.float64)
    position = position * mirror + np.asarray(position_offset_mm, dtype=np.float64)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_matrix(quat_xyzw)
    matrix[:3, 3] = position
    return matrix.reshape(-1).tolist()


def _pose_to_tj_matrix(
    position_xyz: list[float] | np.ndarray,
    quat_xyzw: list[float] | np.ndarray,
    position_scale: float,
    position_offset_mm: list[float],
    mirror_xyz: tuple[bool, bool, bool],
) -> np.ndarray:
    return np.asarray(
        _pose_to_tj_matrix_1x16(position_xyz, quat_xyzw, position_scale, position_offset_mm, mirror_xyz),
        dtype=np.float64,
    ).reshape(4, 4)


def _map_dm_pose_to_tj_pose(
    dm_matrix: np.ndarray,
    dm_reference_matrix: np.ndarray,
    tj_reference_matrix: np.ndarray,
    mapping_mode: str,
) -> np.ndarray:
    if mapping_mode == "absolute":
        return dm_matrix.copy()
    if mapping_mode == "absolute_position":
        target = tj_reference_matrix.copy()
        target[:3, 3] = dm_matrix[:3, 3]
        return target
    if mapping_mode == "position_increment":
        target = tj_reference_matrix.copy()
        target[:3, 3] = tj_reference_matrix[:3, 3] + dm_matrix[:3, 3]
        return target
    if mapping_mode == "pose_increment":
        target = tj_reference_matrix.copy()
        target[:3, :3] = tj_reference_matrix[:3, :3] @ dm_matrix[:3, :3]
        target[:3, 3] = tj_reference_matrix[:3, 3] + dm_matrix[:3, 3]
        return target
    if mapping_mode == "position_relative":
        target = tj_reference_matrix.copy()
        target[:3, 3] = tj_reference_matrix[:3, 3] + (dm_matrix[:3, 3] - dm_reference_matrix[:3, 3])
        return target
    if mapping_mode == "relative":
        target = tj_reference_matrix.copy()
        target[:3, 3] = tj_reference_matrix[:3, 3] + (dm_matrix[:3, 3] - dm_reference_matrix[:3, 3])
        dm_delta_in_base = dm_matrix[:3, :3] @ dm_reference_matrix[:3, :3].T
        target[:3, :3] = tj_reference_matrix[:3, :3] @ dm_delta_in_base
        return target
    raise ValueError(f"Unsupported mapping_mode: {mapping_mode}")


def _apply_target_position_offset(matrix: np.ndarray, offset_mm: list[float]) -> np.ndarray:
    offset = np.asarray(offset_mm, dtype=np.float64)
    if np.allclose(offset, 0.0):
        return matrix
    target = matrix.copy()
    target[:3, 3] += offset
    return target


def _interpolate_matrix(start: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    matrix = target.copy()
    matrix[:3, 3] = start[:3, 3] * (1.0 - alpha) + target[:3, 3] * alpha
    return matrix


class DMExtonTJIKTeleop(Teleoperator):
    config_class = DMExtonTJIKTeleopConfig
    name = "DM"

    def __init__(self, config: DMExtonTJIKTeleopConfig):
        super().__init__(config)
        self.config = config
        self.fx_kine = None
        self.kine = None
        self.rclpy = None
        self.node = None
        self.subscription = None
        self.clutch_subscription = None
        self.gripper_subscription = None
        self.state_publisher = None
        self.pose_stamped_cls = None
        self._spin_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_pose: tuple[list[float], list[float], int, float] | None = None
        self._latest_gripper = 0.0
        self._gripper_count = 0
        self._last_gripper_log_s = 0.0
        self._last_action: RobotAction | None = None
        self._current_joint_action: RobotAction | None = None
        self._dm_reference_matrix: np.ndarray | None = None
        self._tj_reference_matrix: np.ndarray | None = None
        self._dm_increment_accumulated_xyz = np.zeros(3, dtype=np.float64)
        self._dm_increment_accumulated_matrix = np.eye(4, dtype=np.float64)
        self._last_increment_pose_id = 0
        self._last_ik_failure: str | None = None
        self._last_ik_joints: list[float] | None = None
        self._current_tj_matrix: np.ndarray | None = None
        self._last_aligned_dm_matrix: np.ndarray | None = None
        self._last_target_matrix: np.ndarray | None = None
        self._tj_reference_from_feedback = False
        self._clutch = not self.config.use_clutch
        self._prev_clutch = self._clutch
        self._pending_clutch_anchor = False
        self._clutch_anchor_begin_s: float | None = None
        self._soft_start_anchor_matrix: np.ndarray | None = None
        self._soft_start_begin_s: float | None = None
        self._last_debug_log_s = 0.0
        self._last_target_log_s = 0.0
        self._last_pose_log_s = 0.0
        self._last_wait_log_s = 0.0
        self._pose_count = 0
        self._warned_position_only_fallback = False
        self._pose_filter = _PoseFilter(
            freq=self.config.filter_frequency_hz,
            mincutoff=self.config.filter_mincutoff,
            beta=self.config.filter_beta,
            dcutoff=self.config.filter_dcutoff,
        )
        self._gripper_filter = _OneEuroFilter(
            freq=self.config.filter_frequency_hz,
            mincutoff=self.config.gripper_filter_mincutoff,
            beta=self.config.gripper_filter_beta,
            dcutoff=self.config.gripper_filter_dcutoff,
        )
        self._master_to_tj_rotation = _master_to_tj_axes_rotation(
            self.config.master_align_z_deg,
            self.config.master_align_x_deg,
            self.config.master_post_axis_map,
        )
        self._master_to_tj_orientation_rotation = _master_to_tj_axes_rotation(
            self.config.master_align_z_deg,
            self.config.master_align_x_deg,
            self.config.master_orientation_axis_map,
        )
        logger.debug(
            "DM to TJ axis map=%s",
            np.round(self._master_to_tj_rotation, 6).tolist(),
        )
        logger.debug(
            "DM to TJ orientation axis map=%s",
            np.round(self._master_to_tj_orientation_rotation, 6).tolist(),
        )

    def get_pose_log_snapshot(self) -> dict[str, float | int | bool | None]:
        with self._lock:
            latest_pose = self._latest_pose
            clutch = bool(self._clutch)
            gripper = float(self._latest_gripper)

        row: dict[str, float | int | bool | None] = {
            "clutch": clutch,
            "gripper": gripper,
            "dm_pose_id": None,
            "dm_age_s": None,
            "dm_x_m": None,
            "dm_y_m": None,
            "dm_z_m": None,
            "dm_qx": None,
            "dm_qy": None,
            "dm_qz": None,
            "dm_qw": None,
            "tj_x_mm": None,
            "tj_y_mm": None,
            "tj_z_mm": None,
            "tj_qx": None,
            "tj_qy": None,
            "tj_qz": None,
            "tj_qw": None,
            "dm_aligned_x_mm": None,
            "dm_aligned_y_mm": None,
            "dm_aligned_z_mm": None,
            "dm_aligned_qx": None,
            "dm_aligned_qy": None,
            "dm_aligned_qz": None,
            "dm_aligned_qw": None,
            "target_x_mm": None,
            "target_y_mm": None,
            "target_z_mm": None,
            "target_qx": None,
            "target_qy": None,
            "target_qz": None,
            "target_qw": None,
        }
        if latest_pose is not None:
            position, quat, pose_id, received_s = latest_pose
            row.update(
                {
                    "dm_pose_id": int(pose_id),
                    "dm_age_s": float(time.monotonic() - received_s),
                    "dm_x_m": float(position[0]),
                    "dm_y_m": float(position[1]),
                    "dm_z_m": float(position[2]),
                    "dm_qx": float(quat[0]),
                    "dm_qy": float(quat[1]),
                    "dm_qz": float(quat[2]),
                    "dm_qw": float(quat[3]),
                }
            )

        if self._current_tj_matrix is not None:
            quat = _matrix_to_quat_xyzw(self._current_tj_matrix[:3, :3])
            row.update(
                {
                    "tj_x_mm": float(self._current_tj_matrix[0, 3]),
                    "tj_y_mm": float(self._current_tj_matrix[1, 3]),
                    "tj_z_mm": float(self._current_tj_matrix[2, 3]),
                    "tj_qx": float(quat[0]),
                    "tj_qy": float(quat[1]),
                    "tj_qz": float(quat[2]),
                    "tj_qw": float(quat[3]),
                }
            )
        if self._last_aligned_dm_matrix is not None:
            quat = _matrix_to_quat_xyzw(self._last_aligned_dm_matrix[:3, :3])
            row.update(
                {
                    "dm_aligned_x_mm": float(self._last_aligned_dm_matrix[0, 3]),
                    "dm_aligned_y_mm": float(self._last_aligned_dm_matrix[1, 3]),
                    "dm_aligned_z_mm": float(self._last_aligned_dm_matrix[2, 3]),
                    "dm_aligned_qx": float(quat[0]),
                    "dm_aligned_qy": float(quat[1]),
                    "dm_aligned_qz": float(quat[2]),
                    "dm_aligned_qw": float(quat[3]),
                }
            )
        if self._last_target_matrix is not None:
            quat = _matrix_to_quat_xyzw(self._last_target_matrix[:3, :3])
            row.update(
                {
                    "target_x_mm": float(self._last_target_matrix[0, 3]),
                    "target_y_mm": float(self._last_target_matrix[1, 3]),
                    "target_z_mm": float(self._last_target_matrix[2, 3]),
                    "target_qx": float(quat[0]),
                    "target_qy": float(quat[1]),
                    "target_qz": float(quat[2]),
                    "target_qw": float(quat[3]),
                }
            )
        return row

    def _reset_pose_filter(self) -> None:
        self._pose_filter = _PoseFilter(
            freq=self.config.filter_frequency_hz,
            mincutoff=self.config.filter_mincutoff,
            beta=self.config.filter_beta,
            dcutoff=self.config.filter_dcutoff,
        )

    @property
    def _arm_index(self) -> int:
        return 0 if self.config.arm == "A" else 1

    @cached_property
    def action_features(self) -> dict[str, type]:
        features = {f"{joint}.pos": float for joint in self.config.joint_names}
        if self.config.enable_gripper:
            features[f"{self.config.gripper_name}.pos"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.node is not None and self.kine is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._init_tj_kine()
        self._init_ros()
        logger.info("%s listening to DM-EXton2 pose topic %s", self, self.config.pose_topic)

    def _init_tj_kine(self) -> None:
        sdk_python_dir = self.config.sdk_python_dir or _default_sdk_python_dir()
        kine_config_path = self.config.kine_config_path or _default_kine_config_path()
        self.fx_kine = _load_fx_kine(sdk_python_dir)
        self.kine = self.fx_kine.Marvin_Kine()
        self.kine.log_switch(0)

        loaded = self.kine.load_config(self._arm_index, str(kine_config_path))
        if not loaded:
            raise RuntimeError(f"Failed to load TJ kinematics config: {kine_config_path}")
        ok = self.kine.initial_kine(
            int(loaded["TYPE"][self._arm_index]),
            loaded["DH"][self._arm_index],
            loaded["PNVA"][self._arm_index],
            loaded["BD"][self._arm_index],
        )
        if not ok:
            raise RuntimeError(f"Failed to initialize TJ kinematics from {kine_config_path}")
        self._last_ik_joints = list(self.config.tj_initial_joints or self.config.reference_joints)
        self._tj_reference_matrix = self._load_tj_reference_matrix()
        if self.config.dm_reference_pose is not None:
            self._dm_reference_matrix = _pose_to_tj_matrix(
                self.config.dm_reference_pose[:3],
                self.config.dm_reference_pose[3:],
                self.config.position_scale,
                self.config.position_offset_mm,
                (self.config.mirror_x, self.config.mirror_y, self.config.mirror_z),
            )

    def _load_tj_reference_matrix(self) -> np.ndarray:
        assert self.kine is not None
        if self.config.tj_initial_pose_4x4 is not None:
            return np.asarray(self.config.tj_initial_pose_4x4, dtype=np.float64)

        joints = self.config.tj_initial_joints or self.config.reference_joints
        if self.config.tj_initial_joints is None and self.config.mapping_mode != "absolute":
            logger.warning("tj_initial_joints is not set; using reference_joints as TJ mapping reference")
        fk = self.kine.fk(joints)
        if not fk:
            raise RuntimeError(f"Failed to compute TJ FK for reference joints: {joints}")
        matrix = np.asarray(fk, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise RuntimeError(f"TJ FK returned {matrix.shape}, expected (4, 4)")
        return matrix

    def _init_ros(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from std_msgs.msg import Bool, Float32MultiArray, Float64MultiArray
        except ImportError as exc:
            raise ImportError(
                "DM-EXton2 TJ IK teleop requires ROS 2 Python packages: rclpy, geometry_msgs, and std_msgs"
            ) from exc

        self.rclpy = rclpy
        self.pose_stamped_cls = PoseStamped
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(self.config.ros_node_name)
        message_type = self._resolve_pose_message_type()
        logger.info("Resolved DM pose topic %s as %s", self.config.pose_topic, message_type)
        if message_type == "pose_stamped":
            msg_cls = PoseStamped
        elif message_type == "float32_array":
            msg_cls = Float32MultiArray
        elif message_type == "float64_array":
            msg_cls = Float64MultiArray
        else:
            raise ValueError(f"Unsupported resolved pose message type: {message_type}")

        self.subscription = self.node.create_subscription(msg_cls, self.config.pose_topic, self._pose_callback, 1)
        if self.config.publish_state and self.config.state_topic is not None:
            self.state_publisher = self.node.create_publisher(PoseStamped, self.config.state_topic, 10)
        if self.config.use_clutch and self.config.clutch_topic is not None:
            self.clutch_subscription = self.node.create_subscription(
                Bool, self.config.clutch_topic, self._clutch_callback, 1
            )
        if self.config.enable_gripper and self.config.gripper_topic is not None:
            self.gripper_subscription = self.node.create_subscription(
                Float64MultiArray, self.config.gripper_topic, self._gripper_callback, 1
            )
            logger.info(
                "DM-EXton2 gripper trigger listening on %s[%s] -> %s.pos",
                self.config.gripper_topic,
                self.config.gripper_index,
                self.config.gripper_name,
            )
        self._stop_event.clear()
        self._spin_thread = threading.Thread(target=self._spin_ros, name=self.config.ros_node_name, daemon=True)
        self._spin_thread.start()

    def _resolve_pose_message_type(self) -> str:
        assert self.node is not None
        requested = self.config.pose_message_type
        if requested == "pose_stamped":
            return "pose_stamped"

        topic_types: list[str] = []
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            names_and_types = self.node.get_topic_names_and_types()
            topic_types = next((types for name, types in names_and_types if name == self.config.pose_topic), [])
            if topic_types:
                break
            time.sleep(0.05)

        if requested == "float_array":
            if "std_msgs/msg/Float32MultiArray" in topic_types:
                return "float32_array"
            if "std_msgs/msg/Float64MultiArray" in topic_types:
                return "float64_array"
            logger.warning(
                "Could not resolve %s array type from ROS graph; defaulting to Float32MultiArray",
                self.config.pose_topic,
            )
            return "float32_array"

        if "geometry_msgs/msg/PoseStamped" in topic_types:
            return "pose_stamped"
        if "std_msgs/msg/Float32MultiArray" in topic_types:
            return "float32_array"
        if "std_msgs/msg/Float64MultiArray" in topic_types:
            return "float64_array"

        logger.warning(
            "Could not resolve %s message type from ROS graph; defaulting to PoseStamped",
            self.config.pose_topic,
        )
        return "pose_stamped"

    def _spin_ros(self) -> None:
        assert self.rclpy is not None and self.node is not None
        while not self._stop_event.is_set() and self.rclpy.ok():
            try:
                self.rclpy.spin_once(self.node, timeout_sec=self.config.spin_period_s)
            except Exception as exc:
                if exc.__class__.__name__ == "ExternalShutdownException":
                    break
                raise

    def _pose_callback(self, msg: Any) -> None:
        position, quat = self._extract_pose_from_msg(msg)
        timestamp = self._timestamp_from_msg(msg)
        if self.config.mapping_mode not in {"position_increment", "pose_increment"}:
            position, quat = self._pose_filter.process(position, quat, timestamp)
        else:
            quat_array = np.asarray(quat, dtype=np.float64)
            norm = np.linalg.norm(quat_array)
            if norm > 1e-9:
                quat = (quat_array / norm).tolist()
        self._pose_count += 1
        now = time.monotonic()
        if self._pose_count == 1 or now - self._last_pose_log_s >= 1.0:
            logger.info(
                "Received DM pose #%s position=%s quat=%s",
                self._pose_count,
                [round(float(v), 6) for v in position],
                [round(float(v), 6) for v in quat],
            )
            self._last_pose_log_s = now
        with self._lock:
            self._latest_pose = (position, quat, self._pose_count, now)

    def _clutch_callback(self, msg: Any) -> None:
        with self._lock:
            self._clutch = bool(msg.data)

    def _gripper_callback(self, msg: Any) -> None:
        data = list(getattr(msg, "data", []))
        index = self.config.gripper_index
        if index is None or index >= len(data):
            now = time.monotonic()
            if now - self._last_gripper_log_s >= 1.0:
                logger.warning(
                    "Gripper trigger topic %s has %s values; cannot read index %s",
                    self.config.gripper_topic,
                    len(data),
                    index,
                )
                self._last_gripper_log_s = now
            return

        value = float(np.clip(float(data[index]), 0.0, 1.0))
        if self.config.gripper_invert:
            value = 1.0 - value
        filtered_value = float(
            np.clip(self._gripper_filter.filter(np.asarray([value], dtype=np.float64), time.monotonic())[0], 0.0, 1.0)
        )
        with self._lock:
            self._latest_gripper = filtered_value
            self._gripper_count += 1
        now = time.monotonic()
        if self._gripper_count == 1 or now - self._last_gripper_log_s >= 1.0:
            logger.info(
                "Received gripper trigger %s.pos=%.3f filtered=%.3f",
                self.config.gripper_name,
                value,
                filtered_value,
            )
            self._last_gripper_log_s = now

    def _timestamp_from_msg(self, msg: Any) -> float | None:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return time.monotonic()
        sec = float(getattr(stamp, "sec", 0.0))
        nanosec = float(getattr(stamp, "nanosec", 0.0))
        timestamp = sec + nanosec * 1e-9
        return timestamp if timestamp > 0 else time.monotonic()

    def _extract_pose_from_msg(self, msg: Any) -> tuple[list[float], list[float]]:
        if hasattr(msg, "pose"):
            position = [float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z)]
            quat = [
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
                float(msg.pose.orientation.w),
            ]
            return position, quat

        if hasattr(msg, "data"):
            data = list(msg.data)
            start = self.config.pose_array_start_index
            values = data[start : start + 7]
            if len(values) != 7:
                raise RuntimeError(
                    f"DM pose array on {self.config.pose_topic} has {len(data)} values; "
                    f"cannot read 7 values from index {start}"
                )
            return [float(v) for v in values[:3]], [float(v) for v in values[3:]]

        raise TypeError(f"Unsupported DM pose message type: {type(msg)!r}")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def _latest_pose_or_none(self) -> tuple[list[float], list[float], int] | None:
        with self._lock:
            pose = self._latest_pose
        if pose is not None:
            position, quat, pose_id, received_s = pose
            if self.config.mapping_mode not in {"position_increment", "pose_increment"}:
                return position, quat, pose_id
            now = time.monotonic()
            age_s = now - received_s
            if age_s <= self.config.incoming_command_timeout_s:
                return position, quat, pose_id
            if now - self._last_wait_log_s >= 1.0:
                logger.warning(
                    "Halting DM-EXton2 teleop: latest pose on %s is stale (age %.3fs > %.3fs)",
                    self.config.pose_topic,
                    age_s,
                    self.config.incoming_command_timeout_s,
                )
                self._last_wait_log_s = now
            return None

        now = time.monotonic()
        if now - self._last_wait_log_s >= 1.0:
            topic_types = []
            if self.node is not None:
                topic_types = next(
                    (types for name, types in self.node.get_topic_names_and_types() if name == self.config.pose_topic),
                    [],
                )
            if topic_types:
                detail = f"topic exists with type(s): {topic_types}"
            else:
                detail = "topic is not visible in the ROS graph"
            logger.warning("Waiting for DM-EXton2 pose on %s; %s", self.config.pose_topic, detail)
            self._last_wait_log_s = now
        return None

    def _solve_ik_matrix(self, matrix_4x4: np.ndarray) -> list[float] | None:
        assert self.fx_kine is not None and self.kine is not None
        solve_para = self.fx_kine.FX_InvKineSolvePara()
        solve_para.set_input_ik_target_tcp(matrix_4x4.reshape(-1).tolist())
        ref_joints = list(
            self._last_ik_joints
            or (
                [self._current_joint_action[f"{joint}.pos"] for joint in self.config.joint_names]
                if self._current_joint_action is not None
                else self.config.reference_joints
            )
        )
        solve_para.set_input_ik_ref_joint(ref_joints)
        solve_para.set_input_ik_zsp_type(self.config.zsp_type)
        solve_para.set_input_ik_zsp_para(self.config.zsp_para)
        solve_para.set_input_zsp_angle(self.config.zsp_angle)
        solve_para.set_dgr1(self.config.dgr1)
        solve_para.set_dgr2(self.config.dgr2)
        result = self.kine.ik(solve_para)
        if not result or result.m_Output_IsOutRange or result.m_Output_IsJntExd:
            self._last_ik_failure = self._format_ik_failure(result, matrix_4x4)
            return None
        self._last_ik_failure = None
        joints = [float(value) for value in result.m_Output_RetJoint.to_list()]
        self._last_ik_joints = joints
        now = time.monotonic()
        if now - self._last_debug_log_s >= 0.5:
            logger.info(
                "DM target xyz(mm)=%s IK q(deg)=%s",
                [round(float(v), 3) for v in matrix_4x4[:3, 3]],
                [round(float(v), 3) for v in joints],
            )
            self._last_debug_log_s = now
        return joints

    def _format_ik_failure(self, result: Any, matrix_4x4: np.ndarray) -> str:
        xyz = np.round(matrix_4x4[:3, 3], 3).tolist()
        quat = np.round(_matrix_to_quat_xyzw(matrix_4x4[:3, :3]), 6).tolist()
        current_xyz = None
        delta_xyz = None
        if self._current_tj_matrix is not None:
            current_xyz_arr = self._current_tj_matrix[:3, 3]
            current_xyz = np.round(current_xyz_arr, 3).tolist()
            delta_xyz = np.round(matrix_4x4[:3, 3] - current_xyz_arr, 3).tolist()
        if not result:
            return (
                f"no IK result, target xyz(mm)={xyz}, target quat={quat}, "
                f"current xyz(mm)={current_xyz}, delta xyz(mm)={delta_xyz}"
            )

        out_range = bool(getattr(result, "m_Output_IsOutRange", False))
        joint_exceed = bool(getattr(result, "m_Output_IsJntExd", False))
        joints = getattr(result, "m_Output_RetJoint", None)
        if joints is not None and hasattr(joints, "to_list"):
            joints = [round(float(value), 3) for value in joints.to_list()]
        tags = getattr(result, "m_Output_JntExdTag", None)
        if tags is not None and hasattr(tags, "to_list"):
            tags = [bool(value) for value in tags.to_list()]
        return (
            f"target xyz(mm)={xyz}, target quat={quat}, current xyz(mm)={current_xyz}, "
            f"delta xyz(mm)={delta_xyz}, out_range={out_range}, "
            f"joint_exceed={joint_exceed}, q={joints}, joint_exceed_tags={tags}"
        )

    def _commanded_action_or_reference(self) -> RobotAction:
        if self._last_action is not None:
            return dict(self._last_action)
        joints = list(self._last_ik_joints or self.config.tj_initial_joints or self.config.reference_joints)
        action = {f"{joint}.pos": value for joint, value in zip(self.config.joint_names, joints, strict=True)}
        self._add_gripper_action(action)
        self._last_action = action
        return dict(action)

    def _halt_action_or_reference(self) -> RobotAction:
        if self._current_joint_action is not None:
            action = dict(self._current_joint_action)
            self._add_gripper_action(action)
            self._last_action = action
            return dict(action)
        return self._commanded_action_or_reference()

    def _add_gripper_action(self, action: RobotAction) -> None:
        if not self.config.enable_gripper:
            return
        with self._lock:
            action[f"{self.config.gripper_name}.pos"] = float(self._latest_gripper)

    def _apply_clutch_and_soft_start(self, target_matrix: np.ndarray) -> np.ndarray | None:
        now = time.monotonic()
        with self._lock:
            clutch = self._clutch

        if not self.config.use_clutch:
            return target_matrix
        if not clutch:
            self._prev_clutch = False
            self._soft_start_anchor_matrix = None
            self._soft_start_begin_s = None
            return None

        if not self._prev_clutch:
            self._soft_start_anchor_matrix = target_matrix.copy()
            self._soft_start_begin_s = now
            self._prev_clutch = True

        if self.config.soft_start_duration_s == 0 or self._soft_start_begin_s is None:
            return target_matrix

        elapsed_s = now - self._soft_start_begin_s
        alpha = elapsed_s / self.config.soft_start_duration_s
        if alpha >= 1.0 or self._soft_start_anchor_matrix is None:
            return target_matrix
        return _interpolate_matrix(self._soft_start_anchor_matrix, target_matrix, alpha)

    def _prepare_clutch_relative_anchor(self, dm_matrix: np.ndarray) -> bool:
        if not self.config.use_clutch:
            return True

        with self._lock:
            clutch = self._clutch

        if not clutch:
            self._prev_clutch = False
            self._pending_clutch_anchor = False
            self._clutch_anchor_begin_s = None
            self._soft_start_anchor_matrix = None
            self._soft_start_begin_s = None
            return False

        if self._prev_clutch:
            return True

        reference = self._current_tj_matrix.copy() if self._current_tj_matrix is not None else None
        if reference is None:
            now = time.monotonic()
            if now - self._last_wait_log_s >= 1.0:
                logger.warning("Waiting for current TJ feedback before clutch anchor; holding current action")
                self._last_wait_log_s = now
            return False

        now = time.monotonic()
        if not self._pending_clutch_anchor:
            self._pending_clutch_anchor = True
            self._clutch_anchor_begin_s = now
            self._soft_start_anchor_matrix = reference.copy()
            self._soft_start_begin_s = now

        self._tj_reference_matrix = reference
        if self.config.mapping_mode in {"relative", "position_relative"}:
            self._dm_reference_matrix = dm_matrix.copy()
        elif self.config.mapping_mode in {"position_increment", "pose_increment"}:
            self._dm_increment_accumulated_xyz = np.zeros(3, dtype=np.float64)
            self._dm_increment_accumulated_matrix = np.eye(4, dtype=np.float64)

        if self._clutch_anchor_begin_s is not None and now - self._clutch_anchor_begin_s < self.config.clutch_anchor_delay_s:
            return False

        self._pending_clutch_anchor = False
        self._clutch_anchor_begin_s = None
        self._tj_reference_matrix = reference
        if self.config.mapping_mode in {"relative", "position_relative"}:
            self._dm_reference_matrix = dm_matrix.copy()
        elif self.config.mapping_mode in {"position_increment", "pose_increment"}:
            self._dm_increment_accumulated_xyz = np.zeros(3, dtype=np.float64)
            self._dm_increment_accumulated_matrix = np.eye(4, dtype=np.float64)
        self._soft_start_anchor_matrix = reference.copy()
        self._soft_start_begin_s = time.monotonic()
        self._prev_clutch = True
        logger.info(
            "Clutch engaged; anchored DM xyz(mm)=%s to TJ xyz(mm)=%s",
            [round(float(v), 3) for v in dm_matrix[:3, 3]],
            [round(float(v), 3) for v in reference[:3, 3]],
        )
        return True

    def _anchor_motion_from_current_robot(self, skip_pose_id: int | None = None) -> None:
        reference = (
            self._current_tj_matrix.copy()
            if self._current_tj_matrix is not None
            else self._tj_reference_matrix.copy()
            if self._tj_reference_matrix is not None
            else None
        )
        if reference is None:
            return
        self._tj_reference_matrix = reference
        self._dm_reference_matrix = None
        self._dm_increment_accumulated_xyz = np.zeros(3, dtype=np.float64)
        self._dm_increment_accumulated_matrix = np.eye(4, dtype=np.float64)
        self._last_increment_pose_id = int(skip_pose_id or 0)
        self._soft_start_anchor_matrix = reference.copy()
        self._soft_start_begin_s = time.monotonic()
        self._reset_pose_filter()
        logger.info(
            "Clutch engaged; anchored TJ reference xyz(mm)=%s",
            [round(float(v), 3) for v in reference[:3, 3]],
        )

    def _solve_ik(self, position: list[float], quat_xyzw: list[float], pose_id: int | None = None) -> list[float] | None:
        increment_state = self._increment_state_snapshot()
        dm_matrix = _pose_to_tj_matrix(
            position,
            quat_xyzw,
            self.config.position_scale,
            self.config.position_offset_mm,
            (self.config.mirror_x, self.config.mirror_y, self.config.mirror_z),
        )
        if self.config.align_master_axes and self.config.mapping_mode in {
            "absolute",
            "absolute_position",
            "position_increment",
            "pose_increment",
            "relative",
            "position_relative",
        }:
            dm_matrix = _change_increment_frame(
                dm_matrix,
                self._master_to_tj_rotation,
                self._master_to_tj_orientation_rotation,
            )
        self._last_aligned_dm_matrix = dm_matrix.copy()
        if not self._prepare_clutch_relative_anchor(dm_matrix):
            return None
        target_matrix = self._target_matrix_from_dm_matrix(dm_matrix, pose_id)
        target_matrix = _apply_target_position_offset(target_matrix, self.config.target_position_offset_mm)
        target_matrix = self._apply_clutch_and_soft_start(target_matrix)
        if target_matrix is None:
            return None
        self._last_target_matrix = target_matrix.copy()
        joints = self._solve_ik_matrix(target_matrix)
        if joints is not None:
            return joints
        self._restore_increment_state(increment_state)

        if self.config.mapping_mode in {"absolute_position", "position_increment", "pose_increment"}:
            self._last_ik_failure = f"position-only IK failed: {self._last_ik_failure}"
            return None

        full_pose_failure = self._last_ik_failure
        if self.config.fallback_to_position_only_on_ik_failure and self._tj_reference_matrix is not None:
            fallback_matrix = self._tj_reference_matrix.copy()
            fallback_matrix[:3, 3] = target_matrix[:3, 3]
            joints = self._solve_ik_matrix(fallback_matrix)
            if joints is not None:
                if not self._warned_position_only_fallback:
                    logger.warning(
                        "Full DM pose IK failed (%s); falling back to position-only IK with TJ reference orientation",
                        full_pose_failure,
                    )
                    self._warned_position_only_fallback = True
                return joints
            self._last_ik_failure = (
                f"full pose failed: {full_pose_failure}; "
                f"position-only fallback failed: {self._last_ik_failure}"
            )
        self._restore_increment_state(increment_state)
        return None

    def _increment_state_snapshot(self) -> tuple[np.ndarray, np.ndarray, int]:
        return (
            self._dm_increment_accumulated_xyz.copy(),
            self._dm_increment_accumulated_matrix.copy(),
            self._last_increment_pose_id,
        )

    def _restore_increment_state(self, state: tuple[np.ndarray, np.ndarray, int]) -> None:
        xyz, matrix, pose_id = state
        self._dm_increment_accumulated_xyz = xyz.copy()
        self._dm_increment_accumulated_matrix = matrix.copy()
        self._last_increment_pose_id = pose_id

    def _target_matrix_from_dm_matrix(self, dm_matrix: np.ndarray, pose_id: int | None = None) -> np.ndarray:
        if self.config.mapping_mode == "absolute":
            return dm_matrix
        if self._tj_reference_matrix is None:
            raise RuntimeError("TJ reference pose is not initialized")
        if self.config.mapping_mode == "absolute_position":
            return _map_dm_pose_to_tj_pose(
                dm_matrix,
                dm_matrix,
                self._tj_reference_matrix,
                self.config.mapping_mode,
            )
        if self.config.mapping_mode == "position_increment":
            step_xyz = np.zeros(3, dtype=np.float64)
            if pose_id is None or pose_id != self._last_increment_pose_id:
                step_xyz = self._filtered_increment_translation(dm_matrix[:3, 3])
                self._dm_increment_accumulated_xyz += step_xyz
                self._dm_increment_accumulated_xyz = self._clip_accumulated_translation(
                    self._dm_increment_accumulated_xyz
                )
                if pose_id is not None:
                    self._last_increment_pose_id = pose_id
            target = self._tj_reference_matrix.copy()
            target[:3, 3] = self._tj_reference_matrix[:3, 3] + self._dm_increment_accumulated_xyz
            now = time.monotonic()
            if now - self._last_target_log_s >= 0.5:
                logger.info(
                    "DM step xyz(mm)=%s accumulated xyz(mm)=%s -> TJ target xyz(mm)=%s",
                    [round(float(v), 3) for v in step_xyz],
                    [round(float(v), 3) for v in self._dm_increment_accumulated_xyz],
                    [round(float(v), 3) for v in target[:3, 3]],
                )
                self._last_target_log_s = now
            return target
        if self.config.mapping_mode == "pose_increment":
            if pose_id is None or pose_id != self._last_increment_pose_id:
                step = np.eye(4, dtype=np.float64)
                step[:3, :3] = self._filtered_increment_rotation(dm_matrix[:3, :3])
                step[:3, 3] = self._filtered_increment_translation(dm_matrix[:3, 3])
                self._dm_increment_accumulated_matrix[:3, 3] += step[:3, 3]
                self._dm_increment_accumulated_matrix[:3, 3] = self._clip_accumulated_translation(
                    self._dm_increment_accumulated_matrix[:3, 3]
                )
                self._dm_increment_accumulated_matrix[:3, :3] = (
                    self._dm_increment_accumulated_matrix[:3, :3] @ step[:3, :3]
                )
                if pose_id is not None:
                    self._last_increment_pose_id = pose_id
            target = self._tj_reference_matrix.copy()
            target[:3, 3] = self._tj_reference_matrix[:3, 3] + self._dm_increment_accumulated_matrix[:3, 3]
            target[:3, :3] = self._tj_reference_matrix[:3, :3] @ self._dm_increment_accumulated_matrix[:3, :3]
            now = time.monotonic()
            if now - self._last_target_log_s >= 0.5:
                logger.info(
                    "DM pose step xyz(mm)=%s accumulated xyz(mm)=%s -> TJ target xyz(mm)=%s",
                    [round(float(v), 3) for v in dm_matrix[:3, 3]],
                    [round(float(v), 3) for v in self._dm_increment_accumulated_matrix[:3, 3]],
                    [round(float(v), 3) for v in target[:3, 3]],
                )
                self._last_target_log_s = now
            return target
        if self._dm_reference_matrix is None:
            self._dm_reference_matrix = dm_matrix.copy()
            logger.info(
                "Captured DM reference pose from first message on %s xyz(mm)=%s; "
                "TJ reference xyz(mm)=%s",
                self.config.pose_topic,
                [round(float(v), 3) for v in self._dm_reference_matrix[:3, 3]],
                [round(float(v), 3) for v in self._tj_reference_matrix[:3, 3]],
            )
            return self._tj_reference_matrix.copy()
        target = _map_dm_pose_to_tj_pose(
            dm_matrix,
            self._dm_reference_matrix,
            self._tj_reference_matrix,
            self.config.mapping_mode,
        )
        now = time.monotonic()
        if now - self._last_target_log_s >= 0.5:
            logger.info(
                "DM delta xyz(mm)=%s -> TJ target xyz(mm)=%s",
                [round(float(v), 3) for v in (dm_matrix[:3, 3] - self._dm_reference_matrix[:3, 3])],
                [round(float(v), 3) for v in target[:3, 3]],
            )
            self._last_target_log_s = now
        return target

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        pose = self._latest_pose_or_none()
        if pose is None:
            return self._halt_action_or_reference()
        position, quat_xyzw, pose_id = pose
        joints = self._solve_ik(position, quat_xyzw, pose_id)
        if joints is None:
            if self.config.use_clutch and not self._clutch:
                return self._halt_action_or_reference()
            if self.config.hold_last_action_on_ik_failure:
                logger.warning("TJ IK failed (%s); halting at current robot feedback", self._last_ik_failure)
                if self.config.reanchor_on_ik_failure:
                    self._anchor_motion_from_current_robot(skip_pose_id=pose_id)
                    with self._lock:
                        self._latest_pose = None
                return self._halt_action_or_reference()
            raise RuntimeError(f"TJ IK failed for the latest DM-EXton2 pose: {self._last_ik_failure}")
        action = {f"{joint}.pos": value for joint, value in zip(self.config.joint_names, joints, strict=True)}
        if self.config.enable_gripper:
            with self._lock:
                action[f"{self.config.gripper_name}.pos"] = float(self._latest_gripper)
        self._last_action = action
        return action

    def _filtered_increment_translation(self, xyz_mm: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_mm, dtype=np.float64)
        norm = float(np.linalg.norm(xyz))
        if norm < self.config.position_deadband_mm:
            return np.zeros(3, dtype=np.float64)
        if norm > self.config.max_position_step_mm:
            xyz = xyz * (self.config.max_position_step_mm / norm)
        return xyz

    def _clip_accumulated_translation(self, xyz_mm: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz_mm, dtype=np.float64)
        norm = float(np.linalg.norm(xyz))
        if norm > self.config.max_accumulated_position_mm:
            return xyz * (self.config.max_accumulated_position_mm / norm)
        return xyz

    def _filtered_increment_rotation(self, rotation: np.ndarray) -> np.ndarray:
        angle = _rotation_angle_rad(rotation)
        if angle < self.config.rotation_deadband_rad:
            return np.eye(3, dtype=np.float64)
        return _clip_rotation_step(rotation, self.config.max_rotation_step_rad)

    def _publish_state_pose(self, matrix: np.ndarray) -> None:
        if self.state_publisher is None or self.node is None or self.pose_stamped_cls is None:
            return

        msg = self.pose_stamped_cls()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        position = matrix[:3, 3] * self.config.state_position_scale
        quat = _matrix_to_quat_xyzw(matrix[:3, :3])
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        self.state_publisher.publish(msg)

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if self.kine is None:
            return None
        try:
            joints = [float(feedback[f"{joint}.pos"]) for joint in self.config.joint_names]
        except KeyError:
            return None
        self._current_joint_action = {
            f"{joint}.pos": value for joint, value in zip(self.config.joint_names, joints, strict=True)
        }
        fk = self.kine.fk(joints)
        if not fk:
            return None
        matrix = np.asarray(fk, dtype=np.float64)
        if matrix.shape != (4, 4):
            return None

        self._current_tj_matrix = matrix
        self._publish_state_pose(matrix)

        if not self._tj_reference_from_feedback:
            self._last_ik_joints = joints
            self._last_action = {
                f"{joint}.pos": value for joint, value in zip(self.config.joint_names, joints, strict=True)
            }
            self._tj_reference_matrix = matrix
            self._tj_reference_from_feedback = True
            logger.info(
                "Initialized TJ reference pose from current robot joints xyz(mm)=%s",
                [round(float(v), 3) for v in matrix[:3, 3]],
            )
        return None

    @check_if_not_connected
    def disconnect(self) -> None:
        self._stop_event.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
            self._spin_thread = None
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        self.subscription = None
        self.clutch_subscription = None
        self.gripper_subscription = None
        self.state_publisher = None
        self.kine = None
        self.fx_kine = None
