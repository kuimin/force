#!/usr/bin/env python

import numpy as np

from lerobot.teleoperators.dm_exton_tj_ik import DMExtonTJIKTeleop, DMExtonTJIKTeleopConfig
from lerobot.teleoperators.dm_exton_tj_ik.dm_exton_tj_ik import (
    _change_increment_frame,
    _map_dm_pose_to_tj_pose,
    _master_to_tj_axes_rotation,
    _pose_to_tj_matrix_1x16,
)
from lerobot.teleoperators.utils import make_teleoperator_from_config


def test_dm_exton_tj_ik_config_defaults_pose_topic_from_arm():
    assert DMExtonTJIKTeleopConfig().pose_topic == "/target_robot/right_ee/target"
    assert DMExtonTJIKTeleopConfig(pose_topic=None, arm="A").pose_topic == "/target_robot/left_ee/target"
    assert DMExtonTJIKTeleopConfig(pose_topic=None, arm="B").pose_topic == "/target_robot/right_ee/target"
    assert DMExtonTJIKTeleopConfig(pose_topic=None, arm="A").state_topic == "/target_robot/left_ee/state"
    assert DMExtonTJIKTeleopConfig(pose_topic=None, arm="B").state_topic == "/target_robot/right_ee/state"
    assert DMExtonTJIKTeleopConfig(pose_topic=None, arm="A", use_clutch=True).clutch_topic == "/clutch/left"
    assert DMExtonTJIKTeleopConfig(pose_topic=None, arm="B", use_clutch=True).clutch_topic == "/clutch/right"


def test_dm_exton_tj_ik_config_defaults_to_float_array_start():
    config = DMExtonTJIKTeleopConfig()

    assert config.pose_message_type == "pose_stamped"
    assert config.pose_array_start_index == 7
    assert config.mapping_mode == "absolute"
    assert config.position_offset_mm == [0.0, 0.0, 0.0]
    assert config.position_scale == 1000.0
    assert config.state_position_scale == 0.001
    assert config.publish_state is True
    assert config.use_clutch is True
    assert config.position_deadband_mm == 3.0
    assert config.rotation_deadband_rad == 0.003
    assert config.max_position_step_mm == 10000.0
    assert config.incoming_command_timeout_s == 0.25
    assert config.max_accumulated_position_mm == 120.0
    assert config.reanchor_on_ik_failure is False
    assert config.fallback_to_position_only_on_ik_failure is False
    assert config.align_master_axes is True
    assert config.master_align_z_deg == 90.0
    assert config.master_align_x_deg == -90.0


def test_dm_exton_tj_ik_factory_creates_teleop():
    teleop = make_teleoperator_from_config(DMExtonTJIKTeleopConfig(arm="A"))

    assert isinstance(teleop, DMExtonTJIKTeleop)
    assert teleop.name == "DM"


def test_pose_to_tj_matrix_converts_meters_to_millimeters():
    matrix = _pose_to_tj_matrix_1x16(
        position_xyz=[0.1, -0.2, 0.3],
        quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        position_scale=1000.0,
        position_offset_mm=[1.0, 2.0, 3.0],
        mirror_xyz=(False, True, False),
    )

    mat = np.asarray(matrix).reshape(4, 4)
    assert np.allclose(mat[:3, :3], np.eye(3))
    assert np.allclose(mat[:3, 3], [101.0, 202.0, 303.0])


def test_relative_mapping_keeps_first_tj_reference_pose():
    dm_ref = np.eye(4)
    dm_now = np.eye(4)
    tj_ref = np.eye(4)
    tj_ref[:3, 3] = [1.0, 95.0, 768.0]

    target = _map_dm_pose_to_tj_pose(dm_now, dm_ref, tj_ref, "relative")

    assert np.allclose(target, tj_ref)


def test_position_relative_mapping_uses_dm_position_delta_only():
    dm_ref = np.eye(4)
    dm_ref[:3, 3] = [10.0, 20.0, 30.0]
    dm_now = np.eye(4)
    dm_now[:3, 3] = [15.0, 18.0, 33.0]
    tj_ref = np.eye(4)
    tj_ref[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    tj_ref[:3, 3] = [100.0, 200.0, 300.0]

    target = _map_dm_pose_to_tj_pose(dm_now, dm_ref, tj_ref, "position_relative")

    assert np.allclose(target[:3, :3], tj_ref[:3, :3])
    assert np.allclose(target[:3, 3], [105.0, 198.0, 303.0])


def test_pose_increment_mapping_uses_position_and_relative_orientation():
    dm_step = np.eye(4)
    dm_step[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    dm_step[:3, 3] = [5.0, -2.0, 3.0]
    tj_ref = np.eye(4)
    tj_ref[:3, :3] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    tj_ref[:3, 3] = [100.0, 200.0, 300.0]

    target = _map_dm_pose_to_tj_pose(dm_step, np.eye(4), tj_ref, "pose_increment")

    assert np.allclose(target[:3, :3], tj_ref[:3, :3] @ dm_step[:3, :3])
    assert np.allclose(target[:3, 3], [105.0, 198.0, 303.0])


def test_master_axis_alignment_rotates_z_then_x_for_increment_frame():
    rotation = _master_to_tj_axes_rotation(90.0, -90.0)
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 0.0, 1.0]

    aligned = _change_increment_frame(matrix, rotation)

    assert np.allclose(aligned[:3, 3], [0.0, 1.0, -1.0], atol=1e-9)


def test_absolute_position_mapping_uses_dm_position_and_tj_orientation():
    dm_ref = np.eye(4)
    dm_now = np.eye(4)
    dm_now[:3, :3] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    dm_now[:3, 3] = [31.0, 49.0, 801.0]
    tj_ref = np.eye(4)
    tj_ref[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    tj_ref[:3, 3] = [692.0, -54.0, 203.0]

    target = _map_dm_pose_to_tj_pose(dm_now, dm_ref, tj_ref, "absolute_position")

    assert np.allclose(target[:3, :3], tj_ref[:3, :3])
    assert np.allclose(target[:3, 3], [31.0, 49.0, 801.0])


def test_clutch_disabled_holds_last_action():
    teleop = DMExtonTJIKTeleop(DMExtonTJIKTeleopConfig(arm="B", use_clutch=True))
    teleop._last_action = {f"joint_{i}.pos": float(i) for i in range(1, 8)}
    teleop._current_joint_action = {f"joint_{i}.pos": float(i + 10) for i in range(1, 8)}
    teleop._clutch = False

    target = np.eye(4)

    assert teleop._apply_clutch_and_soft_start(target) is None
    assert teleop._halt_action_or_reference() == teleop._current_joint_action


def test_absolute_target_pose_is_kept_like_demo_latest_target():
    teleop = DMExtonTJIKTeleop(DMExtonTJIKTeleopConfig(arm="B", incoming_command_timeout_s=0.1))
    teleop._latest_pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1, -1.0)

    assert teleop._latest_pose_or_none() == ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1)


def test_increment_pose_returns_none_after_command_timeout():
    teleop = DMExtonTJIKTeleop(
        DMExtonTJIKTeleopConfig(arm="B", mapping_mode="pose_increment", incoming_command_timeout_s=0.1)
    )
    teleop._latest_pose = ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1, -1.0)

    assert teleop._latest_pose_or_none() is None


def test_increment_deadband_removes_static_noise():
    teleop = DMExtonTJIKTeleop(DMExtonTJIKTeleopConfig(arm="B"))

    assert np.allclose(teleop._filtered_increment_translation(np.array([0.5, 0.5, 0.5])), [0.0, 0.0, 0.0])
    assert np.allclose(teleop._filtered_increment_translation(np.array([30.0, 0.0, 0.0])), [30.0, 0.0, 0.0])
    assert np.allclose(teleop._filtered_increment_translation(np.array([1000.0, 0.0, 0.0])), [500.0, 0.0, 0.0])
    assert np.allclose(teleop._filtered_increment_rotation(np.eye(3)), np.eye(3))
    assert np.allclose(teleop._clip_accumulated_translation(np.array([240.0, 0.0, 0.0])), [120.0, 0.0, 0.0])


def test_increment_state_can_rollback_after_invalid_servo_command():
    teleop = DMExtonTJIKTeleop(DMExtonTJIKTeleopConfig(arm="B"))
    teleop._dm_increment_accumulated_xyz = np.array([1.0, 2.0, 3.0])
    teleop._dm_increment_accumulated_matrix[:3, 3] = [4.0, 5.0, 6.0]
    teleop._last_increment_pose_id = 7

    snapshot = teleop._increment_state_snapshot()
    teleop._dm_increment_accumulated_xyz[:] = [10.0, 20.0, 30.0]
    teleop._dm_increment_accumulated_matrix[:3, 3] = [40.0, 50.0, 60.0]
    teleop._last_increment_pose_id = 8
    teleop._restore_increment_state(snapshot)

    assert np.allclose(teleop._dm_increment_accumulated_xyz, [1.0, 2.0, 3.0])
    assert np.allclose(teleop._dm_increment_accumulated_matrix[:3, 3], [4.0, 5.0, 6.0])
    assert teleop._last_increment_pose_id == 7


def test_clutch_engage_skips_current_increment_sample():
    teleop = DMExtonTJIKTeleop(DMExtonTJIKTeleopConfig(arm="B"))
    teleop._tj_reference_matrix = np.eye(4)
    teleop._current_tj_matrix = np.eye(4)
    teleop._current_tj_matrix[:3, 3] = [100.0, 200.0, 300.0]

    teleop._anchor_motion_from_current_robot(skip_pose_id=12)

    assert teleop._last_increment_pose_id == 12
    assert np.allclose(teleop._tj_reference_matrix[:3, 3], [100.0, 200.0, 300.0])
    assert np.allclose(teleop._dm_increment_accumulated_matrix, np.eye(4))


def test_extract_pose_from_float_array_uses_configured_start_index():
    class Msg:
        data = [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16]

    teleop = DMExtonTJIKTeleop(DMExtonTJIKTeleopConfig(arm="B", pose_array_start_index=7))

    position, quat = teleop._extract_pose_from_msg(Msg())

    assert position == [10.0, 11.0, 12.0]
    assert quat == [13.0, 14.0, 15.0, 16.0]
