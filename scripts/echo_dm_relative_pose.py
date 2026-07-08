#!/usr/bin/env python

import argparse
import math
import time
from pathlib import Path

import numpy as np


def quat_angle_rad(qx: float, qy: float, qz: float, qw: float) -> float:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        return 0.0
    qw = max(-1.0, min(1.0, qw / norm))
    return 2.0 * math.acos(abs(qw))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print current TJ end-effector pose or DMleader relative increments.")
    parser.add_argument(
        "--source",
        choices=["direct_tj", "robot_state", "relative"],
        default="direct_tj",
        help=(
            "direct_tj connects to the Tianji arm and runs FK from feedback joints; "
            "robot_state reads /target_robot/right_ee/state; relative reads DMleader increment commands."
        ),
    )
    parser.add_argument("--topic", default=None)
    parser.add_argument("--ip", default="192.168.1.190")
    parser.add_argument("--arm", choices=["A", "B"], default="B")
    parser.add_argument("--sdk-python-dir", type=Path, default=None)
    parser.add_argument("--kine-config-path", type=Path, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--stale-timeout", type=float, default=0.5, help="Mark data stale after this many seconds.")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit.")
    args = parser.parse_args()

    if args.source == "direct_tj":
        print_direct_tj_pose(args)
        return

    if args.topic is None:
        args.topic = "/target_robot/right_ee/state" if args.source == "robot_state" else "/custom/right_end_increment"
    if args.scale is None:
        args.scale = 1_000_000.0 if args.source == "robot_state" else 30000.0

    import rclpy
    from geometry_msgs.msg import PoseStamped

    rclpy.init()
    node = rclpy.create_node("echo_tj_ee_pose")
    latest: dict[str, object] = {}

    def callback(msg: PoseStamped) -> None:
        latest["msg"] = msg
        latest["time"] = time.monotonic()

    node.create_subscription(PoseStamped, args.topic, callback, 10)

    period = 1.0 / args.print_hz if args.print_hz > 0 else 0.5
    print(f"Listening on {args.topic}")
    if args.source == "robot_state":
        print("Showing current TJ end-effector pose. raw_xyz is meters; scaled_xyz is raw * 1000000.")
    else:
        print("Showing DMleader relative increments. scaled_mm is raw * teleop scale.")

    try:
        while rclpy.ok():
            end_time = time.monotonic() + period
            while rclpy.ok() and time.monotonic() < end_time:
                rclpy.spin_once(node, timeout_sec=0.02)

            msg = latest.get("msg")
            msg_time = latest.get("time")
            if msg is None or msg_time is None:
                print("waiting for message...")
                continue

            pos = msg.pose.position
            ori = msg.pose.orientation
            raw = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
            scaled_mm = raw * args.scale
            age_ms = (time.monotonic() - float(msg_time)) * 1000.0
            stale = age_ms > args.stale_timeout * 1000.0
            angle_deg = math.degrees(quat_angle_rad(ori.x, ori.y, ori.z, ori.w))
            prefix = "STALE " if stale else "LIVE  "
            if args.source == "robot_state":
                print(
                    f"{prefix}xyz_m="
                    f"[{raw[0]: .7f}, {raw[1]: .7f}, {raw[2]: .7f}]  "
                    "scaled_xyz="
                    f"[{scaled_mm[0]: .3f}, {scaled_mm[1]: .3f}, {scaled_mm[2]: .3f}]  "
                    "quat_xyzw="
                    f"[{ori.x: .6f}, {ori.y: .6f}, {ori.z: .6f}, {ori.w: .6f}]  "
                    f"age={age_ms:.0f}ms"
                )
            if args.once:
                break
            else:
                print(
                    f"{prefix}raw_xyz="
                    f"[{raw[0]: .7f}, {raw[1]: .7f}, {raw[2]: .7f}]  "
                    "scaled_mm="
                    f"[{scaled_mm[0]: .3f}, {scaled_mm[1]: .3f}, {scaled_mm[2]: .3f}]  "
                    f"rot_delta={angle_deg:.4f}deg  age={age_ms:.0f}ms"
                )
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def init_tj_kine(arm: str, sdk_python_dir: Path | None, kine_config_path: Path | None):
    from lerobot.teleoperators.dm_exton_tj_ik.dm_exton_tj_ik import (
        _default_kine_config_path,
        _default_sdk_python_dir,
        _load_fx_kine,
    )

    fx_kine = _load_fx_kine(sdk_python_dir or _default_sdk_python_dir())
    kine = fx_kine.Marvin_Kine()
    kine.log_switch(0)
    loaded = kine.load_config(0 if arm == "A" else 1, str(kine_config_path or _default_kine_config_path()))
    if not loaded:
        raise RuntimeError("Failed to load TJ kinematics config")
    arm_index = 0 if arm == "A" else 1
    ok = kine.initial_kine(
        int(loaded["TYPE"][arm_index]),
        loaded["DH"][arm_index],
        loaded["PNVA"][arm_index],
        loaded["BD"][arm_index],
    )
    if not ok:
        raise RuntimeError("Failed to initialize TJ kinematics")
    return kine


def print_direct_tj_pose(args: argparse.Namespace) -> None:
    from lerobot.robots.tj import TJRobot, TJRobotConfig
    from lerobot.teleoperators.dm_exton_tj_ik.dm_exton_tj_ik import _matrix_to_quat_xyzw

    robot = TJRobot(
        TJRobotConfig(
            ip=args.ip,
            arm=args.arm,
            sdk_python_dir=args.sdk_python_dir,
            disable_on_disconnect=False,
            max_relative_target=None,
        )
    )
    kine = init_tj_kine(args.arm, args.sdk_python_dir, args.kine_config_path)
    period = 1.0 / args.print_hz if args.print_hz > 0 else 0.5

    print(f"Connecting to TJ arm {args.arm} at {args.ip}")
    robot.connect()
    print("Showing current TJ end-effector pose from feedback joints -> TJ FK.")
    print("xyz_mm is the Tianji FK/IK unit; xyz_m is xyz_mm * 0.001.")
    try:
        while True:
            obs = robot.get_observation()
            joints = [float(obs[f"{joint}.pos"]) for joint in robot.config.joint_names]
            fk = kine.fk(joints)
            if not fk:
                print("FK failed")
                time.sleep(period)
                continue
            matrix = np.asarray(fk, dtype=np.float64)
            xyz_mm = matrix[:3, 3]
            quat = _matrix_to_quat_xyzw(matrix[:3, :3])
            print(
                "q_deg="
                f"[{', '.join(f'{value: .3f}' for value in joints)}]  "
                "xyz_mm="
                f"[{xyz_mm[0]: .3f}, {xyz_mm[1]: .3f}, {xyz_mm[2]: .3f}]  "
                "xyz_m="
                f"[{xyz_mm[0] * 0.001: .6f}, {xyz_mm[1] * 0.001: .6f}, {xyz_mm[2] * 0.001: .6f}]  "
                "quat_xyzw="
                f"[{quat[0]: .6f}, {quat[1]: .6f}, {quat[2]: .6f}, {quat[3]: .6f}]"
            )
            if args.once:
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
