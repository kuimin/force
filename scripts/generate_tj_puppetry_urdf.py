#!/usr/bin/env python

import importlib.util
import math
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_FILE = REPO_ROOT / "third_party/tj_marvin_sdk/SDK_PYTHON/fx_kine.py"
CONFIG_FILE = REPO_ROOT / "third_party/tj_marvin_sdk/CommonConfig/ccs_m6_40.MvKDCfg"
OUTPUT_FILE = REPO_ROOT / "third_party/tj_marvin_sdk/urdf/tj_ccs_m6_40_puppetry.urdf"


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _tx(offset: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 3] = offset
    return matrix


def _tz(offset: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[2, 3] = offset
    return matrix


def _matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    if sy > 1e-9:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def _load_tj_config():
    spec = importlib.util.spec_from_file_location("tj_fx_kine_urdf_gen", SDK_FILE)
    fx_kine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx_kine)

    kine = fx_kine.Marvin_Kine()
    loaded = kine.load_config(1, str(CONFIG_FILE))
    if loaded is None:
        raise RuntimeError(f"Failed to load TJ config: {CONFIG_FILE}")
    return loaded


def _mdh_origin(alpha_deg: float, a_mm: float, d_mm: float, theta_deg: float) -> tuple[np.ndarray, tuple[float, float, float]]:
    # TJ SDK uses modified DH: Rx(alpha) Tx(a) Rz(theta) Tz(d).
    # Tz(d) commutes with Rz(theta), so the fixed URDF origin is
    # Rx(alpha) Tx(a) Tz(d) Rz(theta), followed by a revolute z-axis joint.
    matrix = (
        _rx(math.radians(alpha_deg))
        @ _tx(a_mm / 1000.0)
        @ _tz(d_mm / 1000.0)
        @ _rz(math.radians(theta_deg))
    )
    return matrix[:3, 3], _matrix_to_rpy(matrix[:3, :3])


def _fmt(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _joint_xml(prefix: str, dh: list[list[float]], pnva: list[list[float]]) -> list[str]:
    lines = [f'  <link name="{prefix}_link_0"/>']
    lines.append("")
    lines.append(f'  <joint name="{prefix}_base_joint" type="fixed">')
    lines.append('    <parent link="base_link"/>')
    lines.append(f'    <child link="{prefix}_link_0"/>')
    lines.append('    <origin xyz="0 0 0" rpy="0 0 0"/>')
    lines.append("  </joint>")
    lines.append("")

    for index in range(7):
        child = f"{prefix}_link_{index + 1}"
        xyz, rpy = _mdh_origin(*[float(value) for value in dh[index]])
        limit_a, limit_b, velocity_deg_s, _accel = [float(value) for value in pnva[index]]
        lower = math.radians(min(limit_a, limit_b))
        upper = math.radians(max(limit_a, limit_b))
        velocity = math.radians(abs(velocity_deg_s))
        lines.append(f'  <link name="{child}"/>')
        lines.append("")
        lines.append(f'  <joint name="{prefix}_joint_{index + 1}" type="revolute">')
        lines.append(f'    <parent link="{prefix}_link_{index}"/>')
        lines.append(f'    <child link="{child}"/>')
        lines.append(f'    <origin xyz="{_fmt(xyz)}" rpy="{_fmt(rpy)}"/>')
        lines.append('    <axis xyz="0 0 1"/>')
        lines.append(f'    <limit lower="{lower:.9g}" upper="{upper:.9g}" effort="100" velocity="{velocity:.9g}"/>')
        lines.append("  </joint>")
        lines.append("")

    xyz, rpy = _mdh_origin(*[float(value) for value in dh[7]])
    lines.append(f'  <link name="{prefix}_end_effector"/>')
    lines.append("")
    lines.append(f'  <joint name="{prefix}_tcp_fixed" type="fixed">')
    lines.append(f'    <parent link="{prefix}_link_7"/>')
    lines.append(f'    <child link="{prefix}_end_effector"/>')
    lines.append(f'    <origin xyz="{_fmt(xyz)}" rpy="{_fmt(rpy)}"/>')
    lines.append("  </joint>")
    lines.append("")
    return lines


def main() -> None:
    loaded = _load_tj_config()
    dh = loaded["DH"][1]
    pnva = loaded["PNVA"][1]

    lines = [
        '<?xml version="1.0"?>',
        '<!-- Kinematics-only TJ Marvin CCS M6-40 URDF generated from ccs_m6_40.MvKDCfg. -->',
        '<robot name="tj_marvin_ccs_m6_40">',
        '  <link name="base_link"/>',
        "",
    ]
    lines.extend(_joint_xml("left", dh, pnva))
    lines.extend(_joint_xml("right", dh, pnva))
    lines.append("</robot>")
    lines.append("")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
