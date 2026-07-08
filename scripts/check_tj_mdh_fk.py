#!/usr/bin/env python

import importlib.util
import math
from pathlib import Path

import numpy as np


def rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def tx(offset):
    matrix = np.eye(4)
    matrix[0, 3] = offset
    return matrix


def tz(offset):
    matrix = np.eye(4)
    matrix[2, 3] = offset
    return matrix


def main():
    sdk_file = Path("/home/robot/lerobot/third_party/tj_marvin_sdk/SDK_PYTHON/fx_kine.py")
    spec = importlib.util.spec_from_file_location("fx_kine_check", sdk_file)
    fx_kine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fx_kine)

    kine = fx_kine.Marvin_Kine()
    config = "/home/robot/lerobot/third_party/tj_marvin_sdk/CommonConfig/ccs_m6_40.MvKDCfg"
    loaded = kine.load_config(1, config)
    kine.initial_kine(loaded["TYPE"][1], loaded["DH"][1], loaded["PNVA"][1], loaded["BD"][1])
    dh = loaded["DH"][1]

    samples = [
        [0, 0, 0, 0, 0, 0, 0],
        [-21.8, -41.0, 4.75, -63.67, -10.15, 14.72, -7.68],
    ]
    variants = {
        "mdh": lambda alpha, a, d, theta: rx(alpha) @ tx(a) @ rz(theta) @ tz(d),
        "std": lambda alpha, a, d, theta: rz(theta) @ tz(d) @ tx(a) @ rx(alpha),
    }

    for joints in samples:
        sdk_fk = np.asarray(kine.fk(joints), dtype=float)
        print("q_deg", joints)
        print("sdk_xyz_mm", np.round(sdk_fk[:3, 3], 3).tolist())
        for name, build in variants.items():
            matrix = np.eye(4)
            for index, row in enumerate(dh[:7]):
                alpha_deg, a_mm, d_mm, theta_deg = [float(value) for value in row]
                matrix = matrix @ build(
                    math.radians(alpha_deg),
                    a_mm / 1000.0,
                    d_mm / 1000.0,
                    math.radians(theta_deg + joints[index]),
                )
            alpha_deg, a_mm, d_mm, theta_deg = [float(value) for value in dh[7]]
            matrix = matrix @ build(
                math.radians(alpha_deg),
                a_mm / 1000.0,
                d_mm / 1000.0,
                math.radians(theta_deg),
            )
            print(name, "xyz_mm", np.round(matrix[:3, 3] * 1000.0, 3).tolist())


if __name__ == "__main__":
    main()
