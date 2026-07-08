#!/usr/bin/env python

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path


LEFT_URDF = Path(
    "/tmp/marvin_m6_left_urdf/Marvin M6-S-L-CCS-696-V4.0 urdf (FUSION)/urdf/"
    "Marvin M6-S-L-CCS-696-V4.0 urdf.urdf"
)
RIGHT_URDF = Path(
    "/tmp/marvin_m6_right_urdf/Marvin M6-S-R-CCS-696-V4.0 urdf (FUSION)/urdf/"
    "Marvin M6-S-R-CCS-696-V4.0 urdf.urdf"
)
OUTPUT_URDF = Path(
    "/home/robot/lerobot/third_party/tj_marvin_sdk/urdf/"
    "Marvin_M6_S_CCS_696_V4_official_puppetry.urdf"
)


def _copy_minimal_chain(source: Path, robot: ET.Element, side: str) -> None:
    root = ET.parse(source).getroot()
    base_name = f"Base_{side}"
    ET.SubElement(robot, "link", {"name": base_name})
    fixed = ET.SubElement(robot, "joint", {"name": f"{side.lower()}_base_joint", "type": "fixed"})
    ET.SubElement(fixed, "parent", {"link": "base_link"})
    ET.SubElement(fixed, "child", {"link": base_name})
    ET.SubElement(fixed, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})

    for index in range(1, 8):
        link_name = f"Link{index}_{side}"
        ET.SubElement(robot, "link", {"name": link_name})

    for joint in root.findall("joint"):
        joint_copy = ET.Element("joint", joint.attrib)
        for tag in ("origin", "parent", "child", "axis", "limit"):
            element = joint.find(tag)
            if element is not None:
                joint_copy.append(copy.deepcopy(element))
        robot.append(joint_copy)

    ee_name = "left_end_effector" if side == "L" else "right_end_effector"
    ET.SubElement(robot, "link", {"name": ee_name})
    tcp = ET.SubElement(robot, "joint", {"name": f"{side.lower()}_tcp_fixed", "type": "fixed"})
    ET.SubElement(tcp, "parent", {"link": f"Link7_{side}"})
    ET.SubElement(tcp, "child", {"link": ee_name})
    ET.SubElement(tcp, "origin", {"xyz": "0 -0.095 0", "rpy": "1.5708 -1.5708 0"})


def _indent(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    child_indent = "\n" + (level + 1) * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in element:
            _indent(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def main() -> None:
    robot = ET.Element("robot", {"name": "Marvin_M6_S_CCS_696_V4_official_puppetry"})
    ET.SubElement(robot, "link", {"name": "base_link"})
    _copy_minimal_chain(LEFT_URDF, robot, "L")
    _copy_minimal_chain(RIGHT_URDF, robot, "R")
    _indent(robot)

    OUTPUT_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(robot)
    tree.write(OUTPUT_URDF, encoding="utf-8", xml_declaration=True)
    print(OUTPUT_URDF)


if __name__ == "__main__":
    main()
