#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from ..config import RobotConfig


@RobotConfig.register_subclass("dmtac")
@dataclass
class DmTacConfig(RobotConfig):
    dev_id: Union[int, str] = 0
    sdk_dir: Path | None = Path("/home/robot/daimeng/DM-Tac-SDK/SDK_Publish_V1.2.13.1")
    backend: str = "cpu"
    mode: str = "high"
    max_fps: int = 120
    show_fps: bool = False
    cpu_fallback: bool = True

    enable_raw: bool = True
    enable_infer: bool = False
    enable_deformation: bool = False
    enable_depth: bool = False
    enable_shear: bool = False
    enable_force: bool = False
    enable_contact_area: bool = False

    image_height: int = 240
    image_width: int = 320
    image_channels: int = 3
    map_height: int = 240
    map_width: int = 320
    force_dim: int = 6

    wait_timeout_ms: int = 500
    remote_addr: str = "192.168.127.10:50051"
    pc_host: str = "0.0.0.0"
    pc_port: int = 60000

    def __post_init__(self):
        super().__post_init__()
        if self.backend.lower() not in {"cpu", "cuda", "flux"}:
            raise ValueError("backend must be 'cpu', 'cuda', or 'Flux'")
        if self.mode.lower() not in {"standard", "high"}:
            raise ValueError("mode must be 'standard' or 'high'")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if self.wait_timeout_ms <= 0:
            raise ValueError("wait_timeout_ms must be positive")
        if self.force_dim <= 0:
            raise ValueError("force_dim must be positive")
        for name, value in {
            "image_height": self.image_height,
            "image_width": self.image_width,
            "image_channels": self.image_channels,
            "map_height": self.map_height,
            "map_width": self.map_width,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
