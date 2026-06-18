# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["HikrobotCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("hikrobot")
@dataclass
class HikrobotCameraConfig(CameraConfig):
    """Configuration for Hikrobot/Hikvision MVS industrial cameras.

    This backend uses Hikrobot's MVS Python bindings. By default the SDK is
    searched at ``/opt/MVS/Samples/64/Python/MvImport``.
    """

    serial_number: str
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    sdk_path: str | Path = "/opt/MVS/Samples/64/Python/MvImport"
    exposure_time_us: float | None = 20000
    gain_auto: bool | None = None
    white_balance_auto: bool | None = None

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.sdk_path = Path(self.sdk_path)

        if not isinstance(self.serial_number, str) or not self.serial_number:
            raise ValueError(f"`serial_number` must be a non-empty string, got {self.serial_number!r}.")

        values = (self.fps, self.width, self.height)
        if any(v is not None for v in values) and any(v is None for v in values):
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them."
            )
