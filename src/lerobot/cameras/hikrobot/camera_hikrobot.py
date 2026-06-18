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

"""Hikrobot/Hikvision MVS camera backend."""

import importlib
import logging
import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof
from pathlib import Path
from threading import Event, Lock, Thread
from types import ModuleType
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_hikrobot import HikrobotCameraConfig

logger = logging.getLogger(__name__)

DEFAULT_SDK_PATH = Path("/opt/MVS/Samples/64/Python/MvImport")


def _load_mvs_sdk(sdk_path: str | Path = DEFAULT_SDK_PATH) -> ModuleType:
    sdk_path = Path(sdk_path)
    _configure_mvs_environment(sdk_path)
    if str(sdk_path) not in sys.path:
        sys.path.append(str(sdk_path))

    try:
        return importlib.import_module("MvCameraControl_class")
    except ImportError as e:
        raise ImportError(
            "Failed to import Hikrobot MVS Python bindings. Install the Hikrobot MVS SDK and make sure "
            f"`MvCameraControl_class.py` is available under {sdk_path}."
        ) from e


def _configure_mvs_environment(sdk_path: Path) -> None:
    try:
        sdk_root = sdk_path.parents[3]
    except IndexError:
        return

    lib_root = sdk_root / "lib"
    lib64 = lib_root / "64"
    cl_protocol = lib_root / "CLProtocol"

    os.environ.setdefault("MVCAM_SDK_PATH", str(sdk_root))
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(lib_root))
    os.environ.setdefault("MVCAM_SOFTWARE_LIBENV", str(lib_root))
    os.environ.setdefault("MVCAM_GENICAM_CLPROTOCOL", str(cl_protocol))
    os.environ.setdefault("ALLUSERSPROFILE", str(sdk_root / "MVFG"))

    if lib64.is_dir():
        ld_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        if str(lib64) not in ld_paths:
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib64}:{os.environ['LD_LIBRARY_PATH']}" if os.environ.get("LD_LIBRARY_PATH") else str(lib64)
            )


def _decode_c_string(values: Any) -> str:
    chars: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            if value == b"\x00":
                break
            chars.append(value.decode(errors="ignore"))
        else:
            if value == 0:
                break
            chars.append(chr(value))
    return "".join(chars)


def _sdk_layer_type(mvs: ModuleType) -> int:
    layer_type = (
        mvs.MV_GIGE_DEVICE
        | mvs.MV_USB_DEVICE
        | mvs.MV_GENTL_CAMERALINK_DEVICE
        | mvs.MV_GENTL_CXP_DEVICE
        | mvs.MV_GENTL_XOF_DEVICE
    )
    if hasattr(mvs, "MV_GENTL_GIGE_DEVICE"):
        layer_type |= mvs.MV_GENTL_GIGE_DEVICE
    return int(layer_type)


def _sysfs_hikrobot_cameras() -> list[dict[str, Any]]:
    cameras = []
    for device_dir in Path("/sys/bus/usb/devices").glob("*"):
        try:
            if (device_dir / "idVendor").read_text().strip().lower() != "2bdf":
                continue
        except OSError:
            continue

        def read_field(name: str) -> str | None:
            try:
                return (device_dir / name).read_text().strip()
            except OSError:
                return None

        serial = read_field("serial")
        cameras.append(
            {
                "name": read_field("product") or "Hikrobot camera",
                "type": "Hikrobot",
                "id": serial or str(device_dir),
                "serial_number": serial,
                "manufacturer": read_field("manufacturer"),
                "usb_path": str(device_dir),
                "busnum": read_field("busnum"),
                "devnum": read_field("devnum"),
                "usb_speed_mbps": read_field("speed"),
                "sdk_available": False,
            }
        )
    return cameras


class HikrobotCamera(Camera):
    """Camera implementation for Hikrobot/Hikvision industrial cameras through MVS SDK."""

    def __init__(self, config: HikrobotCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number = config.serial_number
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        self.mvs: ModuleType | None = None
        self.cam: Any | None = None
        self.device_info: Any | None = None
        self.is_grabbing = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self.capture_width: int | None = None
        self.capture_height: int | None = None
        self.output_width = self.width
        self.output_height = self.height

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        return self.cam is not None and self.is_grabbing

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        try:
            mvs = _load_mvs_sdk()
        except ImportError:
            return _sysfs_hikrobot_cameras()

        if hasattr(mvs.MvCamera, "MV_CC_Initialize"):
            mvs.MvCamera.MV_CC_Initialize()

        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        ret = mvs.MvCamera.MV_CC_EnumDevices(_sdk_layer_type(mvs), device_list)
        if ret != 0:
            raise RuntimeError(f"Hikrobot MVS device enumeration failed: ret[0x{ret:x}]")

        cameras = []
        for index in range(device_list.nDeviceNum):
            dev_info = cast(device_list.pDeviceInfo[index], POINTER(mvs.MV_CC_DEVICE_INFO)).contents
            info: dict[str, Any] = {"type": "Hikrobot", "sdk_available": True, "index": index}

            if dev_info.nTLayerType == mvs.MV_USB_DEVICE:
                usb_info = dev_info.SpecialInfo.stUsb3VInfo
                serial = _decode_c_string(usb_info.chSerialNumber)
                model = _decode_c_string(usb_info.chModelName)
                info.update(
                    {
                        "name": model or f"Hikrobot USB camera {index}",
                        "id": serial,
                        "serial_number": serial,
                        "transport": "USB",
                    }
                )
            elif dev_info.nTLayerType in (
                mvs.MV_GIGE_DEVICE,
                getattr(mvs, "MV_GENTL_GIGE_DEVICE", mvs.MV_GIGE_DEVICE),
            ):
                gige_info = dev_info.SpecialInfo.stGigEInfo
                model = _decode_c_string(gige_info.chModelName)
                ip = gige_info.nCurrentIp
                serial = _decode_c_string(gige_info.chSerialNumber)
                info.update(
                    {
                        "name": model or f"Hikrobot GigE camera {index}",
                        "id": serial or f"{(ip & 0xFF000000) >> 24}.{(ip & 0x00FF0000) >> 16}.{(ip & 0x0000FF00) >> 8}.{ip & 0x000000FF}",
                        "serial_number": serial,
                        "transport": "GigE",
                        "current_ip": f"{(ip & 0xFF000000) >> 24}.{(ip & 0x00FF0000) >> 16}.{(ip & 0x0000FF00) >> 8}.{ip & 0x000000FF}",
                    }
                )
            else:
                info.update({"name": f"Hikrobot camera {index}", "id": str(index), "transport": "Other"})

            cameras.append(info)

        return cameras

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self.mvs = _load_mvs_sdk(self.config.sdk_path)

        if hasattr(self.mvs.MvCamera, "MV_CC_Initialize"):
            self.mvs.MvCamera.MV_CC_Initialize()

        device_list = self._enumerate_devices()
        index = self._find_device_index(device_list)
        self.device_info = cast(device_list.pDeviceInfo[index], POINTER(self.mvs.MV_CC_DEVICE_INFO)).contents
        self.cam = self.mvs.MvCamera()

        try:
            self._check_ret(self.cam.MV_CC_CreateHandle(self.device_info), "create handle")
            self._check_ret(self.cam.MV_CC_OpenDevice(self.mvs.MV_ACCESS_Exclusive, 0), "open device")
            self._configure_device()
            self._check_ret(self.cam.MV_CC_StartGrabbing(), "start grabbing")
            self.is_grabbing = True
            self._start_read_thread()

            if warmup and self.warmup_s > 0:
                start_time = time.time()
                while time.time() - start_time < self.warmup_s:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                    time.sleep(0.1)
                with self.frame_lock:
                    if self.latest_frame is None:
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")
        except Exception:
            self._cleanup_device()
            raise

        logger.info(f"{self} connected.")

    def _enumerate_devices(self) -> Any:
        if self.mvs is None:
            raise RuntimeError(f"{self} MVS SDK is not loaded.")

        device_list = self.mvs.MV_CC_DEVICE_INFO_LIST()
        ret = self.mvs.MvCamera.MV_CC_EnumDevices(_sdk_layer_type(self.mvs), device_list)
        self._check_ret(ret, "enumerate devices")

        if device_list.nDeviceNum == 0:
            raise ConnectionError("No Hikrobot MVS cameras found.")
        return device_list

    def _find_device_index(self, device_list: Any) -> int:
        if self.mvs is None:
            raise RuntimeError(f"{self} MVS SDK is not loaded.")

        available_serials = []
        for index in range(device_list.nDeviceNum):
            dev_info = cast(device_list.pDeviceInfo[index], POINTER(self.mvs.MV_CC_DEVICE_INFO)).contents
            serial = self._serial_from_device_info(dev_info)
            available_serials.append(serial)
            if serial == self.serial_number:
                return index

        raise ValueError(
            f"No Hikrobot camera found with serial_number={self.serial_number!r}. "
            f"Available serial numbers: {available_serials}"
        )

    def _serial_from_device_info(self, dev_info: Any) -> str:
        if self.mvs is None:
            raise RuntimeError(f"{self} MVS SDK is not loaded.")

        if dev_info.nTLayerType == self.mvs.MV_USB_DEVICE:
            return _decode_c_string(dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        if dev_info.nTLayerType in (
            self.mvs.MV_GIGE_DEVICE,
            getattr(self.mvs, "MV_GENTL_GIGE_DEVICE", self.mvs.MV_GIGE_DEVICE),
        ):
            return _decode_c_string(dev_info.SpecialInfo.stGigEInfo.chSerialNumber)
        return ""

    def _configure_device(self) -> None:
        if self.cam is None or self.mvs is None:
            raise DeviceNotConnectedError(f"{self} camera handle is not initialized.")

        self._check_ret(self.cam.MV_CC_SetEnumValue("TriggerMode", self.mvs.MV_TRIGGER_MODE_OFF), "set trigger mode")

        self._warn_ret(self.cam.MV_CC_SetIntValue("OffsetX", 0), "set offset x")
        self._warn_ret(self.cam.MV_CC_SetIntValue("OffsetY", 0), "set offset y")

        _width_cur, width_min, width_max, width_inc = self._get_int_info("Width")
        _height_cur, height_min, height_max, height_inc = self._get_int_info("Height")

        if self.output_width is None or self.output_height is None:
            self.capture_width = width_max
            self.capture_height = height_max
        elif self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            self.capture_width = self._align_int_to_increment(
                int(self.output_height), width_min, width_max, width_inc
            )
            self.capture_height = self._align_int_to_increment(
                int(self.output_width), height_min, height_max, height_inc
            )
        else:
            self.capture_width = self._align_int_to_increment(int(self.output_width), width_min, width_max, width_inc)
            self.capture_height = self._align_int_to_increment(
                int(self.output_height), height_min, height_max, height_inc
            )

        self._warn_ret(self.cam.MV_CC_SetIntValue("Width", int(self.capture_width)), "set width")
        self._warn_ret(self.cam.MV_CC_SetIntValue("Height", int(self.capture_height)), "set height")

        if self.output_width is None or self.output_height is None:
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = self.capture_height, self.capture_width
            else:
                self.width, self.height = self.capture_width, self.capture_height
            self.output_width, self.output_height = self.width, self.height

        _offset_x_cur, offset_x_min, offset_x_max, offset_x_inc = self._get_int_info("OffsetX")
        _offset_y_cur, offset_y_min, offset_y_max, offset_y_inc = self._get_int_info("OffsetY")
        offset_x = self._align_int_to_increment(
            offset_x_min + (offset_x_max - offset_x_min) // 2,
            offset_x_min,
            offset_x_max,
            offset_x_inc,
        )
        offset_y = self._align_int_to_increment(
            offset_y_min + (offset_y_max - offset_y_min) // 2,
            offset_y_min,
            offset_y_max,
            offset_y_inc,
        )
        self._warn_ret(self.cam.MV_CC_SetIntValue("OffsetX", offset_x), "set centered offset x")
        self._warn_ret(self.cam.MV_CC_SetIntValue("OffsetY", offset_y), "set centered offset y")

        if self.fps is not None:
            self._warn_ret(self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True), "enable fps control")
            self._warn_ret(self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.fps)), "set fps")

        if self.config.exposure_time_us is not None:
            self._warn_ret(self.cam.MV_CC_SetEnumValue("ExposureAuto", self.mvs.MV_EXPOSURE_AUTO_MODE_OFF), "disable auto exposure")
            self._warn_ret(
                self.cam.MV_CC_SetFloatValue("ExposureTime", float(self.config.exposure_time_us)),
                "set exposure time",
            )

        if self.config.gain_auto is not None:
            mode = self.mvs.MV_GAIN_MODE_CONTINUOUS if self.config.gain_auto else self.mvs.MV_GAIN_MODE_OFF
            self._warn_ret(self.cam.MV_CC_SetEnumValue("GainAuto", mode), "set gain auto")

        if self.config.white_balance_auto is not None:
            mode = (
                self.mvs.MV_BALANCEWHITE_AUTO_CONTINUOUS
                if self.config.white_balance_auto
                else self.mvs.MV_BALANCEWHITE_AUTO_OFF
            )
            self._warn_ret(self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", mode), "set white balance auto")

    def _get_int_value(self, key: str) -> int:
        if self.cam is None or self.mvs is None:
            raise DeviceNotConnectedError(f"{self} camera handle is not initialized.")

        return self._get_int_info(key)[0]

    def _get_int_info(self, key: str) -> tuple[int, int, int, int]:
        if self.cam is None or self.mvs is None:
            raise DeviceNotConnectedError(f"{self} camera handle is not initialized.")

        value = self.mvs.MVCC_INTVALUE()
        memset(byref(value), 0, sizeof(self.mvs.MVCC_INTVALUE))
        self._check_ret(self.cam.MV_CC_GetIntValue(key, value), f"get {key}")
        return int(value.nCurValue), int(value.nMin), int(value.nMax), int(value.nInc)

    @staticmethod
    def _align_int_to_increment(value: int, min_value: int, max_value: int, increment: int) -> int:
        value = int(np.clip(value, min_value, max_value))
        if increment <= 1:
            return value
        return min_value + ((value - min_value) // increment) * increment

    def _read_from_hardware(self) -> NDArray[Any]:
        if self.cam is None or self.mvs is None:
            raise DeviceNotConnectedError(f"{self} camera handle is not initialized.")

        out_frame = self.mvs.MV_FRAME_OUT()
        memset(byref(out_frame), 0, sizeof(out_frame))
        ret = self.cam.MV_CC_GetImageBuffer(out_frame, 1000)
        self._check_ret(ret, "get image buffer")

        try:
            width = int(out_frame.stFrameInfo.nWidth)
            height = int(out_frame.stFrameInfo.nHeight)
            rgb_size = width * height * 3

            convert_param = self.mvs.MV_CC_PIXEL_CONVERT_PARAM_EX()
            memset(byref(convert_param), 0, sizeof(convert_param))
            convert_param.nWidth = width
            convert_param.nHeight = height
            convert_param.pSrcData = out_frame.pBufAddr
            convert_param.nSrcDataLen = out_frame.stFrameInfo.nFrameLen
            convert_param.enSrcPixelType = out_frame.stFrameInfo.enPixelType
            convert_param.enDstPixelType = self.mvs.PixelType_Gvsp_BGR8_Packed

            dst_buffer = (c_ubyte * rgb_size)()
            convert_param.pDstBuffer = dst_buffer
            convert_param.nDstBufferSize = rgb_size

            self._check_ret(self.cam.MV_CC_ConvertPixelTypeEx(convert_param), "convert pixel type")
            frame = np.ctypeslib.as_array(dst_buffer, shape=(rgb_size,))
            return frame.reshape(height, width, 3).copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(out_frame)

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels.")

        if self.capture_width is not None and self.capture_height is not None:
            if h != self.capture_height or w != self.capture_width:
                raise RuntimeError(
                    f"{self} frame width={w} or height={h} do not match configured "
                    f"width={self.capture_width} or height={self.capture_height}."
                )

        processed_image = image
        if self.color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        if self.output_width is not None and self.output_height is not None:
            out_w, out_h = int(self.output_width), int(self.output_height)
            if processed_image.shape[1] != out_w or processed_image.shape[0] != out_h:
                processed_image = cv2.resize(processed_image, (out_w, out_h), interpolation=cv2.INTER_AREA)

        return processed_image

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    def _read_loop(self) -> None:
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._postprocess_image(self._read_from_hardware())
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms.")

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None and self.cam is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self._cleanup_device()
        logger.info(f"{self} disconnected.")

    def _cleanup_device(self) -> None:
        if self.thread is not None:
            self._stop_read_thread()

        if self.cam is not None:
            if self.is_grabbing:
                self._warn_ret(self.cam.MV_CC_StopGrabbing(), "stop grabbing")
            self.is_grabbing = False
            self._warn_ret(self.cam.MV_CC_CloseDevice(), "close device")
            self._warn_ret(self.cam.MV_CC_DestroyHandle(), "destroy handle")
            self.cam = None

        if self.mvs is not None and hasattr(self.mvs.MvCamera, "MV_CC_Finalize"):
            self._warn_ret(self.mvs.MvCamera.MV_CC_Finalize(), "finalize MVS SDK")

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def _check_ret(self, ret: int | None, action: str) -> None:
        if ret is None:
            return
        if ret != 0:
            raise RuntimeError(f"{self} failed to {action}: ret[0x{ret:x}]")

    def _warn_ret(self, ret: int | None, action: str) -> None:
        if ret is None:
            return
        if ret != 0:
            logger.warning(f"{self} failed to {action}: ret[0x{ret:x}]")
