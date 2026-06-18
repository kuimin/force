#!/usr/bin/env python

"""他山触觉/接触力传感器读取接口。

这个文件直接包含他山 CH341/I2C 采集所需的最小代码。LeRobot 录制脚本每保存一帧会
调用一次 ``EffortSensor.read``，这里读取一片手指上两个触点的 nf/tf/tfDir，并转换为：
``[fx_0, fy_0, fz_0, fx_1, fy_1, fz_1]``。
"""

from __future__ import annotations

import ctypes
import glob
import logging
import math
import os
import time
from ctypes import Structure, c_byte, c_float, c_long, c_uint16, c_uint32, c_uint8
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Sequence


TASHAN_CH341_LIB = Path(__file__).resolve().parent / "lib" / "ch347" / "libch347.so"
TASHAN_PCA_INDEX = 4
TASHAN_PCA_ADDR = 0x70
INVALID_TF_DIR = 65535
DEFAULT_TASHAN_READ_HZ = 30.0

logger = logging.getLogger(__name__)


class DynamicYddsComTs(Structure):
    _pack_ = 1
    _fields_ = [
        ("nf", c_float),
        ("nfCap", c_uint32),
        ("tf", c_float),
        ("tfCap", c_uint32),
        ("tfDir", c_uint16),
        ("prox", c_uint32),
    ]


class DynamicYddsU16Ts(Structure):
    _pack_ = 1
    _fields_ = [
        ("nf", c_uint16),
        ("tf", c_uint16),
        ("tfDir", c_uint16),
    ]


@dataclass(frozen=True)
class FingerParam:
    prg: int
    pack_len: int
    sensor_num: int
    ydds_num: int
    s_prox_num: int
    m_prox_num: int
    cap_byte: int
    ydds_type: int
    name: str


FINGER_PARAMS = {
    2: FingerParam(2, 62, 8, 1, 1, 0, 4, 2, "通用手指"),
    17: FingerParam(17, 78, 16, 2, 2, 1, 3, 4, "两指-大包"),
    27: FingerParam(27, 66, 16, 1, 1, 0, 3, 4, "通用点阵"),
    44: FingerParam(44, 60, 14, 1, 1, 0, 3, 4, "通用点阵"),
    50: FingerParam(50, 36, 6, 1, 1, 0, 3, 4, "通用大拇指"),
    54: FingerParam(54, 36, 6, 1, 1, 0, 3, 4, "通用小拇指"),
    52: FingerParam(52, 66, 12, 2, 2, 1, 3, 4, "通用中指"),
}


class CH341Device:
    _CMD_I2C_STREAM = 0xAA
    _STM_STA = 0x74
    _STM_STO = 0x75
    _STM_OUT = 0x80
    _STM_IN = 0xC0
    _STM_END = 0x00
    _STM_MS = 0x50
    _STM_MAX = 63
    _STATE_BIT_INT = 0x00000400

    IIC_SPEED_400 = 2

    def __init__(self, lib_path: Path = TASHAN_CH341_LIB) -> None:
        if not lib_path.exists():
            raise FileNotFoundError(f"未找到 CH341 动态库: {lib_path}")
        self.lib_path = lib_path
        self.device_id = ctypes.c_uint32()
        self.fd: int | None = None
        self.ic = ctypes.cdll.LoadLibrary(str(lib_path))
        self.ch341_get_input = self.ic.CH34xGetInput
        self.ch341_close_device = self.ic.CH34xCloseDevice
        self.ch341_write_data = self.ic.CH34xWriteData
        self.ch341_write_read = self.ic.CH34xWriteRead
        self.ch341_set_output = self.ic.CH34xSetOutput
        self.ch341_set_stream = self.ic.CH34xSetStream

    def open(self) -> None:
        devices = sorted(glob.glob("/dev/ch34x_pis*"))
        if not devices:
            raise RuntimeError("未找到 /dev/ch34x_pis*，请先确认 CH341 驱动和 udev 设备节点")
        fd = self.ic.CH34xOpenDevice(devices[0].encode())
        if fd == -1:
            raise RuntimeError(f"CH341 打开失败: {devices[0]}")
        self.fd = fd
        self.set_speed(self.IIC_SPEED_400)
        self.set_int(0)
        time.sleep(1.0)
        self.set_int(1)

    def close(self) -> None:
        if self.fd is not None:
            self.ch341_close_device(self.fd)
            self.fd = None

    def _require_fd(self) -> int:
        if self.fd is None:
            raise RuntimeError("CH341 尚未打开")
        return self.fd

    def write(self, addr: int, data: list[int]) -> int:
        fd = self._require_fd()
        tmp_data = list(data)
        total_len = len(tmp_data)
        pack: list[int] = [self._CMD_I2C_STREAM, self._STM_STA, self._STM_OUT | 1, addr << 1]
        chunk = 20
        full_chunks = total_len // chunk
        remain = total_len % chunk

        for _ in range(full_chunks):
            pack.append(self._STM_OUT | chunk)
            pack.extend(tmp_data[:chunk])
            del tmp_data[:chunk]
            pack.append(self._STM_END)
            if not self._write_pack(fd, pack):
                return 0
            pack = [self._CMD_I2C_STREAM]

        if remain >= 1:
            pack.append(self._STM_OUT | remain)
            pack.extend(tmp_data[:remain])
        pack.extend([self._STM_STO, self._STM_END])
        if not self._write_pack(fd, pack):
            return 0
        return total_len

    def _write_pack(self, fd: int, pack: list[int]) -> bool:
        send_buf = (c_byte * len(pack))()
        for i, value in enumerate(pack):
            send_buf[i] = value
        send_len = (c_byte * 1)()
        send_len[0] = len(pack)
        if not self.ch341_write_data(fd, send_buf, send_len):
            return False
        return send_len[0] != 0

    def read(self, addr: int, data: list[int]) -> int:
        fd = self._require_fd()
        if not data:
            return 0
        read_len_target = len(data)
        read_buf: list[int] = []
        pack_count = read_len_target // 30
        remain = read_len_target % 30
        if remain == 0:
            remain = 30
            pack_count -= 1

        pack: list[int] = [
            self._CMD_I2C_STREAM,
            self._STM_STA,
            self._STM_OUT | 1,
            (addr << 1) | 0x01,
            self._STM_MS | 1,
        ]
        for _ in range(pack_count):
            pack.extend([self._STM_IN | 30, self._STM_END])
            chunk = self._write_read_pack(fd, pack)
            if not chunk:
                return 0
            read_buf.extend(chunk)
            pack = [self._CMD_I2C_STREAM]

        if remain > 1:
            pack.append(self._STM_IN | (remain - 1))
        pack.extend([self._STM_IN | 0, self._STM_STO, self._STM_END])
        chunk = self._write_read_pack(fd, pack)
        if not chunk:
            return 0
        read_buf.extend(chunk)
        data.clear()
        data.extend(read_buf)
        return len(read_buf)

    def _write_read_pack(self, fd: int, pack: list[int]) -> list[int]:
        send_buf = (c_byte * len(pack))()
        for i, value in enumerate(pack):
            send_buf[i] = value
        recv_len = (c_byte * 1)()
        recv_buf = (c_byte * self._STM_MAX)()
        ok = self.ch341_write_read(fd, len(pack), send_buf, self._STM_MAX, 1, recv_len, recv_buf)
        if not ok or recv_len[0] == 0:
            return []
        return [recv_buf[i] for i in range(recv_len[0])]

    def set_int(self, level: int) -> None:
        fd = self._require_fd()
        status = (c_long * 1)()
        self.ch341_get_input(fd, status)
        time.sleep(0.01)
        if level:
            value = status[0] | self._STATE_BIT_INT
        else:
            value = status[0] & (~self._STATE_BIT_INT)
        self.ch341_set_output(fd, 0x03, 0xFF00, value)

    def set_speed(self, speed: int) -> None:
        fd = self._require_fd()
        if not self.ch341_set_stream(fd, speed | 0):
            logger.warning("CH341 I2C 速度设置失败，继续使用当前设备速度")


class SensorCommand:
    CMD_GET_SENSOR_CAP_DATA = 0x60
    CMD_GET_SENSOR_IIC_ADDR = 0x71
    CMD_SET_SENSOR_CDC_START_OFFSET = 0x73
    CMD_SET_SENSOR_SEND_TYPE = 0x7F
    CMD_GET_PRG = 0xA6

    def __init__(self, ch341: CH341Device) -> None:
        self.ch341 = ch341

    @staticmethod
    def add_sum(pack: list[int]) -> None:
        checksum = sum(byte & 0xFF for byte in pack)
        pack.append(checksum & 0xFF)
        pack.append((checksum >> 8) & 0xFF)

    @staticmethod
    def check_sum(pack: list[int]) -> bool:
        if len(pack) <= 5:
            return False
        checksum = sum(byte & 0xFF for byte in pack[:-2])
        return (checksum & 0xFF) == (pack[-2] & 0xFF) and ((checksum >> 8) & 0xFF) == (pack[-1] & 0xFF)

    def _command(self, addr: int, command: int, value: int = 0, response_len: int = 11) -> list[int] | None:
        pack = [0xAA, 0x55, 0x03, command, 0x00, 0x00, 0x00, value & 0xFF, 0x00]
        self.add_sum(pack)
        self.ch341.write(addr, pack)
        response = list(range(response_len))
        time.sleep(0.01)
        self.ch341.read(addr, response)
        if self.check_sum(response):
            return response
        return None

    def get_addr(self, addr: int = 0) -> int:
        response = self._command(addr, self.CMD_GET_SENSOR_IIC_ADDR)
        return response[7] & 0xFF if response else 0

    def get_project(self, addr: int) -> int:
        response = self._command(addr, self.CMD_GET_PRG)
        if response:
            return (response[7] & 0xFF) + ((response[8] & 0xFF) << 8)
        return 0

    def set_send_type(self, addr: int, send_type: int = 0) -> bool:
        response = self._command(addr, self.CMD_SET_SENSOR_SEND_TYPE, send_type)
        return bool(response and (self.CMD_SET_SENSOR_SEND_TYPE | 0x80) == c_uint8(response[3]).value)

    def set_cap_offset(self, addr: int, offset: int) -> bool:
        response = self._command(addr, self.CMD_SET_SENSOR_CDC_START_OFFSET, offset, response_len=6)
        return bool(response and (self.CMD_SET_SENSOR_CDC_START_OFFSET | 0x80) == c_uint8(response[3]).value)

    def get_cap_data(self, addr: int, buffer: list[int]) -> bool:
        target_len = len(buffer)
        if self.ch341.read(addr, buffer) == 0:
            return False
        if len(buffer) != target_len:
            buffer.clear()
            buffer.extend(range(target_len))
        return (
            len(buffer) == target_len
            and (buffer[0] & 0xFF) == 0x55
            and (buffer[1] & 0xFF) == 0xAA
            and self.check_sum(buffer)
        )


@dataclass
class TashanTouchReader:
    pca_index: int = TASHAN_PCA_INDEX
    pca_addr: int = TASHAN_PCA_ADDR
    ch341: CH341Device = field(default_factory=CH341Device)
    cmd: SensorCommand | None = None
    sensor_addr: int | None = None
    param: FingerParam | None = None
    data: list[int] = field(default_factory=list)

    def open(self) -> None:
        self.ch341.open()
        self.cmd = SensorCommand(self.ch341)
        self._select_pca_channel()
        self._connect_sensor()

    def close(self) -> None:
        self.ch341.close()

    def _select_pca_channel(self) -> None:
        self.ch341.write(self.pca_addr, [1 << self.pca_index])

    def _connect_sensor(self) -> None:
        if self.cmd is None:
            raise RuntimeError("SensorCommand 尚未初始化")
        candidates = []
        read_addr = self.cmd.get_addr(0)
        if read_addr:
            candidates.append(read_addr)
        candidates.append(self.pca_index)

        for addr in dict.fromkeys(candidates):
            self._select_pca_channel()
            project = self.cmd.get_project(addr)
            if project in FINGER_PARAMS:
                self.sensor_addr = addr
                self.param = FINGER_PARAMS[project]
                self.cmd.set_send_type(addr, 0)
                self.cmd.set_cap_offset(addr, addr)
                self.data = list(range(self.param.pack_len))
                return
        raise RuntimeError(
            f"未读到有效他山传感器项目号，已尝试 I2C 地址 {candidates}。"
            f"请确认传感器插在 pca_index={self.pca_index} 对应接口并已上电。"
        )

    def read_nf_tf_dir(self) -> tuple[list[float], list[float], list[int]]:
        if self.cmd is None or self.sensor_addr is None or self.param is None:
            self.open()
        assert self.cmd is not None and self.sensor_addr is not None and self.param is not None

        self._select_pca_channel()
        for _ in range(3):
            if self.cmd.get_cap_data(self.sensor_addr, self.data):
                return self._parse_nf_tf_dir(self.data, self.param)
        raise RuntimeError("读取他山传感器数据失败")

    @staticmethod
    def _parse_nf_tf_dir(data: list[int], param: FingerParam) -> tuple[list[float], list[float], list[int]]:
        ydds_offset = 6 + param.sensor_num * param.cap_byte
        nf: list[float] = []
        tf: list[float] = []
        tf_dir: list[int] = []
        if param.ydds_type == 2:
            struct_size = ctypes.sizeof(DynamicYddsComTs)
            for i in range(param.ydds_num):
                offset = ydds_offset + i * struct_size
                raw = bytes((value & 0xFF) for value in data[offset : offset + struct_size])
                instance = DynamicYddsComTs.from_buffer_copy(raw)
                nf.append(float(instance.nf))
                tf.append(float(instance.tf))
                tf_dir.append(int(instance.tfDir))
        elif param.ydds_type == 4:
            struct_size = ctypes.sizeof(DynamicYddsU16Ts)
            for i in range(param.ydds_num):
                offset = ydds_offset + i * struct_size
                raw = bytes((value & 0xFF) for value in data[offset : offset + struct_size])
                instance = DynamicYddsU16Ts.from_buffer_copy(raw)
                nf.append(float(instance.nf) / 100.0)
                tf.append(float(instance.tf) / 100.0)
                tf_dir.append(int(instance.tfDir))
        else:
            raise RuntimeError(f"暂不支持他山 ydds_type={param.ydds_type}")
        return nf, tf, tf_dir


@dataclass
class EffortSensor:
    effort_dim: int
    effort_names: Sequence[str]
    pca_index: int = TASHAN_PCA_INDEX
    pca_indices: Sequence[int] | None = None
    read_hz: float = DEFAULT_TASHAN_READ_HZ
    _ch341: CH341Device = field(init=False, repr=False)
    _readers: list[TashanTouchReader] = field(default_factory=list, init=False, repr=False)
    _reader_slots: list[TashanTouchReader | None] = field(default_factory=list, init=False, repr=False)
    _last_effort: list[float] | None = field(default=None, init=False, repr=False)
    _last_timestamp: float | None = field(default=None, init=False, repr=False)
    _last_stale_warning: float = field(default=0.0, init=False, repr=False)
    _thread: Thread | None = field(default=None, init=False, repr=False)
    _stop_event: Event = field(default_factory=Event, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        indices = list(self.pca_indices) if self.pca_indices is not None else self._default_pca_indices()
        sensor_count = max(1, math.ceil(self.effort_dim / 6))
        if len(indices) > sensor_count:
            logger.warning(
                "TASHAN_PCA_INDICES=%s 超过 effort_dim=%s 需要的传感器数量 %s，只读取前 %s 个接口",
                indices,
                self.effort_dim,
                sensor_count,
                sensor_count,
            )
            indices = indices[:sensor_count]
        env_read_hz = os.environ.get("TASHAN_READ_HZ")
        if env_read_hz:
            self.read_hz = float(env_read_hz)
        if self.read_hz <= 0:
            raise ValueError(f"TASHAN_READ_HZ/read_hz 必须大于 0，当前是 {self.read_hz}")
        self._ch341 = CH341Device()
        self._ch341.open()
        cmd = SensorCommand(self._ch341)
        for index in indices:
            reader = TashanTouchReader(pca_index=index, ch341=self._ch341, cmd=cmd)
            try:
                reader._select_pca_channel()
                reader._connect_sensor()
            except Exception as exc:
                logger.warning("跳过 pca_index=%s 的他山传感器，使用 0 填充该路力数据: %s", index, exc)
                self._reader_slots.append(None)
                continue
            self._readers.append(reader)
            self._reader_slots.append(reader)
        if not self._readers:
            logger.warning("没有连接到任何他山传感器，record 将继续运行并记录全 0 effort 数据")
        self._last_effort = [0.0] * self.effort_dim
        self._last_timestamp = time.perf_counter()
        self._thread = Thread(target=self._read_loop, name="EffortSensor_read_loop", daemon=True)
        self._thread.start()

    def _default_pca_indices(self) -> list[int]:
        env_indices = os.environ.get("TASHAN_PCA_INDICES")
        if env_indices:
            return [int(index.strip()) for index in env_indices.split(",") if index.strip()]
        sensor_count = max(1, math.ceil(self.effort_dim / 6))
        return [self.pca_index + offset for offset in range(sensor_count)]

    @staticmethod
    def _to_fx_fy_fz(nf: float, tf: float, tf_dir: int) -> list[float]:
        if tf_dir == INVALID_TF_DIR:
            return [0.0, 0.0, nf]
        rad = math.radians(float(tf_dir))
        return [tf * math.cos(rad), tf * math.sin(rad), nf]

    def _read_once(self) -> list[float]:
        values: list[float] = []
        for reader in self._reader_slots:
            if reader is None:
                values.extend([0.0] * 6)
                continue
            nf_values, tf_values, tf_dir_values = reader.read_nf_tf_dir()
            for i in range(2):
                nf = float(nf_values[i]) if i < len(nf_values) else 0.0
                tf = float(tf_values[i]) if i < len(tf_values) else 0.0
                tf_dir = int(tf_dir_values[i]) if i < len(tf_dir_values) else INVALID_TF_DIR
                values.extend(self._to_fx_fy_fz(nf, tf, tf_dir))

        if len(values) < self.effort_dim:
            values.extend([0.0] * (self.effort_dim - len(values)))
        return values[: self.effort_dim]

    def _read_loop(self) -> None:
        period_s = 1.0 / self.read_hz
        next_read_time = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_read_time:
                self._stop_event.wait(next_read_time - now)
                continue
            try:
                values = self._read_once()
            except Exception as exc:
                logger.warning("读取他山传感器数据失败，继续使用上一帧力数据: %s", exc)
                next_read_time += period_s
                continue
            with self._lock:
                self._last_effort = values
                self._last_timestamp = time.perf_counter()
            next_read_time += period_s
            if next_read_time < time.perf_counter() - period_s:
                next_read_time = time.perf_counter()

    def read(self) -> list[float]:
        """返回后台线程最近一次读取的 [fx, fy, fz]，长度固定为 ``effort_dim``。"""
        with self._lock:
            if self._last_effort is None:
                self._last_effort = [0.0] * self.effort_dim
            timestamp = self._last_timestamp
            values = list(self._last_effort)

        now = time.perf_counter()
        if timestamp is not None and now - timestamp > 1.0 and now - self._last_stale_warning > 1.0:
            self._last_stale_warning = now
            logger.warning("他山传感器最新数据已 %.2fs 未更新，继续使用上一帧力数据", now - timestamp)
        return values

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._ch341.close()


def make_effort_sensor(effort_dim: int, effort_names: Sequence[str]) -> EffortSensor:
    return EffortSensor(effort_dim=effort_dim, effort_names=effort_names)
