#!/usr/bin/env python

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("noop")
@dataclass
class NoopTeleopConfig(TeleoperatorConfig):
    pass
