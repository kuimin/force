# TJ Robotiq USB Gripper Range

Current implementation uses the Python USB driver:

```python
from pyrobotiqgripper import RobotiqGripper
```

This driver is different from the Robotiq URCap script interface in
`/home/robot/robotic`.

## Official URCap Script

The URCap script uses a percent API:

```text
driver_gripper_client.move(slave_ids, position, RQ_UNIT_PERCENT, all_gripper_limits)
```

Examples from the official scripts:

```text
gripper open.script  -> rq_gripper_position = 0.0
gripper close.script -> rq_gripper_position = 100.0
```

That `0~100` value is interpreted by the URCap layer because it passes
`RQ_UNIT_PERCENT`.

## Python USB Driver

The Python USB driver `pyrobotiqgripper.RobotiqGripper.move()` expects raw
Robotiq position units:

```text
0   = fully open
255 = fully closed
```

Therefore `RobotiqUsbGripper.move_norm()` maps the teleoperation value like
this:

```python
position = int(round(float(np.clip(value, 0.0, 1.0)) * 255.0))
```

So:

```text
gripper.pos = 0.0 -> 0
gripper.pos = 0.5 -> 128
gripper.pos = 1.0 -> 255
```

This is the correct mapping for the current direct USB connection.

## Roll Back To Percent Version

If you intentionally switch to a percent-based API, change:

```python
position = int(round(float(np.clip(value, 0.0, 1.0)) * 255.0))
```

to:

```python
position = int(round(float(np.clip(value, 0.0, 1.0)) * 100.0))
```

Also change the log text in `src/lerobot/robots/tj/tj.py` from:

```python
"TJ sent Robotiq USB gripper command (%s.pos=%.3f -> %s/255)"
```

to:

```python
"TJ sent Robotiq USB gripper command (%s.pos=%.3f -> %s%%)"
```

Do not use the percent version with `pyrobotiqgripper.RobotiqGripper.move()`;
it will not fully close the gripper because that driver expects `0~255`.

