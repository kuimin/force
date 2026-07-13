# DM-EXton TJ IK Filtering Notes

This note explains where the teleoperator filtering code is added and how the
gripper command path works.

## Pose Filter

The end-effector pose filter is implemented in:

```text
src/lerobot/teleoperators/dm_exton_tj_ik/dm_exton_tj_ik.py
```

The reusable filter classes are:

```python
class _LowPassFilter:
    ...

class _OneEuroFilter:
    ...

class _PoseFilter:
    ...
```

`_PoseFilter` uses two `_OneEuroFilter` instances:

```python
self.pos_filter = _OneEuroFilter(freq, mincutoff, beta, dcutoff)
self.quat_filter = _OneEuroFilter(freq, mincutoff, beta, dcutoff)
```

The pose callback applies the filter here:

```python
def _pose_callback(self, msg: Any) -> None:
    position, quat = self._extract_pose_from_msg(msg)
    timestamp = self._timestamp_from_msg(msg)
    if self.config.mapping_mode not in {"position_increment", "pose_increment"}:
        position, quat = self._pose_filter.process(position, quat, timestamp)
```

So the pose data flow is:

```text
/target_robot/right_ee/target
    -> _pose_callback()
    -> _PoseFilter.process()
    -> _latest_pose
    -> get_action()
    -> IK
    -> joint action
```

## Gripper Path

The gripper path does not use a filter now. The gripper callback reads
`/trigger_positions`, clips the value to `[0, 1]`, optionally inverts it, then
saves it directly as `_latest_gripper`:

```python
def _gripper_callback(self, msg: Any) -> None:
    data = list(getattr(msg, "data", []))
    index = self.config.gripper_index
    value = float(np.clip(float(data[index]), 0.0, 1.0))
    if self.config.gripper_invert:
        value = 1.0 - value

    with self._lock:
        self._latest_gripper = value
```

The gripper data flow is:

```text
/trigger_positions
    -> _gripper_callback()
    -> _latest_gripper
    -> action["gripper.pos"]
    -> TJRobot.send_action()
    -> RobotiqUsbGripper.move_norm()
```

## Config Parameters

The pose filter parameters are defined in:

```text
src/lerobot/teleoperators/dm_exton_tj_ik/config_dm_exton_tj_ik.py
```

```python
filter_frequency_hz: float = 1000.0
filter_mincutoff: float = 2.5
filter_beta: float = 0.2
filter_dcutoff: float = 1.0
```

These values mean:

```text
lower mincutoff -> smoother, slower response
higher mincutoff -> faster response, less smoothing
lower beta      -> less speed-based adaptation, more stable
higher beta     -> follows fast motion more aggressively
```

## Current Minimal Teleoperation Command

Most right-arm defaults are already in config:

```text
robot.arm = B
teleop.arm = B
teleop.pose_topic = /target_robot/right_ee/target
fps = 60
teleop.use_clutch = false
```

So the minimal command is:

```bash
python src/lerobot/scripts/lerobot_teleoperate.py \
  --robot.type=tj \
  --robot.robotiq_usb_port=/dev/ttyUSB0 \
  --teleop.type=dm_exton_tj_ik
```
