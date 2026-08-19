---
tags:
  - robotics
  - ros2
  - lidar
  - observations
  - sim-to-real
date: 2026-08-18
---

# 04 — LiDAR and the 79-Value Contract

Previous: [[03 ROS 2 Base, Odometry, and TF]] · Back to [[Physical Robot Deployment - Start Here]] · Next:
[[05 ONNX Policy Node]]

The network does not consume ROS messages directly. It consumes one exact `float32[79]` tensor. This note defines the
real-robot adapter that produces it.

## LiDAR convention

ROS `sensor_msgs/LaserScan` defines zero angle along +X, with angles increasing counterclockwise about +Z. This matches
the policy convention when the LiDAR frame is mounted correctly. See the official
[LaserScan message definition](https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs/msg/LaserScan.msg).

The trained beam centers are:

```text
-180°, -175°, -170°, ... 0°, ... +170°, +175°
```

For each 5° bin:

1. discard non-finite values and ranges outside the message's valid range;
2. take the **minimum** valid return in the bin;
3. use 8.0 m when a bin has no valid return;
4. clip to `[0, 8]` m;
5. divide by 8.0;
6. output exactly 72 `float32` values in the order above.

Taking the minimum preserves thin, nearby obstacles. Do not average and do not select every Nth input ray.

> [!warning] Cover the LiDAR blind zone
> Returns below `range_min` are invalid by the message contract and cannot be treated as reliable free space. Mount
> the scanner so the chassis does not occlude it, and cover the near-field blind region with the independent footprint
> guard, bumpers, or another suitable sensor.

```python
import math
import numpy as np

NUM_BINS = 72
MAX_RANGE_M = 8.0
BIN_WIDTH = 2.0 * math.pi / NUM_BINS


def scan_to_policy_bins(scan) -> np.ndarray:
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
    wrapped = (angles + math.pi) % (2.0 * math.pi) - math.pi

    valid = np.isfinite(ranges)
    valid &= ranges >= max(0.0, float(scan.range_min))
    valid &= ranges <= min(MAX_RANGE_M, float(scan.range_max))

    # Bin centers are -pi + k*5deg. The half-bin offset implements nearest-center assignment.
    indices = np.floor(
        ((wrapped + math.pi + 0.5 * BIN_WIDTH) % (2.0 * math.pi)) / BIN_WIDTH
    ).astype(np.int64)

    output_m = np.full(NUM_BINS, MAX_RANGE_M, dtype=np.float32)
    np.minimum.at(output_m, indices[valid], ranges[valid])
    return np.clip(output_m, 0.0, MAX_RANGE_M) / MAX_RANGE_M
```

Do not use the received array index alone; some scanners publish different angular start points or clockwise data
adapted by their driver. Always use `angle_min + i * angle_increment` and verify a known obstacle.

## LiDAR timing and freshness

The `LaserScan.header.stamp` is the acquisition time of the first ray. At each 20 Hz policy tick:

- require at least one scan;
- reject a scan older than a measured threshold, initially 0.20 s;
- reject inconsistent length/angle metadata;
- stop on non-finite preprocessed output;
- log scan age and maximum observed age.

Use ROS sensor-data QoS for `/scan`. The policy wants the newest sample; a deep reliable backlog makes it act on stale
geometry. If the scanner is only 10 Hz, holding a scan for two policy ticks is acceptable only if this is measured,
logged, and included in later robustness training.

## LiDAR extrinsics

`base_link -> lidar_link` tells TF where the LiDAR is, but a `LaserScan` range is still measured from `lidar_link`.
The current simulator used `(x=0.08, y=0.0, z=0.215, yaw=0)` relative to the robot.

- Best option: mount the real LiDAR at approximately the trained offset and orientation.
- Better long-term option: measure the actual mount, update the Isaac configuration, and retrain/randomize it.
- Advanced option: project every return into `base_link`, then into a virtual trained sensor frame and re-bin.

Do not claim an extrinsic correction merely by renaming the scan frame.

## Relative goal

Goals may arrive in `map` or `odom` as `geometry_msgs/PoseStamped`. Each policy tick must transform the fixed world
goal using the **current** `base_link` pose:

```text
goal_b = TF(base_link <- goal_frame, latest) * goal_position
distance = hypot(goal_b.x, goal_b.y)
bearing = atan2(goal_b.y, goal_b.x)

goal_obs = [
    min(distance, 8.0) / 8.0,
    sin(bearing),
    cos(bearing),
]
```

Do not transform at the original goal-message timestamp on every tick; that freezes the relative goal at the robot
pose that existed when the goal was clicked. Request the latest transform, or copy the goal and set its transform time
to ROS time zero before using `tf2_ros.Buffer.transform()`.

For early tests, publish goals in `odom`. When map localization is ready, publish in `map`; the policy math remains the
same.

## Measured velocity

Use the twist from encoder-backed `/diff_drive_controller/odom` or fused `/odometry/filtered`:

```python
velocity_obs = np.array(
    [odom.twist.twist.linear.x / 0.8,
     odom.twist.twist.angular.z / 1.5],
    dtype=np.float32,
)
velocity_obs = np.clip(velocity_obs, -2.0, 2.0)
```

Do not substitute the last velocity command. The simulator supplied measured body velocity.

## Previous action

Slots `77:79` contain the previous raw normalized actor output after clipping to `[-1, 1]`. They do **not** contain:

- decoded m/s and rad/s;
- wheel speed;
- acceleration-limited command;
- collision-monitor output;
- measured velocity.

Initialize this pair to zero on node startup, model change, emergency stop, or goal completion.

## Pack and assert

```python
def pack_observation(scan72, goal3, velocity2, previous_action2):
    observation = np.concatenate(
        (scan72, goal3, velocity2, previous_action2), dtype=np.float32
    )
    if observation.shape != (79,):
        raise ValueError(f"Expected (79,), received {observation.shape}")
    if not np.all(np.isfinite(observation)):
        raise ValueError("Policy observation contains NaN or Inf")
    return observation
```

The order must match [policy_contract.yaml](../../../policy_contract.yaml). Treat observation-order changes as model
interface changes.

## Required unit tests

Create tests independent of ROS spinning:

1. all-clear 360° scan produces 72 ones;
2. a 1 m return at 0° produces `0.125` in the 0° bin;
3. several returns in one bin select the minimum;
4. `NaN`, `Inf`, below-min, and above-max are handled according to the documented rule;
5. returns around `-180°/+180°` enter the same wraparound bin correctly;
6. front/left/right synthetic goals produce bearings `0`, `+π/2`, and `-π/2`;
7. zero odometry and zero previous action produce the expected final four values;
8. packed dtype is `float32` and shape is exactly `(79,)`;
9. a recorded real scan produces the same tensor on the desktop and Pi;
10. a saved observation gives matching ONNX outputs on the desktop and Pi.

## Live visualization checks

In RViz, display `LaserScan`, robot model, TF, and goal. Place a cardboard box successively at front, left, right, and
rear. Log the selected 72 bins and verify the nearest dip moves through indices corresponding to 0°, +90°, ±180°,
and -90°. Do this before connecting actor output to the motor path.
