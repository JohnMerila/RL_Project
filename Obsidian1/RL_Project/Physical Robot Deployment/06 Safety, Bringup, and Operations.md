---
tags:
  - robotics
  - ros2
  - safety
  - operations
  - lidar
date: 2026-08-18
---

# 06 — Safety, Bringup, and Operations

Previous: [[05 ONNX Policy Node]] · Back to [[Physical Robot Deployment - Start Here]] · Next:
[[07 Calibration and Physical Validation]]

No neural policy, LiDAR, Linux process, or ROS node should have sole authority to keep the platform safe. Use layers
with different failure modes.

## Required safety layers

| Layer | Failure it covers | Suggested initial behavior |
|---|---|---|
| Physical latching E-stop | all software/electronics above motor power | removes motor torque/enable |
| Motor driver protection | overcurrent, overtemperature, undervoltage | disables and reports a latched fault |
| MCU watchdog | Pi crash, serial/CAN loss, stuck ROS | zero/disable after 100–200 ms |
| `diff_drive_controller` timeout | command publisher stops | zero after 0.25 s |
| Policy freshness checks | stale scan, odom, TF, or goal | publish zero and reset policy state |
| Collision monitor | unsafe learned command near an obstacle | slow or stop based on footprint zones |
| Geofence/mission state | leaving controlled area or no active mission | inhibit motion |
| Human observer | unexpected physical behavior | hold remote/physical E-stop |

Test each layer by deliberately breaking the input it monitors. A configured watchdog that has never been triggered is
not a validated watchdog.

> [!danger] Software collision monitoring is not safety-rated
> Nav2 Collision Monitor is a useful additional guard, not a substitute for a certified safety scanner, safety PLC,
> guarded test area, or physical E-stop where those are required.

## Collision Monitor routing

Nav2's Collision Monitor can consume `LaserScan` points and filter velocity before the drive controller. Its design and
parameters are documented in [Collision Monitor](https://docs.nav2.org/configuration/packages/configuring-collision-monitor.html)
and the [usage tutorial](https://docs.nav2.org/tutorials/docs/using_collision_monitor.html).

Starting configuration—measure the footprint and stopping behavior before accepting these dimensions:

```yaml
collision_monitor:
  ros__parameters:
    enabled: true
    base_frame_id: base_link
    odom_frame_id: odom
    cmd_vel_in_topic: /policy/cmd_vel
    cmd_vel_out_topic: /diff_drive_controller/cmd_vel
    enable_stamped_cmd_vel: true
    transform_tolerance: 0.10
    source_timeout: 0.20
    stop_pub_timeout: 0.50

    polygons: [FootprintStop, FrontSlow]

    FootprintStop:
      type: polygon
      points: "[[0.34, 0.27], [0.34, -0.27], [-0.28, -0.27], [-0.28, 0.27]]"
      action_type: stop
      min_points: 3
      visualize: true
      polygon_pub_topic: collision_stop_zone

    FrontSlow:
      type: polygon
      points: "[[0.75, 0.40], [0.75, -0.40], [0.20, -0.40], [0.20, 0.40]]"
      action_type: slowdown
      slowdown_ratio: 0.25
      min_points: 3
      visualize: true
      polygon_pub_topic: collision_slow_zone

    observation_sources: [scan]
    scan:
      type: scan
      topic: /scan
      source_timeout: 0.20
      enabled: true
```

Visualize both polygons in RViz. The stop polygon must contain the physical footprint plus uncertainty, but the LiDAR
must not see the robot's own chassis as an obstacle. Tune `min_points` carefully: too high misses thin legs; too low
may stop on isolated sensor noise. Test both deliberately.

## Braking-distance envelope

Measure stopping distance on every intended floor and with representative battery and payload. A lower bound for free
distance is:

```text
d_required = v * total_delay + v^2 / (2 * measured_deceleration) + fixed_margin
```

`total_delay` includes LiDAR acquisition, message age, policy period, ROS transport, collision monitor, controller,
MCU, and motor response. Use the slowest credible measured deceleration, not the best stop. Increase the slowdown/stop
zones or reduce maximum speed until the robot consistently stops outside the chosen clearance.

A static polygon is easiest to validate at low speed. Later, use velocity-dependent polygons or an additional dynamic
braking guard if speed varies widely.

## Bringup sequence

Use one launch file eventually, but validate components in this order:

1. Place the robot on blocks, engage the E-stop, and verify the area is clear.
2. Start `robot_state_publisher` and inspect the robot model/TF.
3. Start the hardware interface and controller manager while controllers are inactive.
4. Start the LiDAR driver; inspect `/scan`, rate, timestamps, frame, and RViz orientation.
5. Activate `joint_state_broadcaster`, then the diff-drive controller.
6. Test low-speed manual commands and every watchdog.
7. Start odometry fusion/localization if used; verify TF ownership.
8. Start Collision Monitor and publish synthetic/manual commands through it.
9. Start the policy in **shadow mode** with no route to the controller.
10. Record and review proposed commands against live geometry.
11. Connect policy → monitor → controller only after shadow-mode acceptance.
12. Release the E-stop with a human holding the stop and a low software speed cap.

Useful checks:

```bash
ros2 node list
ros2 topic list -t
ros2 topic hz /scan
ros2 topic hz /diff_drive_controller/odom
ros2 topic hz /policy/cmd_vel
ros2 topic info /diff_drive_controller/cmd_vel --verbose
ros2 run tf2_tools view_frames
ros2 doctor --report
```

`topic info --verbose` should show the collision monitor—not the policy—as the publisher connected to the drive
controller.

## Shadow mode

In shadow mode the policy performs all preprocessing and inference but publishes to an unconnected topic such as
`/policy/shadow_cmd_vel`. Teleoperate slowly while recording:

```bash
ros2 bag record --storage mcap \
  /scan \
  /joint_states \
  /diff_drive_controller/odom \
  /odometry/filtered \
  /goal_pose \
  /policy/shadow_cmd_vel \
  /tf /tf_static \
  /diagnostics
```

Review:

- Does the actor turn away from close obstacles with the expected sign?
- Does it point generally toward unobstructed goals?
- Does its command stop immediately when scan/odom/TF is removed?
- Are scan/odom ages and inference deadlines within limits?
- Would the Collision Monitor have intervened?
- Are any actions saturated for long periods?

Shadow mode cannot prove closed-loop performance, but it catches frame and preprocessing failures without giving the
network motor authority.

## Operational modes

Make the robot's state explicit:

```text
DISABLED -> MANUAL -> SHADOW -> AUTONOMOUS -> GOAL_REACHED
                         \-> FAULT <- any state
```

Rules:

- only a deliberate operator action enters `AUTONOMOUS`;
- every transition into `DISABLED`, `GOAL_REACHED`, or `FAULT` publishes zero and disables/clears policy state;
- new goals do not clear a hardware fault;
- reboot does not restore motor enable automatically;
- manual and autonomous command sources use a deliberate, observable mux—never race on one topic;
- the active mode and stop reason are visible in diagnostics and logs.

## Startup service

After manual bringup is repeatable, use a systemd service or supervised launch process. It should:

- wait for the network only if remote resources are actually required;
- source `/opt/ros/jazzy/setup.bash`, the workspace, and the pinned venv;
- set `ROS_DOMAIN_ID` and model/config paths explicitly;
- restart non-motion support nodes where appropriate;
- leave motor enable off after a crash/restart;
- write logs to persistent storage with rotation;
- expose health through ROS diagnostics.

Do not add automatic startup until manual fault handling is proven.

## Pre-motion checklist

- [ ] Test boundary is controlled and the floor is dry/clear.
- [ ] Battery, fuses, connectors, wheels, and LiDAR mount are secure.
- [ ] Physical and remote E-stops have been tested this session.
- [ ] Model checksum and configuration version are the approved pair.
- [ ] Correct ROS domain and robot namespace are active.
- [ ] No duplicate command or TF publishers exist.
- [ ] Scan, odometry, TF, and diagnostics are fresh.
- [ ] Speed cap matches the current validation stage.
- [ ] Rosbag recording is running.
- [ ] A human has uninterrupted line of sight and access to the stop.

