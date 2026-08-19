---
tags:
  - robotics
  - troubleshooting
  - sim-to-real
  - tuning
date: 2026-08-18
---

# 08 — Troubleshooting and Sim-to-Real Tuning

Previous: [[07 Calibration and Physical Validation]] · Back to [[Physical Robot Deployment - Start Here]]

Tune in this order:

1. coordinate signs and observation order;
2. wheel/encoder units and odometry geometry;
3. LiDAR angles, ranges, extrinsics, and timestamps;
4. control rate, latency, acceleration, and watchdogs;
5. safety zones and physical speed caps;
6. simulation nominal model and randomization;
7. rewards/PPO and retraining.

Do not retrain around a broken transform or encoder sign.

## Symptom table

| Symptom | Most likely cause | First test |
|---|---|---|
| Robot spins away from a left-side goal | goal bearing sign or TF direction reversed | synthetic goals at ±Y in `base_link` |
| Robot drives backward | wheel sign/order error or wrong action mapping | raised-wheel positive `TwistStamped` test |
| Robot always curves | wheel-radius/gain mismatch, encoder scaling, mechanical drag | equal low-speed wheel targets and long straight log |
| Odometry distance scale wrong | wheel radius or counts per revolution wrong | marked multi-revolution distance test |
| Odometry yaw scale wrong | effective wheel separation wrong | repeated external 360° rotations |
| RViz scan looks rotated | LiDAR yaw/driver angular convention wrong | box at physical +X and inspect angle 0 |
| Avoids on the wrong side | scan direction or body-axis convention reversed | boxes at ±90° and inspect policy bins |
| Clips corners | physical footprint/safety zone larger than simulation assumption | footprint overlay and measured clearance |
| Stops constantly | safety polygon too large, self-returns, noisy beams, `min_points` too low | RViz collision polygons and raw scan points |
| Does not stop for thin objects | bins/monitor `min_points` too permissive or LiDAR misses | pole tests at several angles |
| Oscillates in yaw | unmodeled delay, low odom rate, poor yaw feedback, excessive speed | step response and timestamp-age plot |
| Surges after a pause | acceleration state uses fixed dt after timer stall | cap elapsed dt and reset on inactive state |
| Moves after reaching goal | no terminal wrapper/latch on physical robot | inspect goal distance and state machine |
| Policy works briefly then stops | stale scan/odom/TF threshold too tight or CPU deadline miss | log ages and inference latency percentiles |
| ONNX action differs from desktop | wrong model, dtype/order, normalizer, runtime/export mismatch | golden-vector parity test and checksums |
| Great open-floor behavior, poor mazes | reactive policy lacks global route/history | provide reachable local waypoints from a global planner |
| Policy proposes motion but robot stays still | collision monitor intervention, inactive controller, MCU fault | compare raw/safe/applied topics and diagnostics |
| Sudden autonomous motion on restart | enable/mux/startup state is unsafe | require explicit enable edge and start in disabled mode |

## Separate raw, safe, and applied commands

Log three stages with distinct topics:

```text
/policy/cmd_vel                         raw proposed body command
/diff_drive_controller/cmd_vel         safety-approved body command
/diff_drive_controller/cmd_vel_out     controller-limited body command
```

Also log measured odometry. This makes it possible to answer whether a behavior came from the actor, collision
monitor, controller limits, low-level motor response, or wheel slip.

## Rate and latency budget

The policy period is 50 ms. Build a measured budget:

| Component | p50 | p95 | maximum |
|---|---:|---:|---:|
| LiDAR acquisition/message age | | | |
| TF lookup | | | |
| scan binning + observation packing | | | |
| ONNX inference | | | |
| safety monitor | | | |
| ROS-to-MCU communication | | | |
| motor response to first motion | | | |

Optimize only after measurement. For this small MLP, one ONNX Runtime thread may outperform a larger thread pool and
leave more CPU for ROS. Use active cooling, avoid console spam, and inspect throttling during a full-system load.

## Map physical findings back to Isaac Lab

| Physical measurement | Simulation/configuration target |
|---|---|
| loaded wheel radius | `wheel_radius` nominal and randomization |
| effective wheel separation | `track_width` nominal and range |
| command-to-motion delay | action FIFO delay in policy frames |
| measured acceleration/braking | command acceleration limits and actuator response |
| left/right bias | asymmetric wheel-radius or actuator gain randomization |
| floor-dependent slip | friction and wheel-response range |
| scan range residuals | LiDAR noise/bias model |
| dropped/stale sectors | dropout and scan-age model |
| LiDAR mount uncertainty | sensor translation/yaw randomization |
| localization noise/jumps | relative-goal noise, delay, and fault tests |

Randomization should bracket real measurements. If a real value is at the edge or outside the trained distribution,
update the simulator and retrain rather than hoping the safety monitor corrects navigation quality.

## When the policy needs a global planner

This actor is a reactive local point-goal controller. One scan cannot reason reliably about a long U-shaped trap,
loop, or blocked room. For longer missions:

1. build/localize in a map with SLAM Toolbox or another localization stack;
2. use a global planner to generate a route;
3. select a reachable local waypoint a short distance ahead;
4. publish that waypoint as `/goal_pose` for the actor;
5. advance it when reached while retaining the independent safety path.

Do not point the reactive actor directly at a distant map goal through walls and interpret failure as a PPO tuning
problem.

## Change-control checklist

Any of these changes require at least contract/parity and regression testing:

- model or ONNX Runtime version;
- observation code, normalization, bin count, or angular convention;
- LiDAR model, driver, firmware, baud rate, mount, or scan mode;
- wheel, tire, payload, motor, gearbox, encoder, or driver;
- controller rate, geometry, limits, timeout, or feedback mode;
- TF frame names/ownership or localization stack;
- policy/safety topic remapping;
- collision polygons or point thresholds;
- OS, ROS distribution, or source-built package commit.

Changes affecting model input distribution or dynamics should trigger held-out simulation evaluation and likely
retraining. Changes affecting safety must repeat the relevant fault-injection tests.

## Useful official references

- [ROS 2 Jazzy Ubuntu installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)
- [ROS 2 on Raspberry Pi](https://docs.ros.org/en/ros2_documentation/kilted/How-To-Guides/Installing-on-Raspberry-Pi.html)
- [`diff_drive_controller` Jazzy documentation](https://control.ros.org/jazzy/doc/ros2_controllers/diff_drive_controller/doc/userdoc.html)
- [`ros2_control` hardware components](https://control.ros.org/jazzy/doc/ros2_control/hardware_interface/doc/hardware_components_userdoc.html)
- [Nav2 transformation setup](https://docs.nav2.org/setup_guides/transformation/setup_transforms.html)
- [Nav2 Collision Monitor](https://docs.nav2.org/configuration/packages/configuring-collision-monitor.html)
- [ONNX Runtime Python installation](https://onnxruntime.ai/docs/get-started/with-python.html)
- [ONNX Runtime thread tuning](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)

