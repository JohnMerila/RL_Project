---
tags:
  - robotics
  - calibration
  - validation
  - sim-to-real
  - safety
date: 2026-08-18
---

# 07 — Calibration and Physical Validation

Previous: [[06 Safety, Bringup, and Operations]] · Back to [[Physical Robot Deployment - Start Here]] · Next:
[[08 Troubleshooting and Sim-to-Real Tuning]]

Treat deployment as a sequence of gates. Stop at the first failed gate, fix the underlying model/interface, and repeat
the earlier tests affected by the change.

## Measurement record

Create a dated record for each mechanical configuration:

| Quantity | Method | Result |
|---|---|---|
| Loaded left/right wheel radius | circumference or multi-turn distance test | |
| Effective wheel separation | repeated in-place turns | |
| Encoder counts per wheel revolution | manual marked-wheel rotation | |
| Maximum repeatable wheel speed | raised-wheel and ground test | |
| Linear acceleration/deceleration | logged command vs encoder velocity | |
| Angular acceleration/deceleration | logged command vs yaw rate | |
| Command-to-motion delay | timestamped step response | |
| Left/right steady-state bias | long straight trial | |
| LiDAR `(x,y,z,yaw)` | mechanical measurement and wall alignment | |
| Scan rate and age distribution | rosbag analysis | |
| Odometry drift | measured paths and rotations | |
| Pi inference p50/p95/max | loaded 20 Hz benchmark | |

Feed these values back into Isaac Lab nominal parameters and randomization ranges. Do not add arbitrary broad
randomization until the nominal simulated response matches the robot.

## Gate 0 — Offline contract and model

- Verify model and contract SHA-256.
- Assert ONNX input shape `(1,79)`, input dtype `float32`, and output shape `(1,2)`.
- Compare desktop and Pi actions for saved observation vectors.
- Run recorded bags through scan preprocessing and inference without publishing motion.
- Inject NaN, stale messages, missing TF, and inference exceptions; each must produce a stop state.

**Pass:** parity tolerance is met for every vector and all injected faults fail closed.

## Gate 1 — Electrical and raised wheels

- Verify E-stop, enable, MCU watchdog, and controller timeout independently.
- Confirm left/right wheel order and forward signs.
- Confirm one physical wheel revolution produces approximately `2π` reported radians.
- Step each wheel separately at low speed; measure response and fault behavior.
- Restart ROS/Pi/MCU in several orders and confirm no unexpected motion.

**Pass:** zero unexpected motion and every interruption stops both wheels within its documented bound.

## Gate 2 — Teleoperated base calibration

With the actor disconnected and speed limited:

1. drive straight 1 m and 2 m in both directions where supported;
2. rotate +360° and -360° several times;
3. drive a square clockwise and counterclockwise;
4. repeat with representative payload and battery state;
5. compare tape/external measurements, encoder odometry, and IMU if present.

If odometry distance is too large, the configured effective wheel radius is too large (or counts-per-revolution is
too small). If odometry yaw is too large for the physical turn, increase effective wheel separation. Diagnose
left/right asymmetry before applying a common multiplier.

**Pass:** signs are correct, scale errors meet your declared tolerance, and drift is repeatable enough to model.

## Gate 3 — LiDAR and TF

Place a flat box at measured distances and angles around the stationary robot:

| Placement | Expected policy bin |
|---|---:|
| 1.0 m forward | 0° bin, normalized range near 0.125 |
| 1.0 m left | +90° bin |
| 1.0 m right | -90° bin |
| 1.0 m rear | -180°/+180° wrap bin |

Then rotate the robot without moving the obstacle and verify `base_link`, `lidar_link`, and scan display remain
consistent. Check thin legs, dark surfaces, glass, sunlight if applicable, and returns below `range_min`.

**Pass:** angle sign, wraparound, range scaling, invalid-return handling, scan freshness, and TF are all verified from
logs—not only RViz appearance.

## Gate 4 — Shadow policy

Teleoperate through an open area and sparse soft obstacles while the policy proposes unconnected commands. Use goals
at front, left, right, and behind the robot. Review the bag offline.

Required observations:

- front clear + front goal → positive forward proposal;
- goal left/right → corresponding positive/negative yaw tendency;
- obstacle inserted in the proposed path → avoidance response before the safety zone;
- removed `/scan`, `/odom`, goal, or TF → zero proposal and explicit diagnostic reason;
- goal inside 0.30 m → latched stop;
- no sustained non-finite, deadline, or action-shape errors.

**Pass:** an operator review and automatic checks find no frame/contract fault.

## Gate 5 — Autonomous open floor

Use a large controlled area, soft boundaries, a human on the E-stop, and initial limits around:

```text
maximum linear speed:  0.10–0.15 m/s
maximum angular speed: 0.30–0.40 rad/s
```

Start with nearby unobstructed goals. Run goals in all directions relative to the initial heading. The robot should
enter and stop inside the visible 0.30 m goal disc/physical marker region. Record every attempt.

Suggested initial gate:

- 20/20 goals reached without contact;
- no watchdog or freshness violation during normal operation;
- no motion after goal completion;
- no reverse motion, because this policy is forward-only;
- acceptable stopping and yaw overshoot at the low cap.

The counts are engineering suggestions, not a safety certification. Set criteria appropriate to the platform and
environment before running the trials.

## Gate 6 — Controlled obstacles

Progress one variable at a time:

1. one large soft obstacle away from the direct goal line;
2. one obstacle directly on the path;
3. two widely separated obstacles;
4. sparse course matching simple simulation layouts;
5. narrower—but comfortably passable—gaps;
6. held-out layouts and changed approach angles.

Record:

- success, collision/contact, timeout, and supervisor intervention rate;
- time and path length on successful trials;
- minimum raw LiDAR clearance;
- closest footprint clearance from external measurement if available;
- raw action and proposed/safe/applied command differences;
- scan/odom age and inference latency;
- action saturation and oscillation;
- battery, payload, and floor condition.

Do not raise speed and obstacle density in the same test block.

## Fault-injection matrix

Test at zero or very low speed with the robot secured:

| Fault | Expected result |
|---|---|
| stop LiDAR publisher / unplug LiDAR | policy and collision monitor command zero |
| stop odometry | policy command zero |
| break required TF | policy command zero |
| kill policy process | drive timeout then MCU watchdog stop |
| kill collision monitor | downstream drive command times out; no bypass publisher |
| unplug Pi–MCU link | MCU stops/disable |
| ONNX file missing/corrupt | node refuses active mode |
| NaN/Inf injected in observation/output | immediate zero, latched diagnostic |
| Wi-Fi removed | onboard control continues safely or stops by declared design |
| localization jump | geofence/goal sanity logic inhibits unsafe motion |

## Comparison with simulation

Replay equivalent test geometry in Isaac Lab and compare:

- first motion delay;
- linear and angular step response;
- stopping distance;
- clearance chosen by the actor;
- scan minima near edges and thin objects;
- goal overshoot and turn oscillation.

A systematic mismatch should update measured simulation parameters followed by training/evaluation. Avoid adding an
unlogged physical-only multiplier that makes the deployed command contract diverge.

## Release record

For every approved deployment, archive:

```text
release_<date>_<model-name>/
├── policy.onnx
├── policy_contract.yaml
├── SHA256SUMS
├── ros_package_versions.txt
├── controller.yaml
├── collision_monitor.yaml
├── robot.urdf
├── hardware_measurements.md
├── parity_results.json
├── validation_results.csv
└── representative_bags/
```

Save `apt` package versions and the git commit of every source-built driver/package. A rollback should be a file and
configuration selection, not an attempt to reconstruct an old robot from memory.

