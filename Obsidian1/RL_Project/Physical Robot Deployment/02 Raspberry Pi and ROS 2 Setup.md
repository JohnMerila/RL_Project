---
tags:
  - robotics
  - ros2
  - raspberry-pi
  - deployment
date: 2026-08-18
---

# 02 — Raspberry Pi and ROS 2 Setup

Previous: [[01 Hardware Architecture and Wiring]] · Back to [[Physical Robot Deployment - Start Here]] · Next:
[[03 ROS 2 Base, Odometry, and TF]]

## 1. Install the operating system

Flash **Ubuntu Server 24.04 LTS 64-bit** with Raspberry Pi Imager. Configure a hostname such as `navbot`, a non-default
user, SSH keys, locale, timezone, and Wi-Fi or Ethernet. Prefer Ethernet while developing.

On first boot:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Use active cooling and confirm there is no power throttling under load. Keep the OS headless; run RViz on a laptop.

## 2. Install ROS 2 Jazzy

Follow the official [Jazzy deb installation](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).
The essential sequence is:

```bash
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe

export ROS_APT_SOURCE_VERSION=$(curl -s \
  https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F 'tag_name' | awk -F'"' '{print $4}')

curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"

sudo dpkg -i /tmp/ros2-apt-source.deb
sudo apt update
sudo apt install -y ros-jazzy-ros-base ros-dev-tools
```

Install the robot packages:

```bash
sudo apt install -y \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-xacro \
  ros-jazzy-tf2-ros \
  ros-jazzy-tf2-tools \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-robot-localization \
  ros-jazzy-nav2-collision-monitor \
  ros-jazzy-diagnostic-updater \
  ros-jazzy-rosbag2-storage-mcap \
  python3-colcon-common-extensions \
  python3-venv \
  python3-numpy
```

Install the vendor LiDAR driver from its official Jazzy instructions. Prefer a released apt package; otherwise pin a
known commit in the workspace and use `rosdep`.

Add ROS setup to the interactive shell:

```bash
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

Do not use Conda for the robot's ROS environment. ROS's official Python-package guide warns that Conda's interpreter
is generally incompatible with the interpreter used for ROS binary packages:
[Using Python Packages with ROS 2](https://docs.ros.org/en/jazzy/How-To-Guides/Using-Python-Packages.html).

## 3. Create the deployment workspace

```bash
mkdir -p ~/robot_ws/src ~/robot_ws/model
cd ~/robot_ws

python3 -m venv --system-site-packages .venv
touch .venv/COLCON_IGNORE
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'onnxruntime==1.27.0'
```

ONNX Runtime publishes CPU wheels for CPython 3.12 on Linux ARM64. The official install is
`pip install onnxruntime`; pin the version after validating it on this robot. See the
[ONNX Runtime Python guide](https://onnxruntime.ai/docs/get-started/with-python.html).

Verify the architecture and runtime:

```bash
uname -m
python - <<'PY'
import onnxruntime as ort
print(ort.__version__)
print(ort.get_available_providers())
PY
```

Expected architecture is `aarch64`; the CPU runtime should include `CPUExecutionProvider`.

## 4. Copy and verify the deployment bundle

On the training computer, playback has already exported the ONNX actor. Copy it to the Pi:

```bash
scp \
  /home/megrad/RL_Project/logs/rsl_rl/diffdrive_lidar_nav/2026-08-18_12-18-31_full_seed42/exported/policy.onnx \
  ubuntu@navbot.local:/home/ubuntu/robot_ws/model/policy.onnx

scp /home/megrad/RL_Project/policy_contract.yaml \
  ubuntu@navbot.local:/home/ubuntu/robot_ws/model/policy_contract.yaml
```

Replace `ubuntu` and `navbot.local` with the configured user and hostname. Also archive the run's `params/env.yaml`,
`params/agent.yaml`, and checkpoint off the robot.

Create checksums on both machines and compare them:

```bash
sha256sum ~/robot_ws/model/policy.onnx ~/robot_ws/model/policy_contract.yaml
```

For the currently exported `full_seed42/model_1999.pt` bundle, the source files are:

```text
c3e941360d05e6075fefcfcc461a1c46829682c2280523d07bc50f96a7e47863  policy.onnx
3bcabbb42df00dac7bbea14affb2b652fd01c50b163e677938c351cc94e75a4f  policy_contract.yaml
```

The ONNX graph was also inspected as `obs: float32[1,79] -> actions: float32[1,2]`. Recompute and record new hashes
whenever a different checkpoint is exported; do not reuse these values for another model.

Never silently replace `policy.onnx` in the field. Give each approved model a version, checksum, validation record,
and rollback copy.

## 5. Workspace layout

A practical ROS workspace is:

```text
~/robot_ws/
├── .venv/
├── model/
│   ├── policy.onnx
│   ├── policy_contract.yaml
│   └── SHA256SUMS
└── src/
    ├── navbot_description/       # URDF/Xacro, meshes, ros2_control block
    ├── navbot_hardware/          # C++ hardware interface to MCU/controller
    ├── navbot_policy/            # Python ONNX policy and scan preprocessor
    └── navbot_bringup/           # launch, controller, EKF, safety configs
```

Build using the same venv every time:

```bash
cd ~/robot_ws
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Add only the final known-good setup sequence to `.bashrc` or a service environment. Avoid sourcing several ROS
distributions or stale workspaces.

## 6. ROS network configuration

For one robot and one development laptop, use the same domain ID:

```bash
echo 'export ROS_DOMAIN_ID=42' >> ~/.bashrc
export ROS_DOMAIN_ID=42
```

Allow ROS discovery on the selected network, but do not expose an unauthenticated robot ROS graph to an untrusted
network. Use a dedicated robot LAN, firewall, VPN, or ROS security configuration as appropriate. Autonomy must keep
working—or stop safely—when Wi-Fi disappears.

## 7. Basic verification

```bash
printenv ROS_DISTRO
ros2 doctor --report
ros2 pkg executables controller_manager
ros2 pkg executables diff_drive_controller
ros2 pkg executables nav2_collision_monitor
```

Expected ROS distribution is `jazzy`. At this point no motor should be enabled. Continue with
[[03 ROS 2 Base, Odometry, and TF]].
