# Installation

## Dependencies
- `libboost-dev`
- `libeigen-dev`

## ROS Noetic installation
- setup pc for accepting ros
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
- setup keys
```
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```
- update and full desktop installation (recommended)
```
sudo apt update
sudo apt install ros-noetic-desktop-full -y
```