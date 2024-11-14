# ROS simulation

## Preparation
- create a folder `unitree_ws/src` for your workspace, inside the folder clone the `unitree_ros` repo
```
git clone https://github.com/unitreerobotics/unitree_ros.git
```
- move the `unitree_legged_msgs` from `unitree_ros_to_real` into `unitree_ws/src`
```
cd unitree_ws
git clone https://github.com/unitreerobotics/unitree_ros_to_real.git
mv unitree_ros_to_real/unitree_legged_msgs/ /src
```
- build the ROS project (you can do an alias for the source or do it by default in the bashrc) 
```
catkin_make
source unitree_ws/devel/setup.bash
```
- run `roslaunch unitree_gazebo z1.launch`: if successfully configured, the simulation interface of Gazebo will be displayed
- clone and build `z1_controller` and `z1_sdk` (NOTE: for the controller, the ROS setup must be sourced)
```
cd unitree_ws/src
git clone https://github.com/unitreerobotics/z1_controller.git 
cd z1_controller && mkdir build && cmake .. && make
cd ../..
git clone https://github.com/unitreerobotics/z1_sdk.git
cd z1_sdk && mkdir build && cmake .. && make
```

## Simulation
* open three terminals
  1. roslaunch
  2. sim_ctrl
  3. what you want from the SDK