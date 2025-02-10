# Real Robot control

## Setup
- makes sure `z1_ctrl` program is turned off before turn on the manipulator
- turn on the robotic arm, when the device is powered on successfully,
the green light is steady on, and the blue light will flash once the self-check passes. Make sure the robot is in the home position (all the joints in zero)
- connect the arm to the PC through an Ethernet cable 
- setup the network configuration through the UI (this should be sufficient only one time), Settings ⇒ Network ⇒ Wired and then set manually the IPv4 as in the following image 

  ![ipv4](./images/ipv4.png)

- the default IP address of the robotics arm is **192.168.123.110**, make sure to ping it when the wired connection is established 

## Experiment
- open two terminals 
    1. run the program `z1_ctrl` (remember to build the source code if not been done)
        ```
        cd unitree_ws/src/z1_controller/build
        ./z1_ctrl
        ```
    2. on the other terminal, run one of the programs (either from c++ or python) inside the SDK, for example run `highcmd_development`:
        ```
        cd unitree_ws/src/z1_sdk/build
        ./highcmd_development
        ```
- NOTE: each experiment can be interrupted by pressing `ctrl + \` in the terminal of the SDK. Remember that the interruption bring the robot in the passive mode, so it will fall down. 

## Keybord
The robotic arm can be command using the keyboard, both in simulation or real experiment:
```
cd unitree_ws/src/z1_controller/build
./sim_ctrl k
# OR
.z1_ctrl k
```
All the possible states and commands can be found on the following [link](https://support.unitree.com/home/en/Z1_developer/keyboard)