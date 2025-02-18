# Qualisys Motion Capture

## System setup
- Prepare the cabling for integrating mocap and the two PCs (mocap and experiment)
- (Insert image and then add description of the cabling)
- Turn on the system by simpling giving current 
- Verify that all the cameras are properly connected. At the startup, an orange/red light will turn on. After some time, the lights turn off and a number is assigned to each camera (you can see on the left bottom of the camera). If it is not the case, probably some bridges between the cameras do not work properly: verify if some cables are broken.

## QTM
- Open **Qualisys Track Manager** in the Mocap PC. At the startup it will ask which project you want to open: at the moment, use *GPS model validation*
- If the project is already setup, you can click on **New** (`Ctrl + N`) and skip the next three steps
- **Project options** (`Ctrl + ,`) &rarr; **Camera System** &rarr; **Cameras** and create a number of groups equal to the number of the cameras. This will provide some delays between the flash time, such that each camera does not see the others
- Adapt settings to environment (like **Capture rate**, **Exposure & Flash Time**, **Marker Threshold**). In the **2D View** you should see the markers in white and all the environment in black.
- Calibrate the system using the *L frame* and the *calibration pole*. Click on **Calibrate** in the Toolbar, then start moving around inside the cameras' environment with the calibration pole until the process finish. Only the markers in the L frame must be static during the process 
- If the calibration is successfull, the global rf will be placed on the rf markers. The std should be < 1 mm, but also some mm may be okay for our purposes  
- Make some *rigid bodies* of the feature you want to track. Select all the desired markers (at least 4) using `Shift + click' on each of them or including inside the selection window, then right click &rarr; **Define rigid body (6DOF)** &rarr; **Current frame** and give it a name
- (Optional) Move the global reference frame on the base of the robot. **Project Options** &rarr; **Processing** &rarr; **6DOF Tracking**, select the base body and then click on **Cordinate System**. Using the cordinate system relative to the global one, click on **Get position** and take note of the xyz coordinates. Then, in project options go to **Input Devices** &rarr; **Calibration** &rarr; **Transformation**. You can click on *Translate origin* and move the global referance frame on the one of the body using its xyz position.

## Communication with other PC

If you want to stream the data from the Mocap PC to another one, you will need to connect the two PCs via Ethernet cable. On the experiment PC, configurate the wired settings with the static IP address **192.168.225.x** and netmask **255.255.255.0**. To verify the connection with the Mocap PC, ping the following IP
```
ping 192.168.225.1
```
**Notice**: deactive the Windows firewall on the Mocap PC, if not the PC would not be found.

Then, on the experiment PC you can use QTM real-time from python to retrieve the streamed data. Be aware to select the compatible version of `qtm_rt` with QTM on Mocap PC (see the Releases at the [link](https://github.com/qualisys/qualisys_python_sdk)). For example:
```
pip install qtm_rt==3.0.1       # For QTM 2023.3
```