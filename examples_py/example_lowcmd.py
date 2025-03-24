import sys
sys.path.append("../lib")
import unitree_arm_interface
import time
import numpy as np
import matplotlib.pyplot as plt
# from orc.utils import plot_utils

print("Press ctrl+\ to quit process.")

np.set_printoptions(precision=3, suppress=True)
arm = unitree_arm_interface.ArmInterface(hasGripper=True)
armModel = arm._ctrlComp.armModel
arm.setFsmLowcmd()

# arm.calibration()

# arm.lowcmd.setZeroKp()
# arm.lowcmd.setZeroKd()
# kp = [20, 30, 30, 20, 15, 10]
# kd = [2000, 2000, 2000, 2000, 2000, 2000]
# kp[3] *= 2
# kd[3] *= 2
# arm.lowcmd.setControlGain(kp, kd)

duration = 1000
lastPos = arm.lowstate.getQ()
targetPos = np.array([0.0, 1.5, -1.0, -0.54, 0.0, 0.0]) #forward

q, qd, qdd, tau = np.zeros((duration, 6)), np.zeros((duration, 6)), np.zeros((duration, 6)), np.zeros((duration, 6))
q_des, qd_des, qdd_des, tau_des = np.zeros((duration, 6)), np.zeros((duration, 6)), np.zeros((duration, 6)), np.zeros((duration, 6))

for i in range(0, duration):
    arm.q = lastPos*(1-i/duration) + targetPos*(i/duration)# set position
    arm.qd = (targetPos-lastPos)/(duration*0.002) # set velocity
    arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6)) # set torque
    arm.gripperQ = -1*(i/duration)

    arm.setArmCmd(arm.q, arm.qd, arm.tau)
    arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
    arm.sendRecv()# udp connection
    # print(arm.lowstate.getQ())

    # Log 
    # print(type(arm.lowstate.getQ()))
    q[i] = arm.lowstate.getQ()
    qd[i] = arm.lowstate.getQd()
    # qdd[i] = arm.lowstate.getQdd()
    # tau[i] = arm.lowstate.getQTau()

    q_des[i] = arm.q
    qd_des[i] = arm.qd
    tau_des[i] = arm.tau

    time.sleep(arm._ctrlComp.dt)

arm.loopOn()
arm.backToStart()
arm.loopOff()

# # Plots
# t = np.linspace(0, duration*2e-3, duration)

# fig, ax = plot_utils.create_empty_figure(3, 2)
# ax = ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, q[:, i], label='q')
#     ax[i].plot(t, q_des[:, i], label='q_des', c='r', ls='--')
#     ax[i].set_ylabel(f'q_{i} (rad)')
#     ax[i].set_xlabel('Time (s)')
#     ax[i].legend()

# fig, ax = plot_utils.create_empty_figure(3, 2)
# ax = ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, qd[:, i], label='qd')
#     ax[i].plot(t, qd_des[:, i], label='qd_des', c='r', ls='--')
#     ax[i].set_ylabel(f'qd_{i} (rad/s)')
#     ax[i].set_xlabel('Time (s)')
#     ax[i].legend()

# fig, ax = plot_utils.create_empty_figure(3, 2)
# ax = ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, qdd[:, i], label='qdd')
#     ax[i].plot(t, qdd_des[:, i], label='qdd_des', c='r', ls='--')
#     ax[i].set_ylabel(f'qdd_{i} (rad/s^2)')
#     ax[i].set_xlabel('Time (s)')
#     ax[i].legend()

# fig, ax = plot_utils.create_empty_figure(3, 2)
# ax = ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, tau[:, i], label='tau')
#     ax[i].plot(t, tau_des[:, i], label='tau_des', c='r', ls='--')
#     ax[i].set_ylabel(f'tau_{i} (Nm)')
#     ax[i].set_xlabel('Time (s)')
#     ax[i].legend()

# plt.show()