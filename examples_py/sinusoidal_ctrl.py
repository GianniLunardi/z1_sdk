import time 
import numpy as np
import unitree_arm_interface
import matplotlib.pyplot as plt
from orc.utils import plot_utils


np.set_printoptions(precision=4, suppress=True)

print("Press ctrl+\ to quit process.")

# 1) Start robot
arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState

armModel = arm._ctrlComp.armModel
joint_lwLimit = np.array(armModel.getJointQMin())
joint_upLimit = np.array(armModel.getJointQMax())
joint_speedLimit = np.array(armModel.getJointSpeedMax())

arm.loopOn() 

arm.startTrack(armState.JOINTCTRL)
q_init = arm.lowstate.getQ()

# Conf 
dt = arm._ctrlComp.dt
n_pre = 500
n = 1000
T_bang = n // 2
j = 5                       # Selected joint
q_arm_up = np.array([0., 0.26178, -0.26178, 0., 0., 0.])
q_per_joint = np.array([0, 0.8, -1.4, -0.5, 0, 0])
q_arm_up[j] = q_per_joint[j]
T_per = T_bang * dt
omega = 2 * np.pi / T_per
ampl = 0.3

# 2) Go to the desired starting position
print('[PRE]')
for i in range(n_pre):
    q = q_init * (1 - i / n_pre) + q_arm_up * (i / n_pre)
    v = (q_arm_up - q_init) / (n_pre * dt)
    arm.setArmCmd(q, v, np.zeros(6))
    # print(arm.lowstate.getQTau())
    time.sleep(dt)          # Arm timer is better (C++ code)

# 3) Bang-bang motion on selected joint

q_log, v_log = np.zeros((n, 6)), np.zeros((n, 6))
q_des, v_des = np.zeros_like(q_log), np.zeros_like(v_log)

time.sleep(2)
q = q_arm_up.copy()
v = np.zeros(6)
q_j = q[j]
print('[SINUSOIDAL]')
for i in range(n):
    start = time.time()
    q[j] = ampl * np.sin(omega * i * dt) + q_j
    v[j] = ampl * omega * np.cos(omega * i * dt)
    # Set the desired joint position and velocity
    arm.setArmCmd(q, v, np.zeros(6))
    q_des[i], v_des[i] = q, v
    # Get the actual joint position and velocity
    q_log[i], v_log[i] = arm.lowstate.getQ(), arm.lowstate.getQd()
    delta_t = time.time() - start
    time.sleep(dt - delta_t)

# 4) Back to start position
time.sleep(2)
print('[BACK]')
arm.backToStart()
arm.loopOff()

# 5) Plot the resulting trajectories 
BOUNDS_BOOL = 1
t = np.linspace(0, dt*n, n)
fig, ax = plot_utils.create_empty_figure(3, 2)
ax = ax.reshape(6)
for j in range(6):
    ax[j].plot(t, q_log[:, j], label='q' + str(j))
    ax[j].plot(t, q_des[:, j], label='q_r' + str(j), c='r', ls='--')
    if BOUNDS_BOOL:
        ax[j].axhline(joint_lwLimit[j], c='k', ls=':', lw=1)
        ax[j].axhline(joint_upLimit[j], c='k', ls=':', lw=1)
    ax[j].set_ylabel(f'q{j} (rad)')
    ax[j].set_xlabel('Time (s)')

fig, ax = plot_utils.create_empty_figure(3, 2)
ax = ax.reshape(6)
for j in range(6):
    ax[j].plot(t, v_log[:, j], label='v' + str(j))
    ax[j].plot(t, v_des[:, j], label='v_r' + str(j), c='r', ls='--')
    if BOUNDS_BOOL:
        ax[j].axhline(joint_speedLimit[j], c='k', ls=':', lw=1)
        ax[j].axhline(-joint_speedLimit[j], c='k', ls=':', lw=1)
    ax[j].set_ylabel(f'v{j} (rad/s)')
    ax[j].set_xlabel('Time (s)')

plt.show()