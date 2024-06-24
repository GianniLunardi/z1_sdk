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
arm.loopOn() 

arm.startTrack(armState.JOINTCTRL)
q_init = arm.lowstate.getQ()

# Conf 
dt = arm._ctrlComp.dt
j = 0
n_pre = 500
bang_cycles = 2
T_bang = 500
# Each cycle has 4 bangs: 1 acc, 2 dec, 1 acc
n_cycle = T_bang * 4            
a_bang = 1 
joint_delta = a_bang * (T_bang * dt)**2
print('Joint delta: ', joint_delta)

# Define the acceleration profile
a_cycle = np.zeros(n_cycle)
a_cycle[:T_bang] = a_bang
a_cycle[T_bang:3*T_bang] = -a_bang  
a_cycle[3*T_bang:] = a_bang

# Repeat the acceleration profile for the number of cycles
a = np.tile(a_cycle, bang_cycles)

q_arm_up = np.array([0., 0.26178, -0.26178, 0., 0., 0.])
q_per_joint = np.array([0, 0., -1.4, -1.2, 0, 0])
q_arm_up[j] = q_per_joint[j]

# 2) Go to the desired starting position
print('[PRE]')
for i in range(n_pre):
    q = q_init * (1 - i / n_pre) + q_arm_up * (i / n_pre)
    v = (q_arm_up - q_init) / (n_pre * dt)
    arm.setArmCmd(q, v, np.zeros(6))
    # print(arm.lowstate.getQTau())
    time.sleep(dt)          # Arm timer is better (C++ code)

# 3) Bang-bang motion on selected joint
n = len(a)
q_log, v_log = np.zeros((n, 6)), np.zeros((n, 6))
q_des, v_des = np.zeros_like(q_log), np.zeros_like(v_log)

time.sleep(2)
q = q_arm_up.copy()
v = np.zeros(6)
q_j, v_j = q[j], 0
print('[BANG]')
for i in range(n):
    start = time.time()
    v_plus = v_j + 0.5 * a[i] * dt
    v_j += a[i] * dt
    q_j += v_plus * dt
    q[j], v[j] = q_j, v_j
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
t = np.linspace(0, dt*n, n)
fig, ax = plot_utils.create_empty_figure(3, 2)
ax = ax.reshape(6)
for j in range(6):
    ax[j].plot(t, q_log[:, j], label='q' + str(j))
    ax[j].plot(t, q_des[:, j], label='q_r' + str(j), c='r', ls='--')
    # ax[j].axhline(arm.lowstate.getJointLimit(j, 0), c='k', ls='--', lw=1)
    # ax[j].axhline(arm.lowstate.getJointLimit(j, 1), c='k', ls='--', lw=1)
    ax[j].set_ylabel(f'q{j} (rad)')
    ax[j].set_xlabel('Time (s)')

fig, ax = plot_utils.create_empty_figure(3, 2)
ax = ax.reshape(6)
for j in range(6):
    ax[j].plot(t, v_log[:, j], label='v' + str(j))
    ax[j].plot(t, v_des[:, j], label='v_r' + str(j), c='r', ls='--')
    ax[j].set_ylabel(f'v{j} (rad/s)')
    ax[j].set_xlabel('Time (s)')

plt.show()