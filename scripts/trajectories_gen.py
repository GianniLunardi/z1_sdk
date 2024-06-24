import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pinocchio as pin
import example_robot_data


DISPLAY_N = 20
mpl.rcParams['axes.grid'] = True
robot = example_robot_data.load('z1')
viz = pin.visualize.MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
viz.initViewer(loadModel=True, open=True)

rmodel = robot.model
rdata = rmodel.createData()
q0 = rmodel.referenceConfigurations['arm_up']

# Change IC depending on which joint we want to move
j = 0 if len(sys.argv) < 2 else int(sys.argv[1])           
q_per_joint = np.array([0, 0.05, -1.2, -0.7, 0, 0])
q0[j] = q_per_joint[j]
viz.display(q0)

dt = 2e-3
bang_cycles = 2
T_bang = 500
# Each cycle has 4 bangs: 1 acc, 2 dec, 1 acc
n_cycle = T_bang * 4            
v_lim = np.min(rmodel.velocityLimit) / 1.6
a_bang = 1 
joint_delta = a_bang * (T_bang * dt)**2

# Define the acceleration profile
a_cycle = np.zeros(n_cycle)
a_cycle[:T_bang] = a_bang
a_cycle[T_bang:3*T_bang] = -a_bang  
a_cycle[3*T_bang:] = a_bang

# Repeat the acceleration profile for the number of cycles
a = np.tile(a_cycle, bang_cycles)

# print('Home configuration\n', q0)
print('Joint delta: ', joint_delta)

n = len(a)
v = np.zeros((n, rmodel.nv))
q = np.full((n, rmodel.nq), q0)
q_j = q0[j]
v_j = 0

time.sleep(5)
for i in range(n):
    start = time.time()
    v_plus = v_j + 0.5 * a[i] * dt
    v_j += a[i] * dt
    q_j += v_plus * dt
    if i < n - 1:
        q[i + 1, j], v[i + 1, j] = q_j, v_j
    if i % DISPLAY_N == 0:
        viz.display(q[i])
    time.sleep(dt)

viz.display(q[-1])

# Plot the position and velocity of the selected joint
t = np.linspace(0, dt*n, n)
fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(t, q[:, j], c='r', label='q', lw=1.5)
ax[0].axhline(rmodel.lowerPositionLimit[j], c='k', ls='--', lw=1)
ax[0].axhline(rmodel.upperPositionLimit[j], c='k', ls='--', lw=1)
ax[0].set_ylabel('Pos (rad)')
ax[1].plot(t, v[:, j], c='b', label='v', lw=1.5)
ax[1].axhline(rmodel.velocityLimit[j], c='k', ls='--', lw=1)
ax[1].axhline(-rmodel.velocityLimit[j], c='k', ls='--', lw=1)
ax[1].set_ylabel('Vel (rad/s)')
ax[2].plot(t, a, c='g', label='a', lw=1.5)
ax[2].set_ylabel('Acc (rad/s^2)')
ax[2].set_xlabel('Time (s)')
plt.suptitle(f'Bang-bang control on joint {j}')
plt.show()
