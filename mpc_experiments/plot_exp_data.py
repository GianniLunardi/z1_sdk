import numpy as np
import matplotlib.pyplot as plt
from safe_mpc.utils import apply_rc_params, BUFFER_SIZE


log = np.load('exp_receding/2025-03-20_18-05-50.npz')["log"]

N = len(log)
t = np.arange(0, N) * 1e-2
mpc_pos = log[:, :6]
mpc_vel = log[:, 6:12]
joint_pos = log[:, 12:18]
joint_vel = log[:, 18:24]
ee_ref = log[:, 24:27] 
solver_time = log[:, 27]
tot_time = log[:, 28] 
mpc_acc = log[:, 29:35] 
joint_tau = log[:, 35:] 

# POS
fig, ax = plt.subplots(6,1, figsize=(20, 8), sharex=True)
for i in range(6):
    ax[i].plot(t, mpc_pos[:, i], label='MPC', c='b')
    ax[i].plot(t, joint_pos[:, i], label='Meas', c='r', ls='--')
    ax[i].set_ylabel(f'q_{i}')
    ax[i].legend()
ax[-1].set_xlabel('Time (s)')

# VEL
fig, ax = plt.subplots(6,1, figsize=(20, 8), sharex=True)
# ax.reshape(6)
for i in range(6):
    ax[i].plot(t, mpc_vel[:, i], label='MPC', c='b')
    ax[i].plot(t, joint_vel[:, i], label='Meas', c='r', ls='--')
    ax[i].set_ylabel(f'v_{i}')
    ax[i].legend()
ax[-1].set_xlabel('Time (s)')

# EE REF
fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i in range(3):
    ax[i].plot(t, ee_ref[:, i], label='ee', c='b')
    ax[i].set_ylabel(f'ee_{i}')
    ax[i].legend()
ax[-1].set_xlabel('Time (s)')

# TIME
first_nan = np.where(np.isnan(solver_time))[0][0]
solver_time = solver_time[:first_nan - 1] * 1e3
tot_time = tot_time[:first_nan - 1] * 1e3
plt.figure(figsize=(10, 8))
plt.boxplot([solver_time, tot_time], labels=["Solver", "Total"])
plt.ylabel("Time (ms)")   

print(f'99th percentile of solver time: {np.quantile(solver_time, 0.99)} ms')
print(f'99.9th percentile of solver time: {np.quantile(solver_time, 0.999)} ms')
print(f'Max solver time: {np.max(solver_time)} ms')
print(f'99th percentile of total time: {np.quantile(tot_time, 0.99)} ms')
print(f'99.9th percentile of total time: {np.quantile(tot_time, 0.999)} ms')
print(f'Max total time: {np.max(tot_time)} ms')

# ACC
fig, ax = plt.subplots(6, 1, figsize=(20, 8), sharex=True)
# ax.reshape(6)
for i in range(6):
    ax[i].plot(t, mpc_acc[:, i], label='MPC', c='b')
    ax[i].set_ylabel(f'a_{i}')
    ax[i].legend()
ax[-1].set_xlabel('Time (s)')

# ACC
fig, ax = plt.subplots(6, 1, figsize=(20, 8), sharex=True)
# ax.reshape(6)
for i in range(6):
    ax[i].plot(t, joint_tau[:, i], label='Meas', c='b')
    ax[i].set_ylabel(f'tau_{i}')
    ax[i].legend()
ax[-1].set_xlabel('Time (s)')

plt.show()