import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from orc.utils import plot_utils
import example_robot_data
import scipy.signal as signal
from scipy.fftpack import fft


def filtering_signal(ts, order, cut_off):
    b, a = signal.butter(order, cut_off / nyq, 'low', analog=False)
    return signal.filtfilt(b, a, ts), fft(ts)


j = 0 if len(sys.argv) < 2 else int(sys.argv[1])
if j < 0 or j > 5:
    print('Invalid joint index. Please provide a joint index between 0 and 5.')
    sys.exit(1)
    
state_log = pd.read_csv(f'../data/state_log_{j}.csv')
state_array = state_log.to_numpy()
q = state_array[:, 0:6]
qd = state_array[:, 6:12]
tau = state_array[:, -6:]

dt = 2e-3
n = len(q)
t = np.linspace(0, n * dt, n)

# Filter the velocity signal
fs = 1 / dt
nyq = 0.5 * fs
freq = np.fft.fftfreq(n, dt)

qd_filt, qd_fft = filtering_signal(qd[:, j], order=2, cut_off=2)
tau_filt, tau_fft = filtering_signal(tau[:, j], 2, 2)

qdd = np.gradient(qd_filt, dt)

np.savez(f'../data/filt_log_{j}.npz', qd=qd_filt, qdd=qdd, tau=tau_filt)

PLOT_POS = 0
PLOT_VEL = 0
PLOT_TORQUE = 0
PLOT_FILT = 0

robot = example_robot_data.load('z1')   
rmodel = robot.model
rdata = rmodel.createData()

if PLOT_POS:
    fig, ax = plot_utils.create_empty_figure(3, 2)
    ax = ax.reshape(6)
    for i in range(6):
        ax[i].axhline(rmodel.lowerPositionLimit[i], c='k', ls='--', lw=1)
        ax[i].axhline(rmodel.upperPositionLimit[i], c='k', ls='--', lw=1)
        ax[i].plot(t, q[:, i], label='q' + str(i))
        ax[i].set_ylabel(f'q{i} (rad)')
        ax[i].set_xlabel('Time (s)')

if PLOT_VEL:
    fig, ax = plot_utils.create_empty_figure(3, 2)
    ax = ax.reshape(6)
    for i in range(6):
        ax[i].axhline(rmodel.velocityLimit[i], c='k', ls='--', lw=1)
        ax[i].axhline(-rmodel.velocityLimit[i], c='k', ls='--', lw=1)
        ax[i].plot(t, qd[:, i], label='v' + str(i))
        ax[i].set_ylabel(f'v{i} (rad/s)')
        ax[i].set_xlabel('Time (s)')

if PLOT_TORQUE:
    fig, ax = plot_utils.create_empty_figure(3, 2)
    ax = ax.reshape(6)
    for i in range(6):
        ax[i].axhline(rmodel.effortLimit[i], c='k', ls='--', lw=1)
        ax[i].axhline(-rmodel.effortLimit[i], c='k', ls='--', lw=1)
        ax[i].plot(t, tau[:, i], label='tau' + str(i))
        ax[i].set_ylabel(fr'$\tau_{i}$ (Nm)')
        ax[i].set_xlabel('Time (s)')

if PLOT_FILT:
    plt.figure()
    plt.plot(t, qd[:, j], label='raw')
    plt.plot(t, qd_filt, label='filtered', c='r', ls='--')
    plt.plot(t, qdd, label='qdd_fd', c='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')

    plt.figure()
    plt.plot(t, tau[:, j], label='raw')
    plt.plot(t, tau_filt, label='filtered', c='r', ls='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')

    plt.figure()
    plt.stem(freq, np.abs(tau_fft),'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')


plt.show()