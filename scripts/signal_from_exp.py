import sys
import adam.numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from orc.utils import plot_utils
import scipy.signal as signal
from scipy.fftpack import fft
from urdf_parser_py.urdf import URDF
import adam
sys.path.append("../lib")
import unitree_arm_interface
import pinocchio as pin 


def filtering_signal(ts, order, cut_off):
    b, a = signal.butter(order, cut_off / nyq, 'low', analog=False)
    return signal.filtfilt(b, a, ts), fft(ts)


j = 0 if len(sys.argv) < 2 else int(sys.argv[1])
if j < 0 or j > 5:
    print('Invalid joint index. Please provide a joint index between 0 and 5.')
    sys.exit(1)

urdf_name = '../z1_description/urdf/z1.urdf'
robot = URDF.from_xml_file(urdf_name)

arm =  unitree_arm_interface.ArmInterface(hasGripper = True)
armModel = arm._ctrlComp.armModel

robot_joints = robot.joints[1:7]         # skip the root joint
joint_names = [joint.name for joint in robot_joints]
kin_dyn = adam.numpy.KinDynComputations(urdf_name, joint_names, robot.get_root())
kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
H_b = np.eye(4)                         # Roto-translation world --> base/root
    
folder = '../data_bang/'
state_log = pd.read_csv(f'{folder}state_log_{j}.csv')
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

qdd = np.zeros((n, 6))
qdd[:, j] = np.gradient(qd_filt, dt)
qd_raw = np.copy(qd[:, j])
qd[:, j] = qd_filt

np.savez(f'{folder}filt_log_{j}.npz', qd=qd_filt, qdd=qdd[:, j], tau=tau_filt)

PLOT_POS = 0
PLOT_VEL = 0
PLOT_TORQUE = 0
PLOT_FILT = 0
PLOT_EE = 0

rmodel = pin.buildModelFromUrdf(urdf_name)
rdata = rmodel.createData()

frame_name = 'link06'
ee_z1, ee_adam = np.zeros((n, 3)), np.zeros((n, 3))
tau_z1, tau_adam = np.zeros((n, 6)), np.zeros((n, 6))
for i in range(n):
    # Forward kinematics
    H_ee = kin_dyn.forward_kinematics(frame_name, H_b, q[i])
    ee_adam[i] = H_ee[:3, 3]
    H_ee = armModel.forwardKinematics(q[i], 5)
    ee_z1[i] = H_ee[:3, 3]

    # Inverse dynamics
    tau_adam[i] = kin_dyn.mass_matrix(H_b, q[i])[6:, 6:] @ qdd[i] + \
                  kin_dyn.bias_force(H_b, q[i], np.zeros(6), qd[i])[6:]
    tau_z1[i] = armModel.inverseDynamics(q[i], qd[i], qdd[i], np.zeros(6))

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
        ax[i].plot(t, tau[:, i], label='tau_' + str(i))
        ax[i].plot(t, tau_z1[:, i], label='z1_' + str(i), c='g', ls='--')
        ax[i].plot(t, tau_adam[:, i], label='adam_' + str(i), c='r', ls='--')
        ax[i].set_ylabel(fr'$\tau_{i}$ (Nm)')
        ax[i].set_xlabel('Time (s)')
        ax[i].legend()

if PLOT_FILT:
    plt.figure()
    plt.plot(t, qd_raw, label='raw')
    plt.plot(t, qd_filt, label='filtered', c='r', ls='--')
    plt.plot(t, qdd[:, j], label='qdd_fd', c='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (rad/s)')
    plt.legend()
    plt.savefig(f'vel_acc_{j}.png')

    plt.figure()
    plt.plot(t, tau[:, j], label='raw')
    plt.plot(t, tau_filt, label='filtered', c='g', ls='--')
    plt.plot(t, tau_adam[:, j], label='adam_' + str(j), c='r', ls='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')

    plt.figure()
    plt.stem(freq, np.abs(tau_fft),'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')

if PLOT_EE:
    fig, ax = plot_utils.create_empty_figure(3, 1)
    for i in range(3):
        ax[i].plot(t, ee_z1[:, i], label='z1_' + str(i), c='g', ls='--')
        ax[i].plot(t, ee_adam[:, i], label='adam_' + str(i), c='r', ls='--')
        ax[i].set_ylabel(f'EE_{i} (m)')
        ax[i].set_xlabel('Time (s)')
        ax[i].legend()


plt.show()