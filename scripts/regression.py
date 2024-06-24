import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from orc.utils import plot_utils
from urdf_parser_py.urdf import URDF
import adam
from adam.numpy import KinDynComputations


def computed_torque(q_m, qd_m, qdd_m):
    M = kin_dyn.mass_matrix(H_b, q_m)[6:, 6:]
    b = kin_dyn.bias_force(H_b, q_m, np.zeros(6), qd_m)[6:]
    return M @ qdd_m + b


def bias(q_m, qdd_m, tau_m):
    # Compute tau_j - M_jj * qdd_j - g_j(q)
    M = kin_dyn.mass_matrix(H_b, q_m)[6:, 6:]
    g = kin_dyn.gravity_term(H_b, q_m)[6:]
    return tau_m[j] - M[j, j] * qdd_m[j] - g[j]
    

def regressor(qd_m, qdd_m):
    # Compute the regressor vector
    return np.array([qdd_m[j], qd_m[j], np.tanh(1e3 * qd_m[j])])


j = 0 if len(sys.argv) < 2 else int(sys.argv[1])
if j < 0 or j > 5:
    print('Invalid joint index. Please provide a joint index between 0 and 5.')
    sys.exit(1)

dt = 2e-3
state_log = pd.read_csv(f'../data/state_log_{j}.csv')
filt_log = np.load(f'../data/filt_log_{j}.npz')
state_array = state_log.to_numpy()
q = state_array[:, 0:6]
n = len(q)
qd = np.zeros((len(q), 6))
qdd, tau = np.copy(qd), np.copy(qd)
qd[:, j], qdd[:, j], tau[:, j] = filt_log['qd'], filt_log['qdd'], filt_log['tau']

urdf_name = '../z1.urdf'
robot = URDF.from_xml_file(urdf_name)

robot_joints = robot.joints[1:]         # skip the root joint
joint_names = [joint.name for joint in robot_joints]
kin_dyn = KinDynComputations(urdf_name, joint_names, robot.get_root())
kin_dyn.set_frame_velocity_representation(adam.Representations.BODY_FIXED_REPRESENTATION)
H_b = np.eye(4)                         # Roto-translation world --> base/root


tau_comp = np.zeros(n)
for i in range(n):
    tau_comp[i] = computed_torque(q[i], qd[i], qdd[i])[j]

# Compute the regressor matrix and the bias vector
bias_vector = np.zeros((n, 1))
regressor_matrix = np.zeros((n, 3))
for i in range(n):
    temp =bias(q[i], qdd[i], tau[i])
    bias_vector[i] = bias(q[i], qdd[i], tau[i])
    regressor_matrix[i] = regressor(qd[i], q[i])
# Solve the LS problem
params = np.linalg.pinv(regressor_matrix) @ bias_vector
# Order --> [k^2 * I_r, F_v, F_s]
print('Regressor parameters: ', params)

tau_w_fric = np.copy(tau_comp)
tau_w_fric += params[1] * qd[:, j] + params[2] * np.sign(qd[:, j])

t = np.linspace(0, n * dt, n)
plt.figure()
plt.plot(t, tau[:, j], label='Measured torque')
plt.plot(t, tau_comp, label='Computed torque', c='r', ls='--')
plt.plot(t, tau_w_fric, label='With friction', c='g', ls='--')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title(f'Joint {j} Torque comparison')


plt.show()
