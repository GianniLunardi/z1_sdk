import numpy as np
import matplotlib.pyplot as plt
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import get_ocp, get_controller, ee_ref, obstacles, \
                           capsules, capsule_pairs
from safe_mpc.controller import SafeBackupController


# log = np.load('exp_receding/2025-03-20_17-46-12.npz')["log"]
# log = np.load('exp_receding/2025-03-20_17-55-00.npz')["log"]
# log = np.load('exp_receding/2025-03-20_17-59-09.npz')["log"]
# log = np.load('exp_receding/2025-03-20_18-03-37.npz')["log"]
# log = np.load('exp_receding/2025-03-20_18-05-50.npz')["log"]
log = np.load('exp_receding/2025-03-20_18-07-35.npz')["log"]        # 2
# log = np.load('exp_receding/2025-03-20_18-09-55.npz')["log"]
# log = np.load('exp_receding/2025-03-20_18-12-05.npz')["log"]
# log = np.load('exp_receding/2025-03-20_18-13-04.npz')["log"]

ee_log = log[:, 24:27]
n = len(ee_log[~np.isnan(ee_log[:,0])])

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True, filename='config.yaml')
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=6)
nq = model.nq
model.ee_ref = ee_ref

cont_name = 'receding'
ocp = get_ocp(cont_name, model, obstacles, capsules, capsule_pairs)
opti = ocp.opti
# Options for the initial guess
opts = {
        'ipopt.print_level': 5,
        'print_time': 0,
        'ipopt.tol': 1e-6,
        'ipopt.constr_viol_tol': 1e-6,
        'ipopt.compl_inf_tol': 1e-6,
        'ipopt.linear_solver': 'ma57',
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.max_iter': params.nlp_max_iter
        }
opti.solver('ipopt', opts)  
controller = get_controller(cont_name, model, obstacles, capsules, capsule_pairs)
safe_ocp = SafeBackupController(model, obstacles, capsules, capsule_pairs)

q0_real = np.array([0., 0.26178, -0.26178, 0., 0., 0.])
q0 = q0_real[:nq]
x0 = np.zeros(model.nx)
x0[:nq] = q0

# Initial guess
print('\n', '*'*5, 'WARM START', '*'*5, '\n')
N = params.N
opti.set_value(ocp.x_init, x0)
for k in range(N):
    opti.set_initial(ocp.X[k], x0)
    opti.set_initial(ocp.U[k], np.zeros(model.nu))
opti.set_initial(ocp.X[-1], x0)

try:
    sol = opti.solve()
    xg = np.array([sol.value(ocp.X[k]) for k in range(params.N + 1)])
    ug = np.array([sol.value(ocp.U[k]) for k in range(params.N)])
except:
    sol = opti.debug
    exit()

# MPC loop
print('\n *** MPC loop *** \n')
x = x0
controller.setGuess(xg, ug)
controller.setReference(ee_ref)
ia, sa_flag = 0, False

x_log = np.empty((n, model.nx)) * np.nan
u_log = np.empty((n, model.nu)) * np.nan
r_log = np.zeros(n) 

for i in range(n):
    if i % 1000 == 0:
        print(f'Time elapsed: {i // 100} s')

    if i % 10 == 0:
        controller.setReference(ee_log[i])

    if sa_flag and ia < safe_ocp.N:
        u = u_abort[ia]
        ia += 1
    else:
        u, sa_flag = controller.step(x)
        x_next, _ = model.integrate(x, u)
        

        if sa_flag:
            print(f'  ABORT at step {i}, u = {u}')
            x_viable = controller.getLastViableState()
            xg = np.full((safe_ocp.N + 1, model.nx), x_viable)
            ug = np.zeros((safe_ocp.N, model.nu))
            safe_ocp.setGuess(xg, ug) 
            status = safe_ocp.solve(x_viable)
            if status != 0:
                print('  SAFE ABORT FAILED')
                print('  Current controller fails:', controller.fails)
                np.save('timings.npy')
                break
            ia = 0 
            u_abort = safe_ocp.u_temp

    # Check next state bounds and collision
    if not model.checkStateConstraints(x_next):   
        print('  FAIL BOUNDS')
        print(f'\tState {i + 1} violation: {np.min(np.vstack((model.x_max - x_next, x_next - model.x_min)), axis=0)}')
        print(f'\tCurrent controller fails: {controller.fails}')
        break
    # if not controller.checkCollision(x_next):
    #     print('  FAIL COLLISION')
    #     break

    # Log 
    x_log[i] = x
    u_log[i] = u
    r_log[i] = controller.r

    x = x_next
    i += 1


# PLOTS
t = np.arange(0, n) * params.dt
t_exp = np.arange(0, len(log)) * params.dt
mpc_pos = log[:, :6]
mpc_vel = log[:, 6:12]
mpc_acc = log[:, 29:35] 

# fig, ax = plt.subplots(6,1, figsize=(20, 8), sharex=True)
# for i in range(6):
#     ax[i].plot(t_exp, mpc_pos[:, i], label='MPC exp', c='b')
#     ax[i].plot(t, x_log[:, i], label='MPC simu', c='r', ls='--')
#     ax[i].set_ylabel(f'q_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

# fig, ax = plt.subplots(6,1, figsize=(20, 8), sharex=True)
# # ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t_exp, mpc_vel[:, i], label='MPC exp', c='b')
#     ax[i].plot(t, x_log[:, i + 6], label='MPC simu', c='r', ls='--')
#     ax[i].set_ylabel(f'v_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

# fig, ax = plt.subplots(6, 1, figsize=(20, 8), sharex=True)
# # ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t_exp, mpc_acc[:, i], label='MPC exp', c='b')
#     ax[i].plot(t, u_log[:, i], label='MPC simu', c='r', ls='--')
#     ax[i].set_ylabel(f'a_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

plt.figure()
plt.plot(t, r_log)


plt.show()