import numpy as np
import matplotlib.pyplot as plt
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.controller import RecedingController
from safe_mpc.utils import obstacles, capsules, capsule_pairs, ee_ref


CUSTOM_PARAMS = {
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 6,
    'lines.markersize': 12,
    'patch.linewidth': 2,
    'axes.grid': True,
    'axes.labelsize': 35,
    'font.family': 'serif',
    'font.size': 30,
    'text.usetex': True,
    'legend.fontsize': 20,
    'legend.loc': 'best',
    'figure.figsize': (10, 7),
    'figure.facecolor': 'white',
    'grid.linestyle': '-',
    'grid.alpha': 0.7,
    'savefig.format': 'pdf'
}

plt.rcParams.update(CUSTOM_PARAMS)


def boundViolations(x):
    pos_viol = model.x_max - x + params.tol_x
    neg_viol = x - model.x_min + params.tol_x
    viol = np.min(np.vstack((pos_viol, neg_viol)), axis=0)
    return np.min(viol)


def checkCollision(x):
    capsules_pos = []
    for capsule in controller.capsules:
        if capsule['type'] == 'moving':
            capsules_pos.append(np.array([capsule['end_points_fk_fun'](x[i] if len(x.shape)>1 else x ) for i in range(x.shape[0] if len(x.shape)>1 else 1)]))
        elif capsule['type'] == 'fixed':
            capsules_pos.append(np.array(capsule['end_points']).reshape(1,2,3,1))
    for pair in controller.capsule_pairs:
        if pair['type'] == 0:
            print('TYPE 0 -- CAPS CAPS')
            print('\tLHS: ', end="")
            print(controller.model.np_segment_dist(capsules_pos[pair['elements'][0]['index']][:,0],capsules_pos[pair['elements'][0]['index']][:,1],
                capsules_pos[pair['elements'][1]['index']][:,0],capsules_pos[pair['elements'][1]['index']][:,1])[0][0])
            print('\tRHS: ', end="")
            print((pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2)
            if not(controller.model.np_segment_dist(capsules_pos[pair['elements'][0]['index']][:,0],capsules_pos[pair['elements'][0]['index']][:,1],
                capsules_pos[pair['elements'][1]['index']][:,0],capsules_pos[pair['elements'][1]['index']][:,1]) >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                return False
        elif pair['type'] == 1:
            print('TYPE 1 -- SPEHERE CAPS')
            A_s = capsules_pos[pair['elements'][0]['index']][:,0]
            B_s = capsules_pos[pair['elements'][0]['index']][:,1]
            dists = np.array([controller.model.np_ball_segment_dist(A_s[i].flatten(),B_s[i].flatten(),pair['elements'][0]['length'],pair['elements'][1]['position']) for i in range(A_s.shape[0] if len(A_s.shape)>1 else 1)]) 
            if not(dists >= (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2).all(): 
                return False
        elif pair['type'] == 2:
            print('TYPE 2 -- BOX CAPS')
            print('\tFirst: ', end="")
            print(capsules_pos[pair['elements'][0]['index']][:,0,2][0][0])
            print('\tSecond: ', end="")
            print(capsules_pos[pair['elements'][0]['index']][:,1,2][0][0])
            print('\tLB: ', end="")
            print(pair['elements'][1]['bounds'][0])
            print('\tUB: ', end="")
            print(pair['elements'][1]['bounds'][1])

            if not(capsules_pos[pair['elements'][0]['index']][:,0,2] >=  pair['elements'][1]['bounds'][0]).all(): return False
            if not(capsules_pos[pair['elements'][0]['index']][:,0,2] <=  pair['elements'][1]['bounds'][1]).all(): return False
            if not(capsules_pos[pair['elements'][0]['index']][:,1,2] >=  pair['elements'][1]['bounds'][0]).all(): return False
            if not(capsules_pos[pair['elements'][0]['index']][:,1,2] <=  pair['elements'][1]['bounds'][1]).all(): return False
    return True


def collisionViolations(x):
    capsules_pos = []
    dist_log = []
    for capsule in controller.capsules:
        if capsule['type'] == 'moving':
            capsules_pos.append(np.array([capsule['end_points_fk_fun'](x[i] if len(x.shape)>1 else x ) for i in range(x.shape[0] if len(x.shape)>1 else 1)]))
        elif capsule['type'] == 'fixed':
            capsules_pos.append(np.array(capsule['end_points']).reshape(1,2,3,1))
    for pair in controller.capsule_pairs:
        if pair['type'] == 0:
            dist_log.append( controller.model.np_segment_dist(capsules_pos[pair['elements'][0]['index']][:,0],capsules_pos[pair['elements'][0]['index']][:,1],
                             capsules_pos[pair['elements'][1]['index']][:,0],capsules_pos[pair['elements'][1]['index']][:,1])[0][0] - (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2  + params.tol_obs)

        elif pair['type'] == 1:
            A_s = capsules_pos[pair['elements'][0]['index']][:,0]
            B_s = capsules_pos[pair['elements'][0]['index']][:,1]
            dists = np.array([controller.model.np_ball_segment_dist(A_s[i].flatten(),B_s[i].flatten(),pair['elements'][0]['length'],pair['elements'][1]['position']) for i in range(A_s.shape[0] if len(A_s.shape)>1 else 1)]) 

            dist_log.append( dists[0][0] - (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2 + params.tol_obs) 
        elif pair['type'] == 2:
            dist_log.append(  capsules_pos[pair['elements'][0]['index']][:,0,2][0][0] - pair['elements'][1]['bounds'][0] + params.tol_obs )
            dist_log.append( -capsules_pos[pair['elements'][0]['index']][:,0,2][0][0] + pair['elements'][1]['bounds'][1] + params.tol_obs )
            dist_log.append(  capsules_pos[pair['elements'][0]['index']][:,1,2][0][0] - pair['elements'][1]['bounds'][0] + params.tol_obs )
            dist_log.append( -capsules_pos[pair['elements'][0]['index']][:,1,2][0][0] + pair['elements'][1]['bounds'][1] + params.tol_obs )
    dist_log = np.array(dist_log)
    return np.min(dist_log)

exp_dict = {
    1: '2025-03-20_17-46-12',
    2: '2025-03-20_17-55-00',
    3: '2025-03-20_17-59-09',
    4: '2025-03-20_18-03-37',
    5: '2025-03-20_18-05-50',
    6: '2025-03-20_18-07-35',
    7: '2025-03-20_18-09-55',
    8: '2025-03-20_18-12-05',
    9: '2025-03-20_18-13-04',
}

exp_num = 1
log = np.load(f'exp_receding/{exp_dict[exp_num]}.npz')["log"]

n = len(log)
t = np.arange(0, n) * 1e-2
x_mpc = log[:, :12]
joint_pos = log[:, 12:18]
joint_vel = log[:, 18:24]
ee_mocap = log[:, 24:27] 
solver_time = log[:, 27]
tot_time = log[:, 28] 
mpc_acc = log[:, 29:35] 
joint_tau = log[:, 35:] 

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True, filename='config.yaml')
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=6)
nq = model.nq
model.ee_ref = ee_ref
controller = RecedingController(model, obstacles, capsules, capsule_pairs)

bound_viol = np.empty(n) * np.nan
coll_viol = np.empty(n) * np.nan
nn_out = np.empty(n) * np.nan
for i in range(n):
    bound_viol[i] = boundViolations(x_mpc[i])
    coll_viol[i] = collisionViolations(x_mpc[i])
    nn_out[i] = float(model.nn_func(x_mpc[i], params.alpha))

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

sort_solver_time = np.sort(solver_time)
sort_tot_time = np.sort(tot_time)
cp_solver = np.arange(1, len(sort_solver_time) + 1) / len(sort_solver_time)
cp_tot = np.arange(1, len(sort_tot_time) + 1) / len(sort_tot_time)

# Plot
plt.figure(figsize=(10, 8))
plt.step(sort_solver_time, cp_solver, c='b', where='post')
plt.step(sort_tot_time, cp_tot, c='r', where='post')
plt.xlabel("Computation time (ms)")
plt.ylabel("Cumulative Percentage")
plt.ylim(0, 1.05)
plt.xlim(left=0)  
plt.savefig(f'exp_receding/exp_timings_{exp_num}', bbox_inches='tight')


fig, ax = plt.subplots(3, 1, figsize=(10, 20), sharex=True)
ax[0].plot(t, bound_viol, c='b')
ax[0].set_ylabel('Bounds')
ax[1].plot(t, coll_viol, c='r')
ax[1].set_ylabel('Obstacles')
ax[2].plot(t, nn_out, c='g')
ax[2].set_ylabel('NN')
ax[2].set_xlabel('Time (s)')
plt.savefig(f'exp_receding/exp_violations_{exp_num}', bbox_inches='tight')

plt.show()


###############
#### EXTRA ####
###############


# # POS
# fig, ax = plt.subplots(6,1, figsize=(20, 8), sharex=True)
# for i in range(6):
#     ax[i].plot(t, x_mpc[:, i], label='MPC', c='b')
#     ax[i].plot(t, joint_pos[:, i], label='Meas', c='r', ls='--')
#     ax[i].set_ylabel(f'q_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

# # VEL
# fig, ax = plt.subplots(6,1, figsize=(20, 8), sharex=True)
# # ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, x_mpc[:, i + nq], label='MPC', c='b')
#     ax[i].plot(t, joint_vel[:, i], label='Meas', c='r', ls='--')
#     ax[i].set_ylabel(f'v_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

# # EE REF
# fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
# for i in range(3):
#     ax[i].plot(t, ee_mocap[:, i], label='ee', c='b')
#     ax[i].set_ylabel(f'ee_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

# # ACC
# fig, ax = plt.subplots(6, 1, figsize=(20, 8), sharex=True)
# # ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, mpc_acc[:, i], label='MPC', c='b')
#     ax[i].set_ylabel(f'a_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')

# # TORQUE
# fig, ax = plt.subplots(6, 1, figsize=(20, 8), sharex=True)
# # ax.reshape(6)
# for i in range(6):
#     ax[i].plot(t, joint_tau[:, i], label='Meas', c='b')
#     ax[i].set_ylabel(f'tau_{i}')
#     ax[i].legend()
# ax[-1].set_xlabel('Time (s)')