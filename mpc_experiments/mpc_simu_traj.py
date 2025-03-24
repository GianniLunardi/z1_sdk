import time
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import get_ocp, get_controller, RobotVisualizer, \
                           obstacles, capsules, capsule_pairs
from safe_mpc.controller import SafeBackupController


def lemniscate(t):
    x = a * np.sin(omega * t) 
    y = a * np.sin(omega * t) * np.cos(omega * t)
    z = b * np.cos(omega * t) - b
    return np.array([x, y, z])

def twisted_lemniscate(t):
    x = a * np.cos(omega * t) 
    y = a * np.sin(omega * 2 * t)
    z = b * np.sin(omega * t)
    return np.array([x, y, z])

def mobius_inspired(t):
    x = a * np.cos(omega * t)
    y = b * np.sin(omega * t) * np.cos(omega * t / 2)
    z = b * np.sin(omega * t) * np.sin(omega * t / 2)
    return np.array([x, y, z])


args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True, filename='config.yaml')
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=6)
nq = model.nq

q0 = np.array([0., 0.26178, -0.26178, 0., 0., 0.])
q0 = q0[:nq]
x0 = np.zeros(model.nx)
x0[:nq] = q0

use_3d_curve = 1
a, b, omega = 0.1, 0.1, 2 * np.pi * 0.1
if use_3d_curve:
    ee_ref = np.array([0.42, 0., 0.378])
    if use_3d_curve == 1:
        ee_ref += lemniscate(0.)
    elif use_3d_curve == 2:
        ee_ref += twisted_lemniscate(0.)
    else:
        ee_ref += mobius_inspired(0.)
model.ee_ref = ee_ref

cont_name = args['controller']
ocp = get_ocp(cont_name, model, obstacles, capsules, capsule_pairs)
opti = ocp.opti
# Options for the initial guess
opts = {
        'ipopt.print_level': 5,
        'print_time': 0,
        'ipopt.tol': 1e-6,
        'ipopt.constr_viol_tol': 1e-6,
        'ipopt.compl_inf_tol': 1e-6,
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.max_iter': params.nlp_max_iter
        }
opti.solver('ipopt', opts)  
controller = get_controller(cont_name, model, obstacles, capsules, capsule_pairs)
safe_ocp = SafeBackupController(model, obstacles, capsules, capsule_pairs)
if args['build']:
    print('*** Ready for running the MPC at the next launch ***')
    exit()

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

# Visualizer
print('\n', '*'*5, 'OPEN VISUALIZER', '*'*5, '\n')
rviz = RobotVisualizer(params, nq)
rviz.viz.display(x0[:nq])
rviz.setTarget(ee_ref)
if params.obs_flag:
    rviz.addObstacles(obstacles)
time.sleep(5)

# MPC loop
x = np.empty((params.n_steps + 1, model.nx)) * np.nan
u = np.empty((params.n_steps, model.nu)) * np.nan
x[0] = x0
controller.setGuess(xg, ug)
controller.resetHorizon(params.N)
ia, sa_flag = 0, False
tot_time, solver_time = np.zeros(params.n_steps), np.zeros(params.n_steps)
for i in range(params.n_steps):

    start_time = time.time()

    if use_3d_curve:
        if use_3d_curve == 1:
            curve_ref = lemniscate(i * params.dt)
        elif use_3d_curve == 2:
            curve_ref = twisted_lemniscate(i * params.dt)
        else:
            curve_ref = mobius_inspired(i * params.dt)
        curve_ref += ee_ref
        controller.setReference(curve_ref)
        if i % 5 == 0:
            rviz.setTarget(curve_ref)
    else:

        if i == 500:
            ee_ref = np.array([0.3, 0., 0.478])
            controller.setReference(ee_ref)
            rviz.setTarget(ee_ref)

        if i == 1000:
            ee_ref = np.array([0.2, -0.28, 0.378])
            controller.setReference(ee_ref)
            rviz.setTarget(ee_ref)

    if sa_flag and ia < safe_ocp.N:
        u[i] = u_abort[ia]
        ia += 1
    else:
        u[i], sa_flag = controller.step(x[i])
        x[i + 1], _ = model.integrate(x[i], u[i])

        if sa_flag:
            print(f'  ABORT at step {i}, u = {u[i]}')
            x_viable = controller.getLastViableState()
            xg = np.full((safe_ocp.N + 1, model.nx), x_viable)
            ug = np.zeros((safe_ocp.N, model.nu))
            safe_ocp.setGuess(xg, ug) 
            status = safe_ocp.solve(x_viable)
            if status != 0:
                print('  SAFE ABORT FAILED')
                print('  Current controller fails:', controller.fails)
                break
            ia = 0 
            u_abort = safe_ocp.u_temp

    # Check next state bounds and collision
    if not model.checkStateConstraints(x[i + 1]):   
        print('  FAIL BOUNDS')
        print(f'\tState {i + 1} violation: {np.min(np.vstack((model.x_max - x[i + 1], x[i + 1] - model.x_min)), axis=0)}')
        print(f'\tCurrent controller fails: {controller.fails}')
        break
    if not controller.checkCollision(x[i + 1]):
        print('  FAIL COLLISION')
        break
    end_time = time.time()

    delta = params.dt - (end_time - start_time)
    solver_time[i] = controller.ocp_solver.get_stats("time_tot")
    tot_time[i] = end_time - start_time
    if i % 3 == 0:
        print(f'Iteration {i+1}/{params.n_steps} - '
              f'Time: {tot_time[i]:.3f}s - '
                f'Total solver time {solver_time[i]:.3f}s \n')
        rviz.display(x[i][:nq])

    time.sleep(delta if delta > 0 else 0)

print('TIMINGS')
tot_time = np.asarray(tot_time)
solver_time = np.asarray(solver_time)
perc = 0.95
print(f'{int(perc * 100)} percentile, tot = {np.quantile(tot_time, perc):.3f}s, '
      f'solver = {np.quantile(solver_time, perc)}')
print(f'Max time, tot = {max(tot_time):.3f}s, '
      f'solver = {max(solver_time)}')

# Save data
np.savez_compressed('data/mpc_simu.npz', x=x, u=u)