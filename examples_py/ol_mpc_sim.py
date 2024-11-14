import time
import numpy as np
from casadi_mpc import z1_conf as params
from casadi_mpc.utils import obstacles, ee_ref, RobotVisualizer
from casadi_mpc.model import Z1Model
from casadi_mpc.ocp import NaiveOCP


model = Z1Model()
model.ee_ref = ee_ref
ocp = NaiveOCP(model, obstacles)
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

# q0 = np.zeros(model.nq)
x0 = np.zeros(model.nx)


# Initial guess
print('\n', '*'*5, 'WARM START', '*'*5, '\n')
N = params.N
nq = model.nq
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

# Visualizer
print('\n', '*'*5, 'OPEN VISUALIZER', '*'*5, '\n')
rviz = RobotVisualizer()
rviz.viz.display(x0[:nq])
rviz.setTarget(ee_ref)
if params.obs_flag:
    rviz.addObstacles(obstacles)
time.sleep(5)

# MPC loop
print('\n', '*'*5, 'MPC LOOP', '*'*5, '\n')
opti = ocp.instantiateProblem()

for i in range(params.n_step):

    start_time = time.time()

    opti.set_value(ocp.x_init, xg[0])
    for k in range(N):
        opti.set_initial(ocp.X[k], xg[k])
        opti.set_initial(ocp.U[k], ug[k])
    opti.set_initial(ocp.X[-1], xg[-1])

    try:
        sol = opti.solve()
        xg = np.array([sol.value(ocp.X[k]) for k in range(params.N + 1)])
        ug = np.array([sol.value(ocp.U[k]) for k in range(params.N)])
        xg, ug = np.roll(xg, -1, axis=0), np.roll(ug, -1, axis=0)
        xg[-1] = xg[-2]
        ug[-1] = ug[-2]
    except:
        sol = opti.debug
        print(sol)
        break

    end_time = time.time()
    rviz.display(xg[0][:nq])
    delta = params.dt - (end_time - start_time)
    print(f'Iteration {i+1}/{params.n_step} - Time: {end_time - start_time:.3f}s')
    time.sleep(delta if delta > 0 else 0)