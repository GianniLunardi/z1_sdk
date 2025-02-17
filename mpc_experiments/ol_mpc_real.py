import time
import sys
sys.path.append("../lib")
import unitree_arm_interface
import numpy as np
from utils import ee_ref, obstacles, RobotVisualizer
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import get_ocp, get_controller
from safe_mpc.controller import SafeBackupController
from pynput import keyboard

keys = []
def on_press(key):
    try:
        k = key.char
    except:
        k = key.name
    if(k in ['left', 'right', 'up', 'down', 'page_up', 'page_down', 'q']):
        keys.append(k)
listener = keyboard.Listener(on_press=on_press)
listener.start()

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True, filename='casadi_mpc/config.yaml')
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=6)
nq = model.nq
model.ee_ref = ee_ref

cont_name = args['controller']
ocp = get_ocp(cont_name, model, obstacles)
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
controller = get_controller(cont_name, model, obstacles)
params.solver_type = 'SQP'
safe_ocp = SafeBackupController(model, obstacles)
if args['build']:
    print('*** Ready for running the MPC at the next launch ***')
    exit()

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

# Prepare the robot
print("Press ctrl+\ to quit process.")

arm =  unitree_arm_interface.ArmInterface(hasGripper=True)
armState = unitree_arm_interface.ArmFSMState
arm.loopOn() 

arm.startTrack(armState.JOINTCTRL)
q_init = arm.lowstate.getQ()
dt = arm._ctrlComp.dt

print('[PRE]')
n_pre = 500
v_des = (q0_real - q_init) / (n_pre * dt)
tau_des = np.zeros(6)
for i in range(n_pre):
    q_des = q_init * (1 - i / n_pre) + q0_real * (i / n_pre)
    arm.setArmCmd(q_des, v_des, tau_des)
    time.sleep(dt)
arm.setArmCmd(q0_real, np.zeros(6), tau_des)
time.sleep(2)

# Visualizer
print('\n', '*'*5, 'OPEN VISUALIZER', '*'*5, '\n')
rviz = RobotVisualizer()
rviz.viz.display(x0[:nq])
rviz.setTarget(ee_ref)
if params.obs_flag:
    rviz.addObstacles(obstacles)
time.sleep(5)

# MPC loop
print('[MPC]')
x = x0
controller.setGuess(xg, ug)
ia, sa_flag = 0, False
tot_time, solver_time = np.nan*np.zeros(params.n_steps), np.nan*np.zeros(params.n_steps)
q_des, v_des = np.copy(q0_real), np.zeros(6)

step_size = 0.02
omega = 6.28*1.5
amp = 0.07
t = 0.0
sin_ref = np.copy(ee_ref)
use_sinusoid = 0

i = 0
q_log, v_log = [], []
while 1:
    start_time = time.time()

    if(use_sinusoid):
        sin_ref[0] = ee_ref[0]
        sin_ref[1] = ee_ref[1] + amp*np.sin(omega*t)
        sin_ref[2] = ee_ref[2] + amp*np.cos(omega*t)
        controller.setReference(sin_ref)
        t += dt
        if(i%10==0):
            rviz.setTarget(sin_ref)
            #rviz.display(x[:nq])

    try:
        k = keys.pop(0)
        if(k=="up"):      
            ee_ref[0] -= step_size
        elif(k=="down"):   
            ee_ref[0] += step_size
        elif(k=="right"):      
            ee_ref[1] += step_size
        elif(k=="left"):   
            ee_ref[1] -= step_size
        elif(k=="page_up"):      
            ee_ref[2] += step_size
        elif(k=="page_down"):   
            ee_ref[2] -= step_size
        elif(k=="q"):
            print("QUITTING...")
            break
        print("\nTarget", ee_ref)
        if(not use_sinusoid):
            controller.setReference(ee_ref)
        rviz.setTarget(ee_ref)
    except:
        pass

    u, sa_flag = controller.step(x)
    x_next, _ = model.integrate(x, u)

    # ATTENTION: skip all the checks
    q_des[:nq] = x_next[:nq]
    v_des[:nq] = x_next[nq:]
    arm.setArmCmd(q_des, v_des, tau_des)

    x = x_next
    if i % 10 == 0:
        rviz.display(x[:nq])
    end_time = time.time()

    if(i<params.n_steps):
        solver_time[i] = controller.ocp_solver.get_stats("time_tot")
        tot_time[i] = end_time - start_time
        i += 1

    # Log data
    q_log.append(arm.lowstate.getQ())
    v_log.append(arm.lowstate.getQd())

    delta = params.dt - (end_time - start_time)
    time.sleep(delta if delta > 0 else 0)

print('[BACK]')
time.sleep(2)
arm.backToStart()
arm.loopOff()

print('TIMINGS')
tot_time = np.asarray(tot_time[:i])
solver_time = np.asarray(solver_time[:i])
print(f'99 percentile, tot = {np.quantile(tot_time, 0.99):.3f}s, '
      f'solver = {np.quantile(solver_time, 0.99):.3f}')
print(f'Max time, tot = {max(tot_time):.3f}s, '
      f'solver = {max(solver_time):.3f}')

# Save data
np.savez_compressed('data/mpc_exp.npz', q=q_log, v=v_log)