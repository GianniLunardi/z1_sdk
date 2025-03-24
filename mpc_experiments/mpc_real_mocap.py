import time
import sys
import asyncio
import qtm_rt
from datetime import datetime
sys.path.append("../lib")
import unitree_arm_interface
import numpy as np
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import get_ocp, get_controller, ee_ref, obstacles, \
                           capsules, capsule_pairs, RobotVisualizer, \
                           BUFFER_SIZE, safe_dist
from safe_mpc.controller import SafeBackupController
from pynput import keyboard


def ridig_body_data(packet):
    global last_meas 
    _, bodies = packet.get_6d()
    try:
        pos, _ = bodies[0]
        x, y, z = pos 
        last_meas = np.array([x, y, z]) * 1e-3
    except:
        print('Error, body not found')
        

async def qtm_stream():
    connection = await qtm_rt.connect("192.168.225.1")
    if connection is None:
        return

    await connection.stream_frames(components=["6d"], on_packet=ridig_body_data)


keys = []
def on_press(key):
    try:
        k = key.char
    except:
        k = key.name
    if(k in ['left', 'right', 'up', 'down', 'page_up', 'page_down', 'm', 'q']):
        keys.append(k)
listener = keyboard.Listener(on_press=on_press)
listener.start()

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"exp_data/{timestamp}.npz"

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True, filename='config.yaml')
params.build = args['build']
params.act = args['activation']
model = AdamModel(params, n_dofs=6)
nq = model.nq
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
        'ipopt.linear_solver': 'ma57',
        'ipopt.hessian_approximation': 'limited-memory',
        'ipopt.max_iter': params.nlp_max_iter
        }
opti.solver('ipopt', opts)  
controller = get_controller(cont_name, model, obstacles, capsules, capsule_pairs)
safe_ocp = SafeBackupController(model, obstacles, capsules, capsule_pairs)
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
VIZ_FLAG = 0
if VIZ_FLAG:
    print('\n', '*'*5, 'OPEN VISUALIZER', '*'*5, '\n')
    rviz = RobotVisualizer(params, nq)
    rviz.displayWithEESphere(x0[:nq], controller.capsules)
    rviz.setTarget(ee_ref)
    if params.obs_flag:
        rviz.addObstacles(obstacles)
        for capsule in controller.capsules:
                rviz.init_capsule(capsule)
time.sleep(5)

# MPC loop
print('[MPC]')
async def mpc_loop(xg, ug):

    asyncio.create_task(qtm_stream())
    await asyncio.sleep(1)

    x = x0
    controller.setGuess(xg, ug)
    ia, sa_flag = 0, False
    q_des, v_des = np.copy(q0_real), np.zeros(6)
    
    step_size = 0.02
    use_mocap = 0
    old_ref = np.copy(ee_ref)

    # LOGS
    # MPC state + Robot state + EE ref + 2 timings (solver and tot) 
    # + MPC control (acceleration) + Measured torque
    # 12 + 12 + 3 + 2 + 6 + 6 = 41
    # Slicing 
    # MPC state -> [:12]
    # Robot joint pos -> [12:18]
    # Robot joint vel -> [18:24]
    # EE ref -> [24:27]
    # Solver time -> [27]
    # Tot time -> [28]
    # MPC control -> [28:34]
    # Measured torque -> [34:]
    log_size = 41 #len(x) * 2 + 7 + len(ug[0]) * 2
    log_array = np.empty((BUFFER_SIZE, log_size)) * np.nan

    i = 0
    while 1 and i < BUFFER_SIZE:
        start_time = time.perf_counter()

        if use_mocap == 1:
            if i % 10 == 0:
                new_ref = last_meas.copy()
                if np.isnan(new_ref).any(): 
                    # Use the old one
                    new_ref = old_ref.copy()
                # if np.linalg.norm(new_ref - old_ref) > 0.1:
                #     print('STOP THE SYSTEM ... ')
                #     break

                ee_ref[0] = new_ref[0] + safe_dist[0]        # 10 cm of safety distance
                ee_ref[1] = new_ref[1] + safe_dist[1]
                ee_ref[2] = new_ref[2] + safe_dist[2]
                controller.setReference(ee_ref)
                old_ref = new_ref.copy()

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
            elif(k=="m"):
                use_mocap = 1
            elif(k=="q"):
                print("QUITTING...")
                break
            # print("\nTarget", ee_ref)
            controller.setReference(ee_ref)
            if VIZ_FLAG and i % 10 == 0:
                rviz.setTarget(ee_ref)
        except:
            pass

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

        q_des[:nq] = x_next[:nq]
        v_des[:nq] = x_next[nq:]
        arm.setArmCmd(q_des, v_des, tau_des)

        # LOG DATA
        log_array[i, :12] = x
        log_array[i, 12:18] = arm.lowstate.getQ()
        log_array[i, 18:24] = arm.lowstate.getQd()
        log_array[i, 24:27] = ee_ref
        log_array[i, 27] = controller.ocp_solver.get_stats("time_tot")
        log_array[i, 29:35] = u
        log_array[i, 35:] = arm.lowstate.getTau()[:6]

        x = x_next
        if VIZ_FLAG and i % 10 == 0:
            rviz.displayWithEESphere(x[:nq], controller.capsules)
        end_time = time.perf_counter()
        log_array[i, 28] = end_time - start_time
        i += 1

        delta = params.dt - (end_time - start_time)
        # time.sleep(delta if delta > 0 else 0)
        await asyncio.sleep(delta if delta > 0 else 0)

    print('[BACK]')
    time.sleep(2)
    arm.backToStart()
    arm.loopOff()

    # first_nan = np.where(np.isnan(log_array))[0][0]
    # log_array = log_array[:first_nan - 1, :]
    np.savez_compressed(filename, log=log_array)


asyncio.run(mpc_loop(xg, ug))
print('Finish process')