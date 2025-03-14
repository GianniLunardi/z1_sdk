import time
import numpy as np
from pynput import keyboard
import multiprocessing as mp    
from utils import ee_ref, obstacles, RobotVisualizer
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import get_ocp, get_controller
from safe_mpc.controller import SafeBackupController


def run_mpc(queue, xg, ug):

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

    x = x0
    controller.setGuess(xg, ug)
    controller.resetHorizon(params.N)
    ia, sa_flag = 0, False

    step_size = 0.02
    omega = 6.28*1.5
    amp = 0.07
    t = 0.0
    sin_ref = np.copy(ee_ref)
    use_sinusoid = 0

    i = 0
    while 1:

        start_time = time.time()

        if(use_sinusoid):
            sin_ref[0] = ee_ref[0]
            sin_ref[1] = ee_ref[1] + amp*np.sin(omega*t)
            sin_ref[2] = ee_ref[2] + amp*np.cos(omega*t)
            controller.setReference(sin_ref)
            t += params.dt

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
                    break
                ia = 0 
                u_abort = safe_ocp.u_temp

        # Check next state bounds and collision
        if not model.checkStateConstraints(x_next):   
            print('  FAIL BOUNDS')
            print(f'\tState {i + 1} violation: {np.min(np.vstack((model.x_max - x_next, x_next - model.x_min)), axis=0)}')
            print(f'\tCurrent controller fails: {controller.fails}')
            break
        if not controller.checkCollision(x_next):
            print('  FAIL COLLISION')
            break
        
        x = x_next

        end_time = time.time()
        delta = params.dt - (end_time - start_time)
        time.sleep(delta if delta > 0 else 0)
        ref = sin_ref if use_sinusoid else ee_ref
        if not queue.full():
            queue.put((x[:nq], ref)) 
        i += 1

def run_visualizer(queue):
    rviz = RobotVisualizer(nq)
    rviz.display(x0[:nq])
    if params.obs_flag:
        rviz.addObstacles(obstacles)
    while 1:
        if not queue.empty():
            x, ref = queue.get()
            rviz.display(x)
            rviz.setTarget(ref)
        time.sleep(0.01)

if __name__ == "__main__":

    args = parse_args()
    model_name = args['system']
    params = Parameters(model_name, rti=True, filename='config.yaml')
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

    q0 = np.array([0., 0.26178, -0.26178, 0., 0., 0.])
    q0 = q0[:nq]
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

    queue = mp.Queue(maxsize=1)  

    # Create processes
    mpc_process = mp.Process(target=run_mpc, args=(queue, xg, ug))
    viz_process = mp.Process(target=run_visualizer, args=(queue,))

    # Start processes
    mpc_process.start()
    viz_process.start()

    # Wait until MPC finish
    mpc_process.join()

    # Terminate the visualizer
    viz_process.terminate()
    viz_process.join()

    print('*** END ***')