import os 
import numpy as np


ROOT_DIR = os.path.expanduser('~') + '/unitree_ws/src/z1_sdk'
description_dir = ROOT_DIR + '/z1_description' 
robot_urdf = description_dir + '/urdf/z1.urdf'
frame_name = 'gripperMover'
urdf_name = 'z1'

dt = 0.01

tol_x = 5e-3               # state -4
tol_tau = 1e-6             # torque
tol_dyn = 1e-6             # dynamics
tol_obs = 1e-4             # obstacle tolerance -6
tol_nn = 1e-4 

tol_conv = 1e-3            # convergence tolerance for the task
tol_cost = 1e-3

N = 6
n_step = 500
obs_flag = True
nlp_max_iter = 100