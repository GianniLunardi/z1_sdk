import numpy as np
import casadi as cs
from . import z1_conf as params


class NaiveOCP:
    """ Define OCP problem and solver (IpOpt) """
    def __init__(self, model, obstacles=None):
        self.model = model
        self.nq = model.nq
        self.obstacles = obstacles

        N = params.N
        opti = cs.Opti()
        x_init = opti.parameter(model.nx)
        cost = 0

        # Define decision variables
        X, U = [], []
        X += [opti.variable(model.nx)]
        for k in range(N):
            X += [opti.variable(model.nx)]
            opti.subject_to(opti.bounded(model.x_min, X[-1], model.x_max))
            U += [opti.variable(model.nu)]

        opti.subject_to(X[0] == x_init)
        Q = 1e2 * np.eye(3)
        R = 5e-3 * np.eye(self.model.nu)
        ee_ref = model.ee_ref
        dist_b = []
        for k in range(N + 1):
                
            ee_pos = model.ee_fun(X[k])
            cost += (ee_pos - ee_ref).T @ Q @ (ee_pos - ee_ref)

            if k < N:
                cost += U[k].T @ R @ U[k]
                # Dynamics constraint
                opti.subject_to(X[k + 1] == model.f_fun(X[k], U[k]))
                # Torque constraints
                opti.subject_to(opti.bounded(model.tau_min, model.tau_fun(X[k], U[k]), model.tau_max))

            if obstacles is not None and params.obs_flag:
                # Collision avoidance
                for obs in obstacles:
                    ee_pos = model.ee_fun(X[k])
                    if obs['name'] == 'floor':
                        lb = obs['bounds'][0]
                        ub = obs['bounds'][1]
                        opti.subject_to(opti.bounded(lb, ee_pos[2], ub))
                    elif obs['name'] == 'ball':
                        lb = obs['bounds'][0]
                        ub = obs['bounds'][1]
                        dist_b += [(ee_pos - obs['position']).T @ (ee_pos - obs['position'])]
                        opti.subject_to(opti.bounded(lb, dist_b[-1], ub))

        opti.minimize(cost)
        self.opti = opti
        self.X = X
        self.U = U
        self.x_init = x_init    
        self.cost = cost
        self.dist_b = dist_b
        self.additionalSetting()

    def additionalSetting(self):
        pass

    def checkCollision(self, x):
        if self.obstacles is not None and params.obs_flag:
            t_glob = self.model.jointToEE(x) 
            for obs in self.obstacles:
                if obs['name'] == 'floor':
                    if t_glob[2] + params.tol_obs < obs['bounds'][0]:
                        return False
                elif obs['name'] == 'ball':
                    dist_b = np.sum((t_glob.flatten() - obs['position']) ** 2)
                    if dist_b + params.tol_obs < obs['bounds'][0]:
                        return False
        return True

    def instantiateProblem(self):
        opti = self.opti
        # MPC settings
        opts = {
            'error_on_fail': False,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-4,
            'ipopt.constr_viol_tol': 1e-4,
            'ipopt.compl_inf_tol': 1e-4,
            'ipopt.linear_solver': 'ma57',
            'ipopt.hessian_approximation': 'limited-memory',
            'ipopt.max_iter': 7,
            'ipopt.sb': 'yes'
        }

        opti.solver('ipopt', opts)
        return opti