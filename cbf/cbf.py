#!/bin/python3
"""

Implementation of the Control Barrier Functions using a numerical
solver. Uses the obstacle list classes defined in the obstacles
module for the corresponding CBF representations.

author: Neelaksh Singh

"""

import sys
import time
import os

import numpy as np

from cvxopt import matrix, solvers
from euclid import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.obstacles import ObstacleList2D
    from cbf.utils import Timer
except:
    raise

class KBM_VC_CBF2D():
    """
    The Kinematic Bicycle Model - Velocity Controlled - Control Barrier Function.

    Details:
    -------
    Supported Obstacle State: Static
    Supported Obstacle Type : Ellipse2D
    
    Observed Safety Actions : Braking
    
    Solver Used       : CVXOPT.solvers.cp
    Class K func used : alpha * h
    """

    def __init__(self, alpha = 1.0):
        self.obstacle_list2d = ObstacleList2D()
        self.__alpha = alpha
        self.__R = matrix(np.eye(2))
        pass
    
    def update_state(self, p, theta):
        self.__p = p
        self.__theta = theta
    
    def set_alpha(self, alpha=1.0):
        self.__alpha = alpha
    
    def set_model_params(self, L):
        self.__L = L
    
    def set_qp_cost_weight(self, R):
        self.__R = matrix(R)
    
    def solve_cbf(self, u_ref):
        """
        A CVXOPT function. Thus multi-dimensional arguments should strictly be
        numpy arrays or cvxopt matrices.
        """
        m = len(self.obstacle_list2d) # No. of non-linear constraints
        n = 2 # dimension of x0 => u0
        u_ref = matrix(u_ref)
        u_ref[1] = u_ref[0] * np.tan(u_ref[1])/self.__L

        if len(self.obstacle_list2d) < 1:
            raise ValueError("Cannot solve CBF for an empty obstacle list. \
                Update the obstacle list so that it is non-empty in order  \
                to move forward.")

        def F(x = None, z=None):

            # if x is None: return m, matrix(0.0, (n, 1))
            if x is None: return m, u_ref

            # for 1 objective function and 1 constraint and 3 state vars
            f = matrix(0.0, (m+1, 1))
            Df = matrix(0.0, (m+1, n))

            f[0] = (x - u_ref).T * self.__R * (x - u_ref)
            Df[0, :] = 2 * (x - u_ref).T * self.__R

            gc = matrix([ [np.cos(self.__theta), np.sin(self.__theta), 0], [0, 0, 1] ])
            # G => Gradient, Gh -> (m,3)
            Gh = matrix([ [self.obstacle_list2d.dx(self.__p)], [self.obstacle_list2d.dy(self.__p)], [self.obstacle_list2d.dtheta(self.__p)] ])
            
            Lxg_h = Gh * gc

            f[1:] = -( Lxg_h * x + ( self.__alpha * self.obstacle_list2d.f(self.__p) ) )
            Df[1:, :] = -Lxg_h

            if z is None: return f, Df
            H = z[0] * 2 * matrix(np.eye(n))
            return f, Df, H
        
        solver_op = solvers.cp(F)
        u = solver_op['x']
        u[1] = np.arctan2(u[1] * self.__L, u_ref[0])
        return solver_op, u

class DBM_CBF_2DS():
    """
    A control barrier function for acceleration controlled dynamical bicycle model.
    Uses the approx. dynamical model with small sideslip angle approximation valid
    only for lateral acceleration < 0.5*nu*g.

    Details:
    -------
    Supported Obstacle State  : Static , Dynamic (except Ellipse)
    Supported Obstacle Type(s): Ellipse2D, CollisionCone2D, PolyLine
    
    Observed Safety Actions  : Braking, Circumventing
    
    Solver Used       : CVXOPT.solvers.cp
    Class K func used : alpha * h
    """
    def __init__(self, alpha = 1.0):
        self.obstacle_list2d = ObstacleList2D()
        self.__alpha = alpha
        self.__p = Point2()
        self.__v = 0
        self.__theta = 0
        self.__R = matrix(np.eye(2))
        pass

    def update_state(self, s: matrix, s_obs_dict: dict=None, buffer: float=None, **kwargs):
        
        self.s = s
        self.s_obs_dict = s_obs_dict
        self.__p = Point2(s[0], s[1])
        self.__theta = s[2]
        self.__v = s[3]
        
        self.obstacle_list2d.update_state(s = s, s_obs_dict = self.s_obs_dict, buffer=buffer)
    
    def set_alpha(self, alpha=1.0):
        self.__alpha = alpha
    
    def set_model_params(self, lr, lf):
        self.__lr = lr
        self.__lf = lf
    
    def set_qp_cost_weight(self, R):
        self.__R = matrix(R)
        if not (self.__R.size[0] == self.__R.size[1] or self.__R.size[0] == 2):
            raise ValueError("Expected a symmetrix matrix of size 2 as input. Please check the value of the matrix R you are using.")
        
    def gc(self, *args, **kwargs):
        return matrix([ [0, 0, 0, 1],\
                 [-self.__v * np.sin(self.__theta), self.__v * np.cos(self.__theta), self.__v/self.__lr, 0] ])
        
    def fc(self, *args, **kwargs):
        return matrix([ self.__v * np.cos(self.__theta), self.__v * np.sin(self.__theta), 0, 0], (4, 1))

    def solve_cbf(self, u_ref, return_solver = False):
        """
        A CVXOPT function. Thus multi-dimensional arguments should strictly be
        numpy arrays or cvxopt matrices.
        """
        m = len(self.obstacle_list2d) # No. of non-linear constraints
        n = 2 # dimension of x0 => u0
        u_ref = matrix(u_ref)
        # delta to beta
        u_ref[1] = np.arctan2(self.__lr * np.tan(u_ref[1]), self.__lf + self.__lr)

        if len(self.obstacle_list2d) < 1:
            raise ValueError("Cannot solve CBF for an empty obstacle list. \
                Update the obstacle list so that it is non-empty in order  \
                to move forward.")

        def F(x = None, z=None):

            # if x is None: return m, matrix(0.0, (n, 1))
            if x is None: return m, u_ref

            # for 1 objective function and 1 constraint and 3 state vars
            f = matrix(0.0, (m+1, 1))
            Df = matrix(0.0, (m+1, n))

            f[0] = (x - u_ref).T * self.__R * (x - u_ref)
            Df[0, :] = 2 * (x - u_ref).T * self.__R

            # State Equation:
            g_c = self.gc()

            f_c = self.fc()

            # G => Gradient, Gh -> (m,3)
            Gh = matrix([ [self.obstacle_list2d.dx()], [self.obstacle_list2d.dy()],\
                 [self.obstacle_list2d.dtheta()], [self.obstacle_list2d.dv()] ])
            
            Lxg_h = Gh * g_c
            Lxf_h = Gh * f_c

            f[1:] = -( Lxf_h + Lxg_h * x + ( self.__alpha * self.obstacle_list2d.f() )  + self.obstacle_list2d.dt())
            Df[1:, :] = -Lxg_h

            if z is None: return f, Df
            H = z[0] * 2 * self.__R
            return f, Df, H
        
        solver_op = solvers.cp(F)
        u = solver_op['x']
        # beta to delta
        u[1] = np.arctan2((self.__lf + self.__lr) * np.tan(u[1]), self.__lr)
        if return_solver:
            return solver_op, u
        else:
            return u

class SADBM_CBF_2DS(DBM_CBF_2DS):
    """
    A control barrier function for state augmented - steer rate controlled dynamic
    bicycle model. Removes the small angle approximation by augmenting beta to the
    state vector and treating its time derivate as the control input. By a CBF law
    beta's derivate and therefore beta is bound to be continuous. By using numerical
    differentiation schemes which are stable and convergent to obtain d(beta)/dt one
    can calculate the steer input required for steer controlled vehicles.

    Details:
    -------
    Supported Obstacle State  : Static , Dynamic (except Ellipse)
    Supported Obstacle Type(s): Ellipse2D, CollisionCone2D, PolyLine
    
    Observed Safety Actions  : Braking, Circumventing
    
    Solver Used       : CVXOPT.solvers.cp
    Class K func used : alpha * h
    """
    def __init__(self, alpha = 1.0, dt = 0.001):
        self.obstacle_list2d = ObstacleList2D()
        self.__alpha = alpha
        self.__p = Point2()
        self.__v = 0
        self.__theta = 0
        self.__R = matrix(np.eye(2))
        self.__DT_MODE_AUTO = 0
        if dt is None:
            self.__DT_MODE_AUTO = 1
            self.__dt = 1e-6
        else:
            self.__dt = dt
        self.t_last = time.time()
        self.beta_ref_last = None
        self.beta_last = None
        pass
    
    def gc(self, *args, **kwargs):
        
        return matrix([ [0, 0, 0, 1, 0], [0, 0, 0, 0, 1] ])
        
    def fc(self, *args, **kwargs):
        
        if self.beta_last is None:
            self.__beta = 0
            
        return matrix([ self.__v * np.cos(self.__theta + self.__beta), 
                       self.__v * np.sin(self.__theta + self.__beta), 
                       self.__v * np.sin(self.__beta)/self.__lr, 
                       0, 0], (4, 1))
    
    def solve_cbf(self, u_ref, return_solver = False):
        """
        A CVXOPT function. Thus multi-dimensional arguments should strictly be
        numpy arrays or cvxopt matrices.
        """
        m = len(self.obstacle_list2d) # No. of non-linear constraints
        n = 2 # dimension of x0 => u0
        u_ref = matrix(u_ref)
        # delta to beta
        u_ref[1] = np.arctan2(self.__lr * np.tan(u_ref[1]), self.__lf + self.__lr)
        
        self.t_current = time.time()
        
        if self.__DT_MODE_AUTO:
            self.__dt = self.t_current - self.t_last
                
        if self.beta_ref_last is None:
            self.beta_ref_dot = 0
        else:
            self.beta_ref_dot = (u_ref[1] - self.beta_ref_last)/self.__dt
        
        # beta to d(beta)/dt as control i/p
        u_ref[1] = self.beta_ref_dot
            
        if len(self.obstacle_list2d) < 1:
            raise ValueError("Cannot solve CBF for an empty obstacle list. \
                Update the obstacle list so that it is non-empty in order  \
                to move forward.")

        def F(x = None, z=None):

            # if x is None: return m, matrix(0.0, (n, 1))
            if x is None: return m, u_ref

            # for 1 objective function and 1 constraint and 3 state vars
            f = matrix(0.0, (m+1, 1))
            Df = matrix(0.0, (m+1, n))

            f[0] = (x - u_ref).T * self.__R * (x - u_ref)
            Df[0, :] = 2 * (x - u_ref).T * self.__R

            # State Equation:
            g_c = self.gc()

            f_c = self.fc()

            # G => Gradient, Gh -> (m,3)
            Gh = matrix([ [self.obstacle_list2d.dx()], [self.obstacle_list2d.dy()],\
                 [self.obstacle_list2d.dtheta()], [self.obstacle_list2d.dv()], [self.obstacle_list2d.dbeta()] ])
            
            Lxg_h = Gh * g_c
            Lxf_h = Gh * f_c

            f[1:] = -( Lxf_h + Lxg_h * x + ( self.__alpha * self.obstacle_list2d.f() )  + self.obstacle_list2d.dt())
            Df[1:, :] = -Lxg_h

            if z is None: return f, Df
            H = z[0] * 2 * self.__R
            return f, Df, H
        
        solver_op = solvers.cp(F)
        u = solver_op['x'] # contains a and d(beta)/dt
        
        if self.__DT_MODE_AUTO:
            self.__dt = self.t_current - self.t_last
            
        # d(beta)/dt to beta for state and final control i/p
        if self.beta_last is None:
            self.__beta = u[1] * self.__dt
        else:
            self.__beta += u[1] * self.__dt
            
        # beta to delta
        u[1] = np.arctan2((self.__lf + self.__lr) * np.tan(self.__beta), self.__lr)
        
        self.t_last = time.time()
        if return_solver:
            return solver_op, u
        else:
            return u