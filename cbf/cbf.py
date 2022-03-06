#!/bin/python3
"""

Implementation of the Control Barrier Functions using a numerical
solver. Uses the obstacle list classes defined in the obstacles
module for the corresponding CBF representations.

author: Neelaksh Singh

"""

import sys
import os

import numpy as np
from cvxopt import matrix, solvers
from euclid import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.obstacles import ObstacleList2D
except:
    raise

class KBM_VC_CBF2D():
    """
    The Kinetic Bicycle Model - Velocity Controlled - Control Barrier Function.
    """

    def __init__(self, gamma = 1.0):
        self.obstacle_list2d = ObstacleList2D()
        self.__gamma = gamma
        pass
    
    def update_state(self, p, theta):
        self.__p = p
        self.__theta = theta
    
    def set_gamma(self, gamma=1.0):
        self.__gamma = gamma
    
    def set_model_params(self, L):
        self.__L = L
    
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
            raise ValueError("Cannot solve CBF for an empty obstacle list.\
                Update the obstacle list so that it is non-empty in order \
                to move forward.")

        def F(x = None, z=None):

            # if x is None: return m, matrix(0.0, (n, 1))
            if x is None: return m, u_ref

            # for 1 objective function and 1 constraint and 3 state vars
            f = matrix(0.0, (m+1, 1))
            Df = matrix(0.0, (m+1, n))

            f[0] = (x - u_ref).T * (x - u_ref)
            Df[0, :] = 2 * (x - u_ref).T

            gc = matrix([ [np.cos(self.__theta), np.sin(self.__theta), 0], [0, 0, 1] ])
            # G => Gradient, Gh -> (m,3)
            Gh = matrix([ [self.obstacle_list2d.dx(self.__p)], [self.obstacle_list2d.dy(self.__p)], [self.obstacle_list2d.dtheta(self.__p)] ])
            
            Lxg_h = Gh * gc

            f[1:] = -( Lxg_h * x + ( self.__gamma * self.obstacle_list2d.f(self.__p) ) )
            Df[1:, :] = -Lxg_h

            if z is None: return f, Df
            H = z[0] * 2 * matrix(np.eye(n))
            return f, Df, H
        
        solver_op = solvers.cp(F)
        u = solver_op['x']
        u[1] = np.arctan(u[1] * self.__L / u_ref[0])
        return solver_op, u