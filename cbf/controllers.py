#!/bin/python3
"""

This script implements some useful controllers for generating
the reference control input to be fed to the CBF. Check the 
docs for a brief summary of the existing controllers.

author: Neelaksh Singh

"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from cbf.utils import normalize_angle

    ## !CAUTION!: All WIP Imports are subject to change
    from cbf.wip import State
except:
    raise

from euclid import *

class LateralStanley():
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """

    def __init__(self, lr = 2.0, lf = 2.0, k = 0.5, ks = 0.01):
        self.__lr = lr
        self.__lf = lf
        # strict ints
        self.__last_target_idx = 0
        self.__current_target_idx = 0

        self.__k = k
        self.__ks = ks

    def update_state(self, x, y, yaw, v):
        self.__x = x
        self.__y = y
        self.__yaw = yaw
        self.__v = v
    
    def set_gains(self, k, ks):
        self.__k = k
        self.__ks = ks
    
    def set_trajectory(self, trajectory):
        self.__trajectory = trajectory
        self.__xdes = [p[0] for p in trajectory]
        self.__ydes = [p[1] for p in trajectory]
        self.__yawdes = [p[2] for p in trajectory]
        self.__vdes = [p[3] for p in trajectory]
        self.__set_traj_flag = True
    
    def _calc_target_index(self, front_coords=None):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        if front_coords is None:
            # Calc front axle position
            fx = self.__x + self.__lf * np.cos(self.__yaw)
            fy = self.__y + self.__lf * np.sin(self.__yaw)
        else:
            fx = front_coords.x
            fy = front_coords.y
        
        self.fx = fx
        self.fy = fy

        # Search nearest point index
        dx = [fx - icx for icx in self.__xdes]
        dy = [fy - icy for icy in self.__ydes]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(self.__yaw + np.pi / 2),
                        -np.sin(self.__yaw + np.pi / 2)]

        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle


    def control(self, trajectory=None, front_coords=None, initial_yaw = 0):
        if front_coords is not None:
            if not isinstance(front_coords, Vector2):
                    raise TypeError("Coordinates of the front wheel must be specified as an\
                        euclid.Vector2() object.")

            self.__current_target_idx, error_front_axle = self._calc_target_index(front_coords=front_coords)
        
        else:
            self.__current_target_idx, error_front_axle = self._calc_target_index()

        if trajectory is not None:
            self.__trajectory = trajectory

        if self.__last_target_idx >= self.__current_target_idx:
            self.__current_target_idx = self.__last_target_idx

        # Recalculating error_front_axle by a different method
        dfx = self.fx - self.__xdes[self.__current_target_idx]
        dfy = self.fy - self.__ydes[self.__current_target_idx]
        cross_track_error =  np.hypot(dfx, dfy)
        
        cross_track_direction = normalize_angle(np.arctan2(dfx, dfy))
        if cross_track_direction < 0:
            cross_track_error = -cross_track_error

        ## Manually calculating the reference yaw ##
        yaw_margin_idx = 2
        if self.__current_target_idx > yaw_margin_idx:
            dxx = self.__xdes[self.__current_target_idx] - self.__xdes[self.__current_target_idx - yaw_margin_idx]
            dyy = self.__ydes[self.__current_target_idx] - self.__ydes[self.__current_target_idx - yaw_margin_idx]
            yaw_des = np.arctan2(dyy, dxx)

        else:
            yaw_des = initial_yaw

        # theta_e corrects the heading error
        theta_e = normalize_angle(self.__yawdes[self.__current_target_idx] - self.__yaw)
        # theta_e = normalize_angle(yaw_des - self.__yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.__k * error_front_axle, self.__v + self.__ks)
        # theta_d = normalize_angle(np.arctan(self.__k * cross_track_error / (self.__ks + self.__v)))
        # Steering control
        delta = theta_e + theta_d

        self.__last_target_idx = self.__current_target_idx

        return delta, self.__current_target_idx

class PID1():

    def __init__(self, kp=1.0, kd=0.0, ki=0.0):
        self.__kp = kp
        self.__kd = kd
        self.__ki = ki
        self.__e = 0
        self.__eprev = 0
        self.__de = 0
        self.__ie = 0
        self.__dt = 0.1
    
    def set_gains(self, kp, kd, ki):
        self.__kp = kp
        self.__kd = kd
        self.__ki = ki
    
    def set_dt(self, dt):
        self.__dt = dt

    def control(self, x, xref):
        self.__e = xref - x
        self.__de = (self.__e - self.__eprev)/self.__dt
        self.__ie += self.__dt * self.__e
        
        u = self.__kp * self.__e + self.__ki * self.__ie + self.__kd * self.__de
        self.__eprev = self.__e
        return u