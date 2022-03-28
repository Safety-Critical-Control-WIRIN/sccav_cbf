#!/bin/python3
"""

!CAUTION!: All WIP imports are subject to change.

WIP -> Work in Progress. This file hosts the experimental objects
which are under development to be added as stable features in the
future. The purpose behind hosting the objects in a separate file
is to allow for better seggregation of stable and unstable scripts.

author: Neelaksh Singh

"""

import numpy as np
import glob
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from cbf.utils import normalize_angle
except:
    raise

class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt
    
    def update_by_vel(self, v_, delta):
        """
        Update the state of the vehicle. But
        instead of using an acceleration based
        control, use direct velocity control.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v = v_
    
    def update_com(self, acceleration, delta):
        delta = np.clip(delta, -max_steer, max_steer)
        beta = np.arctan2(lr * np.tan(delta), lf + lr)
        
        self.x += (self.v * np.cos(self.yaw) - self.v * np.sin(self.yaw) * beta) * dt
        self.y += (self.v * np.sin(self.yaw) + self.v * np.cos(self.yaw) * beta) * dt
        self.yaw += (self.v * beta/lr) * dt
        self.v += acceleration * dt
