#!/bin/python3
"""

utility functions for the CBF library involving Left to Right coordinate
transforms.

author: Neelaksh Singh

"""

import sys
import os

import numpy as np

from euclid import *
from cvxopt import matrix, sqrt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.geometry import Transform, Rotation
except:
    raise

ZERO_TOL = 1e-3

class TimerError(Exception):
    """ Custom Exception for Timer related errors. """
    pass

class Timer():
    
    def __init__(self, timestamp=0.0):
        self.timestamp = timestamp
        
    @property
    def timestamp(self):
        return self.timestamp_
    
    @timestamp.setter
    def timestamp(self, value):
        if self.timestamp > value:
            raise TimerError('''Negative time or Decreasing Timestamp trend detected.\
                    Please make sure that the Timestamp is monotonically increasing when\
                    manually set.''')
        self.timestamp_ = value
    

def convert_LH_to_RH(flipped_axis = 'y', *args):
    if flipped_axis == 'y':
        for arg in args:
            if isinstance(arg, Rotation):
                return Rotation(arg.roll, -arg.pitch, -arg.yaw)
            elif isinstance(arg, Vector3):
                if isinstance(arg, Point3):
                    return Point3(arg.x, -arg.y, arg.z)
                else:
                    return Vector3(arg.x, -arg.y, arg.z)
            else:
                raise TypeError("Invalid input. Expected euclid.Vector3\
                    , euclid.Point3 or cbf.geometry.Rotation objects. Received " + type(arg).__name)
    
    elif flipped_axis == 'x':
        for arg in args:
            if isinstance(arg, Rotation):
                return Rotation(arg.roll, -arg.pitch, -arg.yaw)
            elif isinstance(arg, Vector3):
                if isinstance(arg, Point3):
                    return Point3(-arg.x, arg.y, arg.z)
                else:
                    return Vector3(-arg.x, arg.y, arg.z)
            else:
                raise TypeError("Invalid input. Expected euclid.Vector3\
                    , euclid.Point3 or cbf.geometry.Rotation objects. Received " + type(arg).__name)
    elif flipped_axis == 'z':
        for arg in args:
            if isinstance(arg, Rotation):
                return Rotation(arg.roll, -arg.pitch, -arg.yaw)
            elif isinstance(arg, Vector3):
                if isinstance(arg, Point3):
                    return Point3(arg.x, arg.y, -arg.z)
                else:
                    return Vector3(arg.x, arg.y, -arg.z)
            else:
                raise TypeError("Invalid input. Expected euclid.Vector3\
                    , euclid.Point3 or cbf.geometry.Rotation objects. Received " + type(arg).__name)
    else:
        raise ValueError("Invalid input to the flipped_axis argument. Expected values\
             from ['x', 'y', 'z']. Received " + flipped_axis)

def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def saturation(x, x_min, x_max):
    if x > x_max:
        return x_max
    elif x < x_min:
        return x_min
    else:
        return x

def get_closest_idx(x, x_list):
    dx = [abs(x - ix) for ix in x_list]
    return np.argmin(dx)

def vec_norm(x: matrix):
    return sqrt(x.T * x)[0]