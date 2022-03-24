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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.geometry import Transform, Rotation
except:
    raise

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