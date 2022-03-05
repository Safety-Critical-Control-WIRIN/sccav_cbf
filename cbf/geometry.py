#!/bin/python3
"""

A custom geometry module for the CBFs to reimplement neccessary carla
geometry objects through euclid for use in certain cbf objects.

author: Neelaksh Singh

"""

from turtle import forward
from euclid import *

class Rotation():
    """
    A reimplimentation of the CARLA rotation class with further added
    functionalities including quaternion support. Note that the FOR is:
    x: direction of approach/travel (heading)
    y: direction to the side (left | right)
    z: direction of height (upwards | downwards)
    Therefore, RPY corresponds to YXZ instead of XYZ.
    Note now that we will use a right handed coordinate system. So, Y
    points towards "left" => Z is upwards.
    """
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        # Euclid uses the h, a, b system which corresponds to
        # euler angles as follows:
        # Heading  => Yaw
        # Attitude => Pitch
        # Bank     => Roll
        # All euclid quaternion functions expect the sequence 
        # H, A, B => Y, P, R
        self._quaternion = Quaternion.new_rotate_euler(yaw, pitch, roll)

    def __eq__(self, other):
        if self.pitch == other.pitch and self.yaw == other.yaw and self.roll == other.roll:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return f"{type(self).__name__}(roll = {self.roll}, pitch = {self.pitch}, yaw = {self.yaw}, quaternion = {self.quaternion})"
    
    def __repr__(self):
        return self.__str__ + "\n"

    def get_quaternion(self):
        return self._quaternion

    def get_up_vector(self):
        up = Vector3(0.0, 0.0, 1.0)
        return self._quaternion * up

    def get_right_vector(self):
        right = Vector3(0.0, -1.0, 0.0)
        return self._quaternion * right

    def get_forward_vector(self):
        forward = Vector3(1.0, 0, 0)
        return self._quaternion * forward
