#!/bin/python3
"""

A custom geometry module for the CBFs to reimplement neccessary carla
geometry objects through euclid for use in certain cbf objects.

author: Neelaksh Singh

"""

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
    def __init__(self, roll=0.0, pitch=0.0, yaw=0.0, right_handed=True):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.heading = self.yaw
        self.attitude = self.pitch
        self.bank = self.roll
        self.__right_handed = right_handed
        # Euclid uses the h, a, b system which corresponds to
        # euler angles as follows:
        # Heading  => Yaw
        # Attitude => Pitch
        # Bank     => Roll
        # All euclid quaternion functions expect the sequence 
        # H, A, B => Y, P, R
        self.__quaternion = Quaternion.new_rotate_euler(yaw, pitch, roll)

    def __eq__(self, other):
        if self.pitch == other.pitch and self.yaw == other.yaw and self.roll == other.roll:
            return True
        else:
            return False
        
    def update(self, roll=0.0, pitch=0.0, yaw=0.0, right_handed=True):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.heading = self.yaw
        self.attitude = self.pitch
        self.bank = self.roll
        self.__right_handed = right_handed
        # Euclid uses the h, a, b system which corresponds to
        # euler angles as follows:
        # Heading  => Yaw
        # Attitude => Pitch
        # Bank     => Roll
        # All euclid quaternion functions expect the sequence 
        # H, A, B => Y, P, R
        self.__quaternion = Quaternion.new_rotate_euler(yaw, pitch, roll)       

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self):
        return f"{type(self).__name__}(roll = {self.roll}, pitch = {self.pitch}, yaw = {self.yaw}, quaternion = {self._quaternion})"
    
    def __repr__(self):
        return f"{type(self).__name__}(roll = {self.roll}, pitch = {self.pitch}, yaw = {self.yaw}, quaternion = {self._quaternion})\n"

    def set_right_handed_flag(self, _right_handed):
        self.__right_handed = _right_handed

    def get_quaternion(self):
        return self.__quaternion

    def get_up_vector(self):
        up = Vector3(0.0, 0.0, 1.0)
        return self.__quaternion * up

    def get_right_vector(self):
        if self.__right_handed:
            right = Vector3(0.0, -1.0, 0.0)
            return self.__quaternion * right
        else:
            right = Vector3(0.0, 1.0, 0.0)
            return self.__quaternion * right

    def get_forward_vector(self):
        forward = Vector3(1.0, 0, 0)
        return self.__quaternion * forward
    
    @classmethod
    def from_quaternion(cls, w=1.0, x=0.0, y=0.0, z=0.0):
        quaternion = Quaternion(w=w, x=x, y=y, z=z)
        heading, attitude, bank = quaternion.get_euler()
        return cls(roll=bank, pitch=attitude, yaw=heading)


class Transform():
    """
    A CARLA style implementation of the Transform object by wrapping
    over the euclid Matrix4 class. 
    """
    def __init__(self, location=Vector3(), rotation=Rotation()):
        self.location = location
        self.rotation = rotation
        self.__matrix = Matrix4()
        self.__matrix.rotate_euler(rotation.heading, rotation.attitude, rotation.bank)
        self.__matrix.translate(location.x, location.y, location.z)
    
    def __eq__(self, other):
        return self.location == other.location and self.rotation == other.rotation
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __str__(self, other):
        return self.__matrix.__str__

    def __repr__(self, other):
        return self.__matrix.__repr__
    
    def transform(self, p = Point3()):
        return self.__matrix.transform(p)
    
    def transform_inverse(self, p = Point3()):
        matrix_inv = self.__matrix.inverse()
        return matrix_inv.transform(p)
    
    def get_forward_vector(self):
        return self.rotation.get_forward_vector()

    def get_inverse_matrix(self):
        return self.__matrix.inverse()

    def get_matrix(self):
        return self.__matrix

    def get_right_vector(self):
        return self.rotation.get_right_vector()

    def get_up_vector(self):
        return self.rotation.get_up_vector()