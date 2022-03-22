#!/bin/python3
"""

The Obstacle classes containing the neccessary gradients and hessian functions for
seamless integration with optimal solvers, includes several utility objects like 
the obstacle list for use in real time simulation.

author: Neelaksh Singh

"""

import numpy as np
from euclid import *
from cvxopt import matrix

import warnings
from collections.abc import MutableMapping
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from cbf.geometry import Rotation, Transform
except:
    raise

# Identity Objects
DICT_EMPTY_UPDATE = ()

# Object Selectors for utility
ELLIPSE2D = 0

class BoundingBox():
    def __init__(self, extent=Vector3(), location=Vector3(), rotation=Rotation()):
        self.extent = extent
        self.location = location
        self.rotation = rotation

    def __eq__(self, other):
        return self.location == other.location and self.extent == other.extent
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def get_local_vertices(self):
        up = self.rotation.get_up_vector().normalized()
        right = self.rotation.get_right_vector().normalized()
        forward = self.rotation.get_forward_vector().normalized()
        v1 = -self.extent.z*up + self.extent.x*forward + self.extent.y*right
        v2 = -self.extent.z*up + self.extent.x*forward - self.extent.y*right
        v3 = -self.extent.z*up - self.extent.x*forward - self.extent.y*right
        v4 = -self.extent.z*up - self.extent.x*forward + self.extent.y*right
        v5 = self.extent.z*up + self.extent.x*forward + self.extent.y*right
        v6 = self.extent.z*up + self.extent.x*forward - self.extent.y*right
        v7 = self.extent.z*up - self.extent.x*forward - self.extent.y*right
        v8 = self.extent.z*up - self.extent.x*forward + self.extent.y*right
        return [v1, v2, v3, v4, v5, v6, v7, v8]

    def get_world_vertices(self, transform=Transform):
        v_list = self.get_local_vertices()
        return [transform.transform(v) for v in v_list]

class Obstacle2DBase():
    """
    The base class each 2D obstacle class will inherit from. Created to enforce specific
    validation checks in the obstacle list objects and creating the neccessary interface
    for all 2D obstacle CBF classes.
    """
    def __init__(self):
        pass

    def evaluate(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")

    def gradient(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")
        return matrix(0.0, (3,1))

    def f(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")
        return 0
    
    def dx(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")
        return 0
    
    def dy(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dtheta(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def dv(self, p):
        if not isinstance(p, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(p).__name__ + ".")
        return 0

    def update(self):
        pass

    def update_coords(self, xy):
        if not isinstance(xy, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg p, but got " + type(xy).__name__ + ".")
        pass

    def update_orientation(self):
        pass

class Ellipse2D(Obstacle2DBase):
    def __init__(self, a, b, center = Vector2(0, 0), theta=0, buffer=0):
        """
        Generates the 2D Ellipse obstacle representation for use in control barrier functions.
        Exposes the required functionality for direct usage in CBF as a barrier constraint.

        """
        if not isinstance(center, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg center, but got " + type(center).__name__ + ".")
        self.center = center
        self.theta = theta
        self.a = a + buffer
        self.b = b + buffer
        self.buffer = buffer
        self.BUFFER_FLAG = True
    
    def apply_buffer(self):
        if not self.BUFFER_FLAG:
            self.a = self.a + self.buffer
            self.b = self.b + self.buffer
            self.BUFFER_FLAG = True
        else:
            warnings.warn("Warning: Buffer already applied. Call Ignored.")
        
    def remove_buffer(self):
        if self.BUFFER_FLAG:
            self.a = self.a - self.buffer
            self.b = self.b - self.buffer
            self.BUFFER_FLAG = False
        else:
            warnings.warn("Warning: Buffer already removed. Call Ignored.")
    
    def evaluate(self, p):
        """
        Evaluate the value of the ellipse at a given point.
        """
        super().evaluate(p)
        dx = p.x - self.center.x
        dy = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        eval = ( ( dx * ct + dy * st )/self.a )**2 + ( ( -dx * st + dy * ct )/self.b )**2 - 1
        return eval

    def gradient(self, p):
        super().gradient(p)
        return matrix([self.dx(p), self.dy(p), self.dtheta(p)])

    # f = evaluate
        
    def f(self, p):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(p)
    
    def dx(self, p):
        super().dx(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        dx_ = (2 * ct/(self.a**2)) * ( xd * ct + yd * st ) + (-2 * st/(self.b**2)) * ( -xd * st + yd * ct )
        return dx_
    
    def dy(self, p):
        super().dy(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        dy_ = (2 * st/(self.a**2)) * ( xd * ct + yd * st ) + (2 * ct/(self.b**2)) * ( -xd * st + yd * ct )
        return dy_

    def dv(self, p):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dy(p)
    
    def update(self, a=None, b=None, center=None, theta=None, buffer=None):
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if center is not None:
            if not isinstance(center, Vector2):
                raise TypeError("Expected an object of type euclid.Vector2 for arg center.")
            self.center = center
        if theta is not None:
            self.theta = theta
        if buffer is not None:
            if self.BUFFER_FLAG:
                self.a = self.a - self.buffer + buffer
                self.b = self.b - self.buffer + buffer
                self.buffer = buffer
            else:
                self.buffer = buffer
    
    def update_coords(self, xy):
        super().update_coords(xy)
        self.center = xy

    def update_orientation(self, yaw):
        self.theta = yaw

    def update_by_bounding_box(self, bbox=BoundingBox()):
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
            
        a = bbox.extent.x
        b = bbox.extent.y
        center = Vector2(bbox.location.x, bbox.location.y)
        theta = bbox.rotation.yaw
        self.update(a=a, b=b, center=center, theta=theta)

    def dtheta(self, p):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dtheta(p)

    def __repr__(self):
        return f"{type(self).__name__}(a = {self.a}, b = {self.b}, center = {self.center}, theta = {self.theta}, buffer = {self.buffer}, buffer_applied: {self.BUFFER_FLAG} )\n"
    
    @classmethod
    def from_bounding_box(cls, bbox = BoundingBox(), buffer = 0.5):
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        a = bbox.extent.x
        b = bbox.extent.y
        center = Vector2(bbox.location.x, bbox.location.y)
        theta = bbox.rotation.yaw
        return cls(a, b, center, theta, buffer)
        
class ObstacleList2D(MutableMapping):

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)
    
    def __getitem__(self, key):
        return self.mapping[key]
    
    def __delitem__(self, key):
        del self.mapping[key]
    
    def __setitem__(self, key, value):
        """
        Contaions the enforced base class check to ensure it contains
        an object derived from the 2D obstacle base class.
        """
        # Enforcing base class check using mro.
        if not Obstacle2DBase in value.__class__.__mro__:
            raise TypeError("Expected an object derived from Obstacle2DBase as value. Received " + type(value).__name__)
        self.mapping[key] = value

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"

    def update_by_bounding_box(self, bbox_dict=None, obs_type=ELLIPSE2D, buffer=0.5):
        """
        Will update the obstacle based on the dynamic obstacle
        list criteria. Remove the IDs which are not present in
        the scene and add those which entered the scene. Update
        the IDs which have changed locations for reformulation 
        of the contained obstacle objects.
        """
        if bbox_dict is not None:
            for key, bbox in bbox_dict.items():
                if key in self.mapping.keys():
                    self.mapping[key].update_by_bounding_box(bbox)
                else:
                    if obs_type == ELLIPSE2D:
                        self.__setitem__(key, Ellipse2D.from_bounding_box(bbox, buffer))
            for key in self.mapping.keys():
                if key not in bbox_dict.keys():
                    self.pop(key)

    def f(self, p):
        f = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            f[idx] = obs.f(p)
            idx = idx + 1
        return f

    def dx(self, p):
        dx = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dx[idx] = obs.dx(p)
            idx = idx + 1
        return dx

    def dy(self, p):
        dy = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dy[idx] = obs.dy(p)
            idx = idx + 1
        return dy
    
    def dtheta(self, p):
        dtheta = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dtheta[idx] = obs.dtheta(p)
            idx = idx + 1
        return dtheta
    
    def dv(self, p):
        dv = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dv[idx] = obs.dv(p)
            idx = idx + 1
        return dv

    def gradient(self, p):
        df = matrix(0.0, (len(self.mapping), 3))
        idx = 0
        for obs in self.mapping.values():
            df[idx,:] = obs.gradient(p).T
            idx = idx + 1
        return df
