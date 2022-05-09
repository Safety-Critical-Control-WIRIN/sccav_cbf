#!/bin/python3
"""

The Obstacle classes containing the neccessary gradients and hessian functions for
seamless integration with optimal solvers, includes several utility objects like 
the obstacle list for use in real time simulation.

author: Neelaksh Singh

"""
# Removal of the following method for Type Hinting Enclosing
# classes is possible. Be cautious about the changes.
from __future__ import annotations

import sys
import os
import warnings
import enum

import numpy as np

from euclid import *
from cvxopt import matrix
from collections.abc import MutableMapping

from cbf.utils import vec_norm


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from cbf.geometry import Rotation, Transform
    from cbf.utils import Timer
except:
    raise

# Identity Objects
class IdentityObjects(enum.Enum):
    """
    Enumerations for Required Empty Identity Objects.
    """
    DICT_EMPTY_UPDATE = ()

# Object Selectors for utility
class Obstacle2DTypes(enum.Enum):
    """
    Enumerations for the available 2D obstacle classes.
    """
    ELLIPSE2D = 0
    COLLISION_CONE2D = 1

class BoundingBox():
    def __init__(self, extent=Vector3(), location=Vector3(), rotation=Rotation(), velocity:float = 0.0):
        self.extent = extent
        self.location = location
        self.rotation = rotation
        self.velocity = velocity

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
    
    def dt(self, p: Point2):
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
    """
    Generates the 2D Ellipse obstacle representation for use in control barrier functions.
    Exposes the required functionality for direct usage in CBF as a barrier constraint.

    """
    def __init__(self, a: float, b: float, center: Vector2 = Vector2(0, 0), theta: float=0, buffer: float=0):
        """
        Initializes the Ellipse2D Object. 
        """
        if not isinstance(center, Vector2):
            raise TypeError("Expected an object of type euclid.Vector2 for arg center, but got " + type(center).__name__ + ".")
        self.center = center
        self.theta = theta
        self.vel = Vector2()
        self.a = a + buffer
        self.b = b + buffer
        self.buffer = buffer
        self.BUFFER_FLAG = True

    def __repr__(self):
        return f"{type(self).__name__}(a = {self.a}, b = {self.b}, center = {self.center}, theta = {self.theta}, buffer = {self.buffer}, buffer_applied: {self.BUFFER_FLAG} )\n"
    
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
    
    def evaluate(self, p: Point2):
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

    def gradient(self, p: Point2):
        super().gradient(p)
        return matrix([self.dx(p), self.dy(p), self.dtheta(p)])

    # f = evaluate
        
    def f(self, p: Point2):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(p)
    
    def dx(self, p: Point2):
        super().dx(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        dx_ = (2 * ct/(self.a**2)) * ( xd * ct + yd * st ) + (-2 * st/(self.b**2)) * ( -xd * st + yd * ct )
        return dx_
    
    def dy(self, p: Point2):
        super().dy(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        dy_ = (2 * st/(self.a**2)) * ( xd * ct + yd * st ) + (2 * ct/(self.b**2)) * ( -xd * st + yd * ct )
        return dy_

    def dv(self, p: Point2):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dy(p)
    
    def update(self, a: float=None, b: float=None, center: float=None, theta: float=None, buffer: float=None):
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
    
    def update_coords(self, xy: Point2):
        super().update_coords(xy)
        self.center = xy
    
    def update_state(self, xy: Point2, theta: float, v: Vector2):
        self.update_coords(xy)
        self.vel = v
        self.theta = theta
    
    def update_velocity_by_magnitude(self, v: float):
        """
        Assumes that theta is the heading the calculates the vector.
        """
        self.vel = Vector2(x=v*np.cos(self.theta), y=v*np.sin(self.theta))
        pass

    def update_velocity(self, v: Vector2):
        """
        Sets the velocity using the Vector2 object. Note that this will
        create a copy of the argument's object to avoid external mutation
        of the attributes.
        """
        self.vel = v.copy()
        pass

    def update_orientation(self, yaw: float):
        self.theta = yaw
        _v_mag = self.vel.magnitude()
        self.update_velocity_by_magnitude(_v_mag)
        pass

    def update_by_bounding_box(self, bbox: BoundingBox):
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
            
        a = bbox.extent.x
        b = bbox.extent.y
        center = Vector2(bbox.location.x, bbox.location.y)
        theta = bbox.rotation.yaw
        self.update(a=a, b=b, center=center, theta=theta)

    def dtheta(self, p: Point2):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dtheta(p)
    
    def dt(self, p: Point2):
        super().dt(p)
        xd = p.x - self.center.x
        yd = p.y - self.center.y

        dt_ = -2 * ( (xd/self.a**2) * self.vel.x + (yd/self.b**2) * self.vel.y )
        return dt_

    
    @classmethod
    def from_bounding_box(cls, bbox = BoundingBox(), buffer = 0.5) -> Ellipse2D:
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        a = bbox.extent.x
        b = bbox.extent.y
        center = Vector2(bbox.location.x, bbox.location.y)
        theta = bbox.rotation.yaw
        return cls(a, b, center, theta, buffer)
    
class CollisionCone2D(Obstacle2DBase):
    """
    Generates a 2D Collision Cone based CBF for dynamic obstacle avoidance.
    """
    def __init__(self, 
                 a: float = 0.0, 
                 s: matrix = matrix(0, (4,1)), 
                 s_obs: matrix = matrix(0, (4,1)),
                 buffer: float=0.5):
        """
        Initializes the CollisionCone2D Object. 
        """
        self.s = s
        self.s_obs = s_obs
        self.a = a + buffer
        self.buffer = buffer
        self.BUFFER_FLAG = True
        
        self.s = matrix(s)
        self.s_obs = matrix(s_obs)
        self.cx = self.s_obs[0]
        self.cy = self.s_obs[1]
        self.s_vx = s[3]*np.cos(s[2])
        self.s_vy = s[3]*np.sin(s[2])
        self.s_obs_vx = s_obs[3]*np.cos(s_obs[2])
        self.s_obs_vy = s_obs[3]*np.sin(s_obs[2])
        self.p_rel = self.s[:2] - self.s_obs[:2]
        self.v_rel = matrix([ self.s_vx - self.s_obs_vx, self.s_vy - self.s_obs_vy])
        self.dist = vec_norm(self.p_rel)
        self.v_rel_norm = vec_norm(self.v_rel)
        self.cone_boundary = 0
        if (self.dist - self.a) >= 0:
            self.cone_boundary = np.sqrt(self.dist**2 - self.a**2)
        self.cos_phi = self.cone_boundary/self.dist
        
    def __repr__(self):
        return f"{type(self).__name__}(a = {self.a}, b = {self.b}, center = {self.center}, theta = {self.theta}, buffer = {self.buffer}, buffer_applied: {self.BUFFER_FLAG} )\n"
    
    def apply_buffer(self):
        if not self.BUFFER_FLAG:
            self.a = self.a + self.buffer
            self.BUFFER_FLAG = True
        else:
            warnings.warn("Warning: Buffer already applied. Call Ignored.")
        
    def remove_buffer(self):
        if self.BUFFER_FLAG:
            self.a = self.a - self.buffer
            self.BUFFER_FLAG = False
        else:
            warnings.warn("Warning: Buffer already removed. Call Ignored.")
    
    def evaluate(self, p: Point2):
        """
        Since the cone depends on relative parameter this function uses the 
        current state to calculate the evaluation of the cone at the current point
        therefore doesn't take any other arguments. It is mandatory to update the
        state of the vehicle for this obstacle type to function properly.
        """
        eval = (self.p_rel.T * self.v_rel) + (self.dist * self.v_rel_norm * self.cos_phi)
        return eval

    def gradient(self, p: Point2):
        return matrix([self.dx(), self.dy(), self.dtheta(), self.dv()])

    # f = evaluate
        
    def f(self, p: Point2):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(p)
    
    def dx(self, p: Point2):

        q_dx = self.s_vx - self.s_obs_vx
        phi_term_dx = self.v_rel_norm * (self.s[0] - self.cx)/self.cone_boundary
        dx_ = q_dx + phi_term_dx
        return dx_
    
    def dy(self, p: Point2):
        
        q_dy = self.s_vy - self.s_obs_vy
        phi_term_dy = self.v_rel_norm * (self.s[1] - self.cy)/self.cone_boundary
        dy_ = q_dy + phi_term_dy
        return dy_

    def dv(self, p: Point2):
        
        q_dv = (self.s[0] - self.cx) * np.cos(self.s[2]) + (self.s[1] - self.cy) * np.sin(self.s[2])
        phi_term_dv = ( (self.s_vx - self.s_obs_vx)*np.cos(self.s[2]) + (self.s_vy - self.s_obs_vy)*np.sin(self.s[2]) ) * self.cone_boundary/self.v_rel_norm
        dv_ = q_dv + phi_term_dv
        return dv_
    
    def dtheta(self, p: Point2):
        
        q_dtheta = - (self.s[0] - self.cx) * self.s_vy + (self.s[1] - self.cy) * self.s_vx
        phi_term_dtheta = ( -(self.s_vx - self.s_obs_vx)*self.s_vy + (self.s_vy - self.s_obs_vy)*self.s_vx ) * self.cone_boundary/self.v_rel_norm
        dtheta_ = q_dtheta + phi_term_dtheta
        return dtheta_
    
    def dt(self, p: Point2):
        
        q_dt = - (self.s_vx - self.s_obs_vx) * self.s_obs_vx - (self.s_vy - self.s_obs_vy) * self.s_obs_vy
        phi_term_dt = -self.v_rel_norm * ( (self.s[0] - self.cx)*self.s_obs_vx + (self.s[1] - self.cy)*self.s_obs_vy )/self.cone_boundary
        dt_ = q_dt + phi_term_dt
        return dt_

    
    def update(self, a: float=None, s: matrix=None, s_obs: matrix=None, buffer: float=None):
        if a is not None:
            self.a = a
        if s is not None:
            self.s = matrix(s)
        if s_obs is not None:
            self.s_obs = matrix(s_obs)
        if buffer is not None:
            if self.BUFFER_FLAG:
                self.a = self.a - self.buffer + buffer
                self.buffer = buffer
            else:
                self.buffer = buffer
        
        self.cx = self.s_obs[0]
        self.cy = self.s_obs[1]
        self.s_vx = s[3]*np.cos(s[2])
        self.s_vy = s[3]*np.sin(s[2])
        self.s_obs_vx = s_obs[3]*np.cos(s_obs[2])
        self.s_obs_vy = s_obs[3]*np.sin(s_obs[2])
        self.p_rel = self.s[:2] - self.s_obs[:2]
        self.v_rel = matrix([ self.s_vx - self.s_obs_vx, self.s_vy - self.s_obs_vy])
        self.dist = vec_norm(self.p_rel)
        self.v_rel_norm = vec_norm(self.v_rel)
        self.cone_boundary = np.sqrt(self.dist**2 - self.a**2)
        self.cos_phi = self.cone_boundary/self.dist
    
    def update_state(self, s: matrix, s_obs: matrix):
        self.update(s=s, s_obs=s_obs)
    
    def get_half_angle(self):
        """Returns the apex half angle of the collision cone.
        """
        return np.arccos(self.cos_phi)

    def update_by_bounding_box(self, bbox: BoundingBox):
        """Updates the obstacle state for the collision cone using the
        obstacle's BoundingBox object. Calls the update function after
        making the obstacle state vector. The `a` parameter is taken as
        the diagonal of the box's base/projection on ground plane.
        
        Parameters:
        ----------
            bbox (BoundingBox): The bounding box object associated with the obstacle.

        Raises:
        ------
            TypeError: The argument has to be strictly of type `cbf.obstacles.BoundingBox`
        """
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
            
        self.a = np.hypot(bbox.extent.x, bbox.extent.y)
        s_obs = matrix([bbox.location.x, bbox.location.y, 0.0, bbox.velocity])
        self.update(s_obs=s_obs)
    
    @classmethod
    def from_bounding_box(cls, bbox = BoundingBox(), buffer = 0.5) -> CollisionCone2D:
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        a = np.hypot(bbox.extent.x, bbox.extent.y)
        s_obs = matrix([bbox.location.x, bbox.location.y, 0.0, bbox.velocity])
        return cls(a=a, s_obs=s_obs, buffer=buffer)
        
class ObstacleList2D(MutableMapping):

    def __init__(self, data=()):
        self.mapping = {}
        self.update(data)
        self.timestamp = 0.0
    
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
    
    def set_timestamp(self, timestamp: float):
        self.timestamp = timestamp

    def update_by_bounding_box(self, bbox_dict=None, obs_type=Obstacle2DTypes.ELLIPSE2D, buffer=0.5):
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
                    if obs_type == Obstacle2DTypes.ELLIPSE2D:
                        self.__setitem__(key, Ellipse2D.from_bounding_box(bbox, buffer))
                    if obs_type == Obstacle2DTypes.COLLISION_CONE2D:
                        self.__setitem__(key, CollisionCone2D.from_bounding_box(bbox, buffer))
            
            rm_keys = []
            for key in self.mapping.keys():
                if key not in bbox_dict.keys():
                    rm_keys.append(key)
            
            for key in rm_keys:
                self.pop(key)

    def f(self, p: Point2) -> float:
        f = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            f[idx] = obs.f(p)
            idx = idx + 1
        return f

    def dx(self, p: Point2) -> float:
        dx = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dx[idx] = obs.dx(p)
            idx = idx + 1
        return dx

    def dy(self, p: Point2) -> float:
        dy = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dy[idx] = obs.dy(p)
            idx = idx + 1
        return dy
    
    def dtheta(self, p: Point2) -> float:
        dtheta = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dtheta[idx] = obs.dtheta(p)
            idx = idx + 1
        return dtheta
    
    def dv(self, p: Point2) -> float:
        dv = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dv[idx] = obs.dv(p)
            idx = idx + 1
        return dv
    
    def dt(self, p: Point2) -> float:
        dt = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dt[idx] = obs.dt(p)
            idx = idx + 1
        return dt

    def gradient(self, p: Point2) -> float:
        df = matrix(0.0, (len(self.mapping), 3))
        idx = 0
        for obs in self.mapping.values():
            df[idx,:] = obs.gradient(p).T
            idx = idx + 1
        return df
