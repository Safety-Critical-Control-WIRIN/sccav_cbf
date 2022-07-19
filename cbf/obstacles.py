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
from multiprocessing.sharedctypes import Value

import sys
import os
from typing import Dict
import warnings
import enum

import numpy as np
import scipy as sci

from euclid import *
from cvxopt import matrix
from collections.abc import MutableMapping
from numpy.polynomial.polynomial import Polynomial
from sympy import Point

from cbf.utils import vec_norm


sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from cbf.geometry import Rotation, Transform
    from cbf.utils import Timer, ZERO_TOL
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
    POLY_LANE = 2

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
    All the error checks can be added to the base members for specific args and kwargs
    expected from each obstacle type. Thus this base class also serves as an interface 
    for throwing errors. However, it is more preferable to have a separate error handler
    class once the applications and obstacle types grow.
    """
    def __init__(self):
        pass

    def evaluate(self, *args, **kwargs):
        return 0

    def gradient(self, *args, **kwargs):
        return matrix(0.0, (4,1))

    def f(self, *args, **kwargs):
        return 0
    
    def dx(self, *args, **kwargs):
        return 0
    
    def dy(self, *args, **kwargs):
        return 0

    def dtheta(self, *args, **kwargs):
        return 0

    def dv(self, *args, **kwargs):
        return 0
    
    def dbeta(self, *args, **kwargs):
        return 0
    
    def dt(self, *args, **kwargs):
        return 0
    
    def update(self, *args, **kwargs):
        pass

    def update_coords(self, *args, **kwargs):
        pass

    def update_orientation(self, *args, **kwargs):
        pass

class Ellipse2D(Obstacle2DBase):
    """
    Generates the 2D Ellipse obstacle representation for use in control barrier functions.
    Exposes the required functionality for direct usage in CBF as a barrier constraint.

    """
    
    def __init__(self, a: float, b: float, center: Vector2 = Vector2(0, 0), theta: float=0, buffer: float=0, **kwargs):
        """
        Initializes the Ellipse2D Object. 
        """
        self.type = Obstacle2DTypes.ELLIPSE2D
        if 'id' in kwargs.keys():
            self.id = kwargs['id']
        
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
    
    def evaluate(self, **kwargs):
        """
        Evaluate the value of the ellipse at a given point.
        """
        p = Point2(self.s[0], self.s[1])
        dx = p.x - self.center.x
        dy = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        eval = ( ( dx * ct + dy * st )/self.a )**2 + ( ( -dx * st + dy * ct )/self.b )**2 - 1
        return eval

    def gradient(self, **kwargs):
        return matrix([self.dx(**kwargs), 
                       self.dy(**kwargs), 
                       self.dtheta(**kwargs), 
                       self.dv(**kwargs)])

    # f = evaluate
        
    def f(self, **kwargs):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(**kwargs)
    
    def dx(self, **kwargs):
        p = Point2(self.s[0], self.s[1])
        super().dx(**kwargs)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        dx_ = (2 * ct/(self.a**2)) * ( xd * ct + yd * st ) + (-2 * st/(self.b**2)) * ( -xd * st + yd * ct )
        return dx_
    
    def dy(self, **kwargs):
        p = Point2(self.s[0], self.s[1])
        super().dy(**kwargs)
        xd = p.x - self.center.x
        yd = p.y - self.center.y
        ct = np.cos(self.theta)
        st = np.sin(self.theta)

        dy_ = (2 * st/(self.a**2)) * ( xd * ct + yd * st ) + (2 * ct/(self.b**2)) * ( -xd * st + yd * ct )
        return dy_

    def dv(self, **kwargs):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dy(**kwargs)
    
    def update(self, s: matrix=None, s_obs: matrix=None, center: float=None, buffer: float=None, **kwargs):
        
        if 'a' in kwargs.keys():
            self.a = kwargs['a']
            
        if 'b' in kwargs.keys():
            self.a = kwargs['b']
            
        if 'theta' in kwargs.keys():
            self.theta = kwargs['theta']
        
        if s_obs is not None:
            center = Point2(s_obs[0], s_obs[1])
            self.center = center
        
        if s is not None:
            self.s = s
            self.vel = s[3]
            self.theta = s[2]
        
        if buffer is not None:
            if self.BUFFER_FLAG:
                self.a = self.a - self.buffer + buffer
                self.b = self.b - self.buffer + buffer
                self.buffer = buffer
            else:
                self.buffer = buffer
    
    def update_coords(self, xy: Point2):
        self.center = xy
    
    def update_state(self, s: matrix, s_obs: matrix, **kwargs):
        self.update(s=s, s_obs=s_obs)
    
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

    def dtheta(self, *kwargs):
        """
        Despite being zero. This function is still created for the sake of completeness w.r.t API.
        """
        return super().dtheta(**kwargs)
    
    def dt(self, **kwargs):
        p = Point2(self.s[0], self.s[1])
        super().dt(**kwargs)
        xd = p.x - self.center.x
        yd = p.y - self.center.y

        dt_ = -2 * ( (xd/self.a**2) * self.vel.x + (yd/self.b**2) * self.vel.y )
        return dt_
    
    @classmethod
    def from_bounding_box(cls, bbox = BoundingBox(), buffer = 0.5, **kwargs) -> Ellipse2D:
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        if 'id' in kwargs.keys():
            id = kwargs['id']
        
        a = bbox.extent.x
        b = bbox.extent.y
        center = Vector2(bbox.location.x, bbox.location.y)
        theta = bbox.rotation.yaw
        return cls(a, b, center, theta, buffer, id=id)
    
class CollisionCone2D(Obstacle2DBase):
    """
    Generates a 2D Collision Cone based CBF for dynamic obstacle avoidance.
    """
    def __init__(self, 
                 a: float = 0.0, 
                 s: matrix = matrix(0, (4,1)), 
                 s_obs: matrix = matrix(0, (4,1)),
                 buffer: float=1.50,
                 **kwargs):
        """
        Initializes the CollisionCone2D Object. F
        """
        self.type = Obstacle2DTypes.COLLISION_CONE2D
        if 'id' in kwargs.keys():
            self.id = kwargs['id']
        
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
        self.cone_boundary = ZERO_TOL
        
        if abs(self.dist) > abs(self.a):
            self.cone_boundary = np.sqrt(self.dist**2 - self.a**2) + ZERO_TOL
        
        if self.dist > ZERO_TOL:
            self.cos_phi = self.cone_boundary/self.dist
        else:
            self.cos_phi = 0
        
    def __repr__(self):
        return f"{type(self).__name__}(a = {self.a}, cone_boundary = {self.cone_boundary}, apex = {np.arccos(self.cos_phi)}, buffer = {self.buffer}, buffer_applied: {self.BUFFER_FLAG} )\n \
            s_obs = {self.s_obs.T}"
    
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
    
    def evaluate(self, **kwargs):
        """
        Since the cone depends on relative parameter this function uses the 
        current state to calculate the evaluation of the cone at the current point
        therefore doesn't take any other arguments. It is mandatory to update the
        state of the vehicle for this obstacle type to function properly.
        """
        eval = (self.p_rel.T * self.v_rel) + (self.dist * self.v_rel_norm * self.cos_phi)
        return eval

    def gradient(self, **kwargs):
        return matrix([self.dx(**kwargs), 
                       self.dy(**kwargs),
                       self.dtheta(**kwargs),
                       self.dv(**kwargs)])

    # f = evaluate
        
    def f(self, **kwargs):
        """
        Alias of the evaluate function, semantically significant for cvxopt.
        """
        return self.evaluate(**kwargs)
    
    def dx(self, **kwargs):

        q_dx = self.s_vx - self.s_obs_vx
        phi_term_dx = self.v_rel_norm * (self.s[0] - self.cx)/(self.cone_boundary + ZERO_TOL)
        dx_ = q_dx + phi_term_dx
        return dx_
    
    def dy(self, **kwargs):
        
        q_dy = self.s_vy - self.s_obs_vy
        phi_term_dy = self.v_rel_norm * (self.s[1] - self.cy)/(self.cone_boundary + ZERO_TOL)
        dy_ = q_dy + phi_term_dy
        return dy_

    def dv(self, **kwargs):
        
        q_dv = (self.s[0] - self.cx) * np.cos(self.s[2]) + (self.s[1] - self.cy) * np.sin(self.s[2])
        phi_term_dv = ( (self.s_vx - self.s_obs_vx)*np.cos(self.s[2]) + (self.s_vy - self.s_obs_vy)*np.sin(self.s[2]) ) * self.cone_boundary/(self.v_rel_norm + ZERO_TOL)
        dv_ = q_dv + phi_term_dv
        return dv_
    
    def dtheta(self, **kwargs):
        
        q_dtheta = - (self.s[0] - self.cx) * self.s_vy + (self.s[1] - self.cy) * self.s_vx
        phi_term_dtheta = ( -(self.s_vx - self.s_obs_vx)*self.s_vy + (self.s_vy - self.s_obs_vy)*self.s_vx ) * self.cone_boundary/(self.v_rel_norm + ZERO_TOL)
        dtheta_ = q_dtheta + phi_term_dtheta
        return dtheta_
    
    def dt(self, **kwargs):
        
        q_dt = - (self.s_vx - self.s_obs_vx) * self.s_obs_vx - (self.s_vy - self.s_obs_vy) * self.s_obs_vy
        phi_term_dt = -self.v_rel_norm * ( (self.s[0] - self.cx)*self.s_obs_vx + (self.s[1] - self.cy)*self.s_obs_vy )/(self.cone_boundary + ZERO_TOL)
        dt_ = q_dt + phi_term_dt
        return dt_

    def update(self, s: matrix=None, s_obs: matrix=None, buffer: float=None, **kwargs):
        if 'a' in kwargs.keys():
            self.a = kwargs['a']
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
        self.s_vx = self.s[3]*np.cos(self.s[2])
        self.s_vy = self.s[3]*np.sin(self.s[2])
        self.s_obs_vx = self.s_obs[3]*np.cos(self.s_obs[2])
        self.s_obs_vy = self.s_obs[3]*np.sin(self.s_obs[2])
        self.p_rel = self.s[:2] - self.s_obs[:2]
        self.v_rel = matrix([ self.s_vx - self.s_obs_vx, self.s_vy - self.s_obs_vy])
        self.dist = vec_norm(self.p_rel)
        self.v_rel_norm = vec_norm(self.v_rel)
        if abs(self.dist) > abs(self.a):
            self.cone_boundary = np.sqrt(self.dist**2 - self.a**2) + ZERO_TOL
        else:
            self.cone_boundary = ZERO_TOL
        if self.dist > ZERO_TOL:
            self.cos_phi = self.cone_boundary/self.dist
        else:
            self.cos_phi = 0
    
    def update_state(self, s: matrix, s_obs: matrix, **kwargs):
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
    def from_bounding_box(cls, s: matrix = matrix(0.0, (4,1)), bbox = BoundingBox(), buffer = 0.5, **kwargs) -> CollisionCone2D:
        if not isinstance(bbox, BoundingBox):
            raise TypeError("Expected an object of type cbf.obstacles.BoundingBox as an input to fromBoundingBox() method, but got ", type(bbox).__name__)
        
        if 'id' in kwargs.keys():
            id = kwargs['id']
        
        a = np.hypot(bbox.extent.x, bbox.extent.y)
        s_obs = matrix([bbox.location.x, bbox.location.y, 0.0, bbox.velocity])
        return cls(a=a, s=s, s_obs=s_obs, buffer=buffer, id = id)
    
class PolyLane(Obstacle2DBase):
    
    def __init__(self, coefficients: np.ndarray, **kwargs):
        
        if 'id' in kwargs.keys():
            self.id = kwargs['id']
        
        self.type = Obstacle2DTypes.POLY_LANE
        self.update_coeffs(coefficients)
        
    def update_coeffs(self, coefficients: np.ndarray):
        """Update the equation coefficients. Note that this may end
        of unintentionally changing the order of the equation if the
        size of the coefficient array is not properly maintained.

        Parameters:
            coefficients (np.ndarray): Ordered array of coefficients
           
          >>> coefficients = [a0, a1, a2, ...]
          >>> f(x) = a0 + (a1 * x) + (a2 * x^2) + ... 
        """
        self.coeffs = np.asarray(coefficients)
        self._polynomial = Polynomial(coefficients)
        self._d1_polynomial = self._polynomial.deriv(m=1)
        self._d2_polynomial = self._polynomial.deriv(m=2)
        self.order = int(coefficients.size - 1)
    
    def evaluate(self, **kwargs):
        """Evaluate the value of the polynomial f(x) at one or more
        points.

        Parameters:
            \**kwargs: See below
        
        Keyword Parameters:
            x (np.ndarray): Array of data points to be evaluated

        Returns:
            np.ndarray : Array of f(x) values
        """
        x = kwargs['x']
        return self._polynomial(x)
    
    def f(self, **kwargs):
        r"""
        Alias of the evaluate function, semantically significant for cvxopt.
        
        Parameters:
        ----------
            \**kwargs: See below
        
        Keyword Parameters:
        ------------------
            x (np.ndarray): Array of data points to be evaluated

        Returns:
        -------
            np.ndarray : Array of f(x) values
        """
        x = kwargs['x']
        if np.isscalar(x):
            return self.evaluate(x)
        
    def update(self, s: matrix=None, s_obs: matrix=None, buffer: float=None, **kwargs):
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
        
        self.cx = self.get_shortest_distance_x(Point2(self.s[0], self.s[1]), x0 = s[0])
        self.g = self._polynomial(self.cx)
        # Note: g is g(x) = cy. Shortest dist pt. is (cx, cy)
        self.dg = self._d1_polynomial(self.cx)
        self.ddg = self._d2_polynomial(self.cx)
        self.eta = 1 + self.dg * self.ddg + self.dg**2 - self.s[1] * self.ddg
        
        if abs(self.eta) < ZERO_TOL:
            self.eta = ZERO_TOL
            
    def update_state(self, s: matrix, s_obs: matrix, **kwargs):
        self.update(s=s, s_obs=s_obs)
            
    def get_shortest_distance_x(self, p: Point2, x0: np.ndarray, options: dict ={'xtol': 1e-8, 'disp': False}):
        """Calculates the shortest distance point on the curve from a given point
        p = (x, y) in 2D. The problem is posed as an unconstrained optimization
        and solved using Newton Conjugate Descent from scipy.optimize.

        Parameters:
        ----------
            p (euc.Vector2): Point to calculate shortest distance from
            x0 (np.ndarray): Optimization starting point
            options (dict, optional): Dictionary of scipy.optimize options.
            Defaults to {'xtol': 1e-8, 'disp': False}

        Returns:
        -------
            (np.ndarray): Solution to the euclidean distance minimization
        """
        
        x0 = np.asarray(x0)
        
        def g(x):
            f = self._polynomial(x)
            return (x - p.x)**2 + (f - p.y)**2
        
        def dg(x):
            f = self._polynomial(x)
            df = self._d1_polynomial(x)
            return 2*(x - p.x) + 2*(f - p.y)*df
        
        def ddg(x):
            f = self._polynomial(x)
            df = self._d1_polynomial(x)
            ddf = self._d2_polynomial(x)
            return 2*(1 + df**2 + f * ddf - p.y * df)
        
        res = sci.optimize.minimize(g, x0, method='Newton-CG',
                                    jac = dg, hess = ddg,
                                    options = options)
        
        return res.x
    
    def dx(self, **kwargs):
        
        dx_ = ( 2/self.eta ) * ( (self.s[0] - self.cx) * (self.eta - 1) - (self.s[1] - self.g) * self.dg )
        return dx_
    
    def dy(self, **kwargs):
        
        dy_ = ( 2/self.eta ) * ( -(self.s[0] - self.cx) * self.dg - (self.s[1] - self.g) * (self.eta - self.dg**2) )
        return dy_
    
    # dtheta, dv and dt are zero, so we can let them be the defaults from base class.
    
    
    def update_coeffs_by_curve_fit(self,
                                   x_pts: np.ndarray, 
                                   y_pts: np.ndarray, 
                                   n: int = 3,
                                   x_fixed_pts: np.ndarray = None,
                                   y_fixed_pts: np.ndarray = None,
                                   fixed_pts_idx: np.ndarray = None,
                                   alpha: float = 0.01,
                                   sigma: np.ndarray = None,
                                   initial_coeffs: np.ndarray = None):
        
        self.update_coeffs(self.fit_polynomial_curve(x_pts,
                                                     y_pts,
                                                     n = n,
                                                     x_fixed_pts = x_fixed_pts,
                                                     y_fixed_pts = y_fixed_pts,
                                                     fixed_pts_idx = fixed_pts_idx,
                                                     alpha = alpha,
                                                     sigma = sigma,
                                                     initial_coeffs = initial_coeffs))
        
    @classmethod
    def fit_polynomial_curve(x_pts: np.ndarray, 
                             y_pts: np.ndarray, 
                             n: int = 3,
                             x_fixed_pts: np.ndarray = None,
                             y_fixed_pts: np.ndarray = None,
                             fixed_pts_idx: np.ndarray = None,
                             alpha: float = 0.01,
                             sigma: np.ndarray = None,
                             initial_coeffs: np.ndarray = None):
        
        x_pts = np.asarray(x_pts).flatten()
        y_pts = np.asarray(y_pts).flatten()
        
        if x_pts.size != y_pts.size:
            raise ValueError("Incompatible array sizes for x points and y points. \
                             Received: ", x_pts.shape, " and ", y_pts.shape)
        
        if sigma is not None:
            sigma = sigma
        else:
            sigma = np.zeros_like(x_pts)
        
        if x_fixed_pts is None and y_fixed_pts is not None:
            raise ValueError("Both fixed point arrays have to be specified. \
                             Received empty x fixed points.")
        elif x_fixed_pts is not None and y_fixed_pts is None:
            raise ValueError("Both fixed point arrays have to be specified. \
                             Received empty y fixed points.")
        else:
            x_fixed_pts = np.asarray(x_fixed_pts).flatten()
            y_fixed_pts = np.asarray(y_fixed_pts).flatten()
            x_pts = np.append(x_pts, x_fixed_pts)
            y_pts = np.append(y_pts, y_fixed_pts)
            sigma = np.append(sigma, alpha*np.ones_like(x_fixed_pts))
        
        if fixed_pts_idx is not None:
            sigma[fixed_pts_idx] = alpha
        
        if initial_coeffs is None:
            initial_coefficients = initial_coeffs
        else:
            initial_coefficients = np.zeros(n + 1)
            
        # Using the scipy curve fit function to fit the curve
        
        def func(x, *p):
            return Polynomial(p)(x)
        
        new_coeffs, _ = sci.optimize.curve_fit(func, 
                                               x_pts, 
                                               y_pts, 
                                               initial_coefficients, 
                                               sigma = sigma)
        return new_coeffs
        
    @classmethod
    def update_coeffs_by_curve_fit(cls,
                                   x_pts: np.ndarray, 
                                   y_pts: np.ndarray, 
                                   n: int = 3,
                                   x_fixed_pts: np.ndarray = None,
                                   y_fixed_pts: np.ndarray = None,
                                   fixed_pts_idx: np.ndarray = None,
                                   alpha: float = 0.01,
                                   sigma: np.ndarray = None,
                                   initial_coeffs: np.ndarray = None):
        
        return cls(PolyLane.fit_polynomial_curve(x_pts,
                                                 y_pts,
                                                 n = n,
                                                 x_fixed_pts = x_fixed_pts,
                                                 y_fixed_pts = y_fixed_pts,
                                                 fixed_pts_idx = fixed_pts_idx,
                                                 alpha = alpha,
                                                 sigma = sigma,
                                                 initial_coeffs = initial_coeffs))
    
        
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
                        self.__setitem__(key, Ellipse2D.from_bounding_box(bbox=bbox, buffer=buffer, id=key))
                    if obs_type == Obstacle2DTypes.COLLISION_CONE2D:
                        self.__setitem__(key, CollisionCone2D.from_bounding_box(bbox=bbox, buffer=buffer, id=key))
            
            # rm_keys = []
            # for key in self.mapping.keys():
            #     if key not in bbox_dict.keys():
            #         rm_keys.append(key)
            
            for key in list(self.mapping.keys()):
                if key not in list(bbox_dict.keys()):
                    self.pop(key)
            
            # for key in rm_keys:
            #     self.pop(key)

    def update_state(self, s: matrix, s_obs_dict: dict=None, buffer: float=None, **kwargs):
        
        if s_obs_dict is None:                
            for obstacle in self.mapping.values():            
                obstacle.update(s=s, s_obs=s_obs_dict, buffer=buffer, **kwargs)
        else:
            if isinstance(s_obs_dict, dict):
                for key in s_obs_dict.keys():
                    if key in self.mapping.keys():
                        self.mapping.update(s=s, s_obs=s_obs_dict[key], buffer=buffer, **kwargs)
                    else:
                        warnings.warn("Unknown key provided in s_obs_dict. Corresponsing key not found in the obstacle list.\
                            key: {key}")
            else:
                ValueError("Expected dictionary for obstacle dictionary")
    
    def f(self, *args, **kwargs) -> float:
        f = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            f[idx] = obs.f(**kwargs)
            idx = idx + 1
        return f

    def dx(self, *args, **kwargs) -> float:
        dx = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dx[idx] = obs.dx(**kwargs)
            idx = idx + 1
        return dx

    def dy(self, *args, **kwargs) -> float:
        dy = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dy[idx] = obs.dy(**kwargs)
            idx = idx + 1
        return dy
    
    def dtheta(self, *args, **kwargs) -> float:
        dtheta = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dtheta[idx] = obs.dtheta(**kwargs)
            idx = idx + 1
        return dtheta
    
    def dv(self, *args, **kwargs) -> float:
        dv = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dv[idx] = obs.dv(**kwargs)
            idx = idx + 1
        return dv
    
    def dt(self, *args, **kwargs) -> float:
        dt = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dt[idx] = obs.dt(**kwargs)
            idx = idx + 1
        return dt
    
    def dbeta(self, *args, **kwargs) -> float:
        dbeta = matrix(0.0, (len(self.mapping), 1))
        idx = 0
        for obs in self.mapping.values():
            dbeta[idx] = obs.dbeta(**kwargs)
            idx = idx + 1
        return dbeta

    def gradient(self, *args, **kwargs) -> float:
        df = matrix(0.0, (len(self.mapping), 3))
        idx = 0
        for obs in self.mapping.values():
            df[idx,:] = obs.gradient(**kwargs).T
            idx = idx + 1
        return df
