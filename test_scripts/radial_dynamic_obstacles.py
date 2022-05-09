#!/bin/python3
"""

Generates uniformly distributed obstacles over the periphery of a circle
with fixed radius and attached to the ego vehicle for testing of the TV-CBFs.
The vehicles are treated as obstacles through the cbf library and the CBFs
are solved through the cvxopt.qp solver, instead of the cp solver used by 
other programs.

author: Neelaksh Singh [https://www.github.com/TheGodOfWar007]

"""
# TODO:
#   1. Add a label to each obstacle according to its id.

import sys
import os
import warnings
import matplotlib

import numpy as np
import euclid as geometry
# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.axes as axes

matplotlib.use('Qt5Agg') # requires PyQt5 installed.

from typing import TypeVar
from cvxopt import solvers, matrix

# Custom Imports
import stanley_controller_ellipse as sce

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.obstacles import Ellipse2D, ObstacleList2D
except:
    raise

# Defining important TypeVars for type hinting.
_NARRAY = TypeVar("_NARRAY", np.ndarray, matrix)

# Global Variables used throughout the code.
_OBSTACLE_COUNT = 1
_OBSTACLE_SPAWN_INTERVAL = 10.0
_ZERO_TOL = 1e-3
_ANIMATION_FPS = 30.0
_ANIMATION_RUNTIME = 20.0 # sec
_NUM_FRAMES = int(_ANIMATION_RUNTIME * _ANIMATION_FPS)
_OBSTACLE_RADIUS_RANGE = geometry.Vector2(1.5, 2.0)
_SPAWN_RADIUS_RANGE = geometry.Vector2(10.00, 20.00)

# Controller/Vehicle associated global vars
_L = 2.9  # [m] Wheel base of vehicle
_lr = _L/2
_lf = _L - _lr
_max_steer = np.radians(30.0)  # [rad] max steering angle

class RadialObstacleSpawner():
    """
    The class that will be used to spawn obstacles randomly at regular intervals.
    Some stationary, some for the sole purpose of assassinating the ego vehicle.
    """
    def __init__(self, state: sce.State, 
                 spawn_radius: float = None, 
                 obstacle_radius_range: geometry.Vector2 = geometry.Vector2(x = 0.5, y = 1.0),
                 spawn_radius_range: geometry.Vector2 = geometry.Vector2(x = 10.0, y = 20.0),
                 obstacle_max_count: int=10,
                 spawn_annulus_options: dict = {},
                 obstacle_patch_options: dict = {}):
        
        self.r = spawn_radius
        self.state = state
        self.obs_r_min = obstacle_radius_range.x
        self.obs_r_max = obstacle_radius_range.y
        if spawn_radius is None:
            self.r_min = spawn_radius_range.x
            self.r_max = spawn_radius_range.y
        else:
            self.r_max = self.r
            self.r_min = self.r
        self.max_count = int(obstacle_max_count)
        self.obstacle_list = ObstacleList2D()
        # + 1 for spawn annulus
        self.is_seeker = [False for ii in range(self.max_count + 1)]
        self.is_spawned = self.is_seeker.copy()
        self.id_seekers = []
        self.id_spawned = []
        
        self.patches = [patches.Annulus((state.x, state.y), 
                                        self.r_max, 
                                        self.r_max - self.r_min, 
                                        angle=0.0, 
                                        **spawn_annulus_options)]
        
        self.id_head = 1
        self.patches += [patches.Circle((0, 0), 
                                        radius = 0.0,
                                        visible = False, 
                                        **obstacle_patch_options) for ii in range(self.max_count)]
        
        self._id_drawn = []
        
        pass
    
    def __str__(self):
        print_str = f"""Obstacle Spawner (id_head: {self.id_head},
        spawned id(s): {self.id_spawned},
        seeker id(s) : {self.id_seekers},
        spawn range  : ({self.obs_r_min}, {self.obs_r_max}),
        current state: {self.state},
        obstacle_list: {self.obstacle_list},
        )
        """
        return print_str

    def spawn_obstacle(self, 
                       seeker: bool=True,
                       theta: float=None,
                       radius: float=None,
                       spawn_radius: float=None):
        """Spawns an obstacle and activates the corresponding patch object
        for further updates if made a `seeker`.

        ### Parameters:
            seeker (bool, optional): Spawn a seeker. Defaults to True.
            theta (float, optional): Specifies the obstacle orientation. Defaults to None (Random).
            radius (float, optional): Radius of the obstacle. Defaults to None (Random).
            spawn_radius (float, optional): Spawn distance from ego's position. Defaults to None.

        ### Returns:
            bool: True if obstacle was successfully spawned. False otherwise. For
            e.g. if all obstacles have been exhausted and id_head has reached the
            end of the patches list.
        """
        if self.id_head > self.max_count:
            warnings.warn("All obstacles spawned. id_head has reached the end. All further spawns are cancelled.")
            return False
        
        obs_patch = self.patches[self.id_head]
        obs_patch.set_visible(True)
        
        if radius is not None:
            obs_r = radius
        else:
            obs_r = np.random.uniform(self.obs_r_min, self.obs_r_max)

        obs_patch.set_radius(obs_r)

        if theta is not None:
            theta = theta
        else:
            theta = np.random.uniform(0, 2*np.pi)
        
        if self.r is not None:
            spawn_r = self.r
        elif spawn_radius is not None:
            spawn_r = spawn_radius
        else:
            spawn_r = np.random.uniform(self.r_min, self.r_max)

        obs_x = self.state.x + spawn_r*np.cos(theta)
        obs_y = self.state.y + spawn_r*np.sin(theta)
        obs_patch.set_center((obs_x, obs_y))

        self.is_seeker[self.id_head] = seeker
        self.id_spawned.append(self.id_head)
        self.is_spawned[self.id_head] = True

        obs_yaw = np.arctan2(self.state.y - obs_y, self.state.x - obs_x)

        self.obstacle_list.update(
            {
                self.id_head: Ellipse2D(a=obs_r, 
                b=obs_r, 
                center=geometry.Vector2(obs_x, obs_y),
                theta=obs_yaw)
            }
        )
        
        if seeker:
            self.obstacle_list[self.id_head].update_velocity_by_magnitude(self.state.v)
            self.id_seekers.append(self.id_head)

        self.id_head += 1
        return True

    def update_seekers(self, dt: float, v: geometry.Vector2=None, v_min: float = 3.0, k: float=0.2):
        """
        Will update the obstacle CBF objects and the corresponding patches
        for the animation and the dynamic obstacle CBF simulation. If `v` is
        not provided then the default proportional law is used for seeking
        velocity `k * (xy_ego - xy_obs)`.

        ### Parameters:
        
            dt (float): The update time step.
            v (euclid.Vector2, optional): Velocity of the seekers. Defaults to None.
            v_min (float, optional): Min. velocity of the obstacle. Defaults to 3.0 m/s.
            k (float): Proportional law for the seeking law stated above. Defaults to 0.2.
        
        ### Returns:
            None
        """
        for id in self.id_seekers:
            obs_patch = self.patches[id]
            obs_object: Ellipse2D = self.obstacle_list[id]

            center: geometry.Vector2 = obs_object.center

            # Updating yaw (will correct velocity vector)
            obs_yaw = np.arctan2(self.state.y - center.y, self.state.x - center.x)
            obs_object.update_orientation(obs_yaw)

            # Updating obstacle velocity (will use updated yaw)
            if v is None:
                v_mag = k * np.hypot(self.state.x - center.x, self.state.y - center.y)
                if v_mag < v_min:
                    v_mag = v_min
                obs_object.update_velocity_by_magnitude(v_mag)
            else:
                obs_object.update_velocity(v)     

            obs_v = obs_object.vel

            # Updating Obstacle Circle's center coords
            # Obstacle Object
            center.x = center.x + obs_v.x * dt
            center.y = center.y + obs_v.y * dt
            # Matplotlib Patch
            obs_patch.set_center((center.x, center.y))
            # set_angle should be added if the patch is changed
            # to ellipse in the future.
        pass
    
    def update_state(self, new_state: sce.State):
        """Updates the ego state attribute and thus updates the ego
        attached annulus patch. Updating is possible by explicit
        attribute assignment, but then the annulus might not be updated.

        ### Parameters:
            new_state (State): The new ego_state to be assigned in the 
            form of the state object.
        """
        self.state = new_state
        ego_annulus: patches.Annulus = self.patches[0]
        ego_annulus.set_center((self.state.x, self.state.y))
    
    def draw_patches(self, ax: axes.Axes):
        """Used when the animated option is set to true in the 
        obstacle_patch_options or the annulus options. This method provides
        a convenient way of drawing the patches which are spawned but not drawn
        yet.

        ### Parameters:
            fig (matplotlib.figure): The figure artist.
        """
        for id in self.id_spawned:
            ax.draw_artist(self.patches[id])
        
        ax.draw_artist(self.patches[0])
    
    def add_patches(self, ax: axes.Axes):
        """Used for adding all the patches to the axes object through
        ax.add_patch. This method has a mandatory call but is provided
        as a separate method for conditional addition of patches.

        Parameters:
        ----------
            ax (axes.Axes) : The axes object to add the patches to. 
            - Allows addition to more than one axes.
        """
        for patch in self.patches:
            ax.add_patch(patch)

# Declaring Matplotlib animation artists in the global scope
# to not lose sight of the artist's pointer.
_fig = plt.figure()
_gs = _fig.add_gridspec(4, 3)
_ax = _fig.add_subplot(_gs[:3, :]) # Main animation axis
_ax1 = _fig.add_subplot(_gs[3, 0]) # Distance b/w vehicle and obstacle
_ax2 = _fig.add_subplot(_gs[3, 1]) # acceleration i/p
_ax3 = _fig.add_subplot(_gs[3, 2]) # steering i/p

_ax.set_xlim(left = -_SPAWN_RADIUS_RANGE.y, right = _SPAWN_RADIUS_RANGE.y)
_ax.set_ylim(bottom = -_SPAWN_RADIUS_RANGE.y, top = _SPAWN_RADIUS_RANGE.y)
_ax.set_xlabel("x (m)")
_ax.set_ylabel("y (m)")
_ax.set_title("Ego Vehicle and Obstacles in 2D Plane")

_ax1.set_xlabel("t (sec)")
_ax1.set_ylabel("dist(ego, obs) (m)")
_ax1.set_title("Distance to collision")

_ax2.set_xlabel("t (sec)")
_ax2.set_ylabel("a (m/s^2)")
_ax2.set_title("Acceleration cmd")

_ax3.set_xlabel("t (sec)")
_ax3.set_ylabel("delta (rad)")
_ax3.set_title("Steering cmd")

# _fig.tight_layout()

_plot_objects = []

# Defining the properties of the spawn annulus and the obstacle patch.
_annulus_options = dict(
    animated = True,
    fill = False,
    facecolor = None,
    edgecolor = 'k',
    linestyle = '--',
    linewidth = 1.50
)

_obs_patch_options = dict(
    animated = True,
    fill = True,
    facecolor = 'r',
    edgecolor = 'b',
    linestyle = '-',
    linewidth = 1.00
)

_ego_state = sce.State()

# Just like the artist, we don't want to lose sight of this either
# hence the spawner will have global scope.
obstacle_spawner = RadialObstacleSpawner(
    state = _ego_state,
    spawn_radius = None,
    obstacle_radius_range = _OBSTACLE_RADIUS_RANGE,
    spawn_radius_range = _SPAWN_RADIUS_RANGE,
    obstacle_max_count = _OBSTACLE_COUNT,
    spawn_annulus_options = _annulus_options,
    obstacle_patch_options = _obs_patch_options
)

# Ego Vehicle's marker
_ego_marker = _ax.plot(_ego_state.x, _ego_state.y, "xg", label="Ego Vehicle")[0]

## LOGGING AND PLOTTING ADDITIONAL DATA
# Extra variables for logging important values.
_t_arr = []
_ego_obstacle_separation = []
_a_cbf_list = []
_delta_cbf_list = []

_ego_obstacle_separation_plot = _ax1.plot([], [], linewidth=1.0, animated = True, color="g")[0]
_a_cbf_plot = _ax2.plot([], [], linewidth=1.0, animated = True, color="k")[0]
_delta_cbf_plot = _ax3.plot([], [], linewidth=1.0, animated = True, color="b", linestyle="--")[0]

def init_animation():
    _plot_objects = [_ego_marker] + obstacle_spawner.patches
    obstacle_spawner.add_patches(_ax)
    _ax.set_aspect('equal', adjustable='box')
    return _plot_objects

## Controller functions for modifying the vehicle control i/p(s).
def single_obstacle_CBF1(s: _NARRAY,
                         u_ref: _NARRAY,
                         c_obs: _NARRAY,
                         c_obs_dot: _NARRAY,
                         a: float, 
                         b: float, 
                         gamma: float,
                         kv: float=1.0):
    
    m = 1 # No. of Constraints
    n = 2 # Dimension of x0 i.e. u
    u_ref = matrix(u_ref)
    
    # delta to beta
    u_ref[1] = np.arctan2(_lr * np.tan(u_ref[1]), _lf + _lr)

    def F(x=None, z=None):
        if x is None: return m, matrix(0.0, (n, 1))
        # if x is None: return m, u_des    
        # for 1 objective function and 1 constraint and 3 state vars
        f = matrix(0.0, (m+1, 1))
        Df = matrix(0.0, (m+1, n))
        f[0] = (x - u_ref).T * (x - u_ref)
        # CBFs
        # partials of h
        h1 = ((s[0] - c_obs[0])/a)**2 + ((s[1] - c_obs[1])/b)**2 - 1 - (kv*s[3]/(1 + s[3]))
        # h1 = ((s[0] - c_obs[0])/a)**2 + ((s[1] - c_obs[1])/b)**2 - 1 + (kv*s[3]/(1 + s[3]))
        # h1 = ((s[0] - c_obs[0])/a)**2 + ((s[1] - c_obs[1])/b)**2 - 1 - (kv*s[3])
        # h1 = ((s[0] - c_obs[0])/a)**2 + ((s[1] - c_obs[1])/b)**2 - 1 - (kv/(1 + s[3]))
        # h1 = ((s[0] - c_obs[0])/a)**2 + ((s[1] - c_obs[1])/b)**2 - 1
        h1_dx = 2*(s[0] - c_obs[0])/(a**2)
        h1_dy = 2*(s[1] - c_obs[1])/(b**2)
        h1_dtheta = 0.0
        h1_dv = -kv/((1 + s[3])**2)
        # h1_dv = kv/((1 + s[3])**2)
        # h1_dv = -kv
        # h1_dv = kv/((1 +s[3])**2)
        # h1_dv = 0.0
        # TV-CBF
        h1_dt = -2 * ( ((s[0] - c_obs[0])/(a**2)) * c_obs_dot[0] + ((s[1] - c_obs[1])/(b**2)) * c_obs_dot[1] )
        # temp vars are written as tn
        Dh1 = matrix([h1_dx, h1_dy, h1_dtheta, h1_dv], (1, 4))
        f_c = matrix([ s[3] * np.cos(s[2]), s[3] * np.sin(s[2]), 0, 0], (4, 1))
        g_c = matrix([ [0, 0, 0, 1], [-s[3] * np.sin(s[2]), s[3] * np.cos(s[2]), s[3]/_lr, 0] ])

        f[1] = -(Dh1 * (f_c + g_c * x) + gamma * h1 + h1_dt)

        Df[0, :] = 2 * (x - u_ref).T
        Df[1, :] = -1 * (Dh1 * g_c)

        if z is None: return f, Df
        H = z[0] * 2 * matrix(np.eye(n))
        return f, Df, H
    
    solver_op = solvers.cp(F)
    u = solver_op['x']
    # beta to delta
    u[1] = np.arctan2((_lf + _lr) * np.tan(u[1]), _lr)
    
    return u

def animate(i):
    dt = 1/_ANIMATION_FPS
    t = i*dt
    _t_arr.append(t)
    
    ###
        # This implements the spawn law. For now, lets just spawn one
        # obstacle in the beginning.
    if i == 1:
        obstacle_spawner.spawn_obstacle(seeker = True)

    ###
    
    ### Control Code
        # Something Happens to Ego Vehicle and Obstacles here   #
        # Control i/p(s) are fed through CBF to the ego vehicle.#
    # The u_ref 0, stationary ego vehicle case.
    u_ref = np.array([0.0, 0.0])
    # some vars for CBF
    if obstacle_spawner.is_spawned[1]:
        s = np.array([_ego_state.x, _ego_state.y, _ego_state.yaw, _ego_state.v])
        obs1_center: geometry.Vector2 = obstacle_spawner.obstacle_list[1].center
        obs1_vel: geometry.Vector2 = obstacle_spawner.obstacle_list[1].vel
        c_obs = np.array([obs1_center.x, obs1_center.y])
        c_obs_dot = np.array([obs1_vel.x, obs1_vel.y])
        obs1_a = obstacle_spawner.obstacle_list[1].a
        obs1_b = obs1_a
        
        # CBF
        u_cbf = single_obstacle_CBF1(s = s, 
                                    u_ref = u_ref,
                                    c_obs = c_obs,
                                    c_obs_dot = c_obs_dot,
                                    a = obs1_a,
                                    b = obs1_b,
                                    gamma = 1.0,
                                    kv = 1.0)
        
        _ego_obstacle_separation.append(np.hypot(_ego_state.x - obs1_center.x,
                                                _ego_state.y - obs1_center.y))
    else:
        u_cbf = u_ref
        _ego_obstacle_separation.append(0.0)
    ###
    
    
    ### Updates to the states
        # Ego State object and obstacle patches are updated.  #
        
    # Updating ego state according to vehicle dynamics
    acc_cbf = u_cbf[0]
    delta_cbf = u_cbf[1]
    _a_cbf_list.append(acc_cbf)
    _delta_cbf_list.append(delta_cbf)
    _ego_state.update_com(acc_cbf, delta_cbf, _dt = dt)
    
    # Updating ego state in obstacle spawner
    obstacle_spawner.update_state(_ego_state)
    
    # Updating all seekers.
    obstacle_spawner.update_seekers(dt = dt)
    ###
    
    # Update Ego Marker
    _ego_marker.set_data([_ego_state.x], [_ego_state.y])
    _ego_obstacle_separation_plot.set_data(_t_arr, _ego_obstacle_separation)
    _a_cbf_plot.set_data(_t_arr, _a_cbf_list)
    _delta_cbf_plot.set_data(_t_arr, _delta_cbf_list)
    
    _ax1.set_xlim(left = 0, right = t)
    _ax1.set_ylim(bottom = min(_ego_obstacle_separation), top = max(_ego_obstacle_separation))
    
    _ax2.set_xlim(left = 0, right = t)
    _ax2.set_ylim(bottom = min(_a_cbf_list), top = max(_a_cbf_list))
    
    _ax3.set_xlim(left = 0, right = t)
    _ax3.set_ylim(bottom = min(_delta_cbf_list), top = max(_delta_cbf_list))
    
    _plot_objects = [_ego_marker, _ego_obstacle_separation_plot, _a_cbf_plot, _delta_cbf_plot] + obstacle_spawner.patches
    
    return _plot_objects

def main():
    # dt here is in ms and not in sec.
    dt = int(1000/_ANIMATION_FPS) # Has to be an integer for the interval param
    
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
    
    # Calling the matplotlib.animation.FuncAnimation object. This will
    # recursively call the animation function with frame number i which
    # will be the timestamping reference. Blitting improves the animation
    # quality by a huge extent. Try toggling it and see the difference.
    animator = animation.FuncAnimation(_fig,
                                       animate,
                                       init_func = init_animation,
                                       frames = _NUM_FRAMES,
                                       interval = dt,
                                       blit = True,
                                       repeat=False
                                       )
    
    # Render the plot/save animation
    plt.show()
    pass

if __name__ == '__main__':
    main()