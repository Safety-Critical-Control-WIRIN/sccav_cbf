#!/bin/python3
"""

Generates uniformly distributed obstacles over the periphery of a circle
with fixed radius and attached to the ego vehicle for testing of the TV-CBFs.
The vehicles are treated as obstacles through the cbf library and the CBFs
are solved through the cvxopt.qp solver, instead of the cp solver used by 
other programs.

author: Neelaksh Singh [https://www.github.com/TheGodOfWar007]

"""

import sys
import os

import numpy as np
import euclid as geometry
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

matplotlib.use('Qt5Agg') # requires PyQt5 installed.
# Custom Imports
import stanley_controller_ellipse as sce

from cvxopt import solvers, matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.obstacles import Ellipse2D, ObstacleList2D
except:
    raise

# Global Variables used throughout the code.
_OBSTACLE_COUNT = 1
_OBSTACLE_SPAWN_INTERVAL = 10.0
_ZERO_TOL = 1e-3
_ANIMATION_FPS = 30.0
_OBSTACLE_RADIUS_RANGE = geometry.Vector2(1.5, 2.0)
_SPAWN_RADIUS_RANGE = geometry.Vector2(10.00, 20.00)

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
        self.is_seeker = [False for ii in range(self.max_count)]
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
                                        **obstacle_patch_options) for ii in range(self.max_count + 2)]
        pass

    def spawn_obstacle(self, 
                       seeker: bool=True,
                       theta: float=None,
                       radius: float=None,
                       spawn_radius: float=None):
        
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
        pass

    def update_seekers(self, dt: float, v: geometry.Vector2=None, k: float=0.2):
        """
        Will update the obstacle CBF objects and the corresponding patches
        for the animation and the dynamic obstacle CBF simulation. If `v` is
        not provided then the default proportional law is used for seeking
        velocity `k * (xy_ego - xy_obs)`.

        ### Parameters:
        
            dt (float): The update time step.
            v (euclid.Vector2, optional): Velocity of the seekers. Defaults to None.
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

            # Updating obstacle velocity
            if v is None:
                v_mag = k * np.hypot(self.state.x - center.x, self.state.y - center.y)
                obs_object.update_velocity_by_magnitude(self.state.v)
            else:
                obs_object.update_velocity(v)     

            obs_v = obs_object.vel

            # Updating Obstacle Circle's center coords
            # Obstacle Object
            center.x = center.x + obs_v.x * dt
            center.y = center.y + obs_v.y * dt
            # Matplotlib Patch
            obs_patch.set_center(center)
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
    
    # Setting the state as a property so that explicit attribute assignment
    # updates the annulus as well.
    state = property(update_state)

# Declaring Matplotlib animation artists in the global scope
# to not lose sight of the artist's pointer.
_fig, _ax = plt.subplots()

# Defining the properties of the spawn annulus and the obstacle patch.
_annulus_options = dict(
    animated = True,
    fill = False,
    fillcolor = None,
    edgecolor = 'k',
    linestyle = '--',
    linewidth = 1.50
)

_obs_patch_options = dict(
    animated = True,
    fill = True,
    fillcolor = 'r',
    edgecolor = 'b',
    linestyle = '-',
    linewidth = 1.00
)

_initial_ego_state = sce.State()

obstacle_spawner = RadialObstacleSpawner(
    state = _initial_ego_state,
    spawn_radius = None,
    obstacle_radius_range = _OBSTACLE_RADIUS_RANGE,
    spawn_radius_range = _SPAWN_RADIUS_RANGE,
    obstacle_max_count = _OBSTACLE_COUNT,
    spawn_annulus_options = _annulus_options,
    obstacle_patch_options = _obs_patch_options
)

def main():
    dt = 1/_ANIMATION_FPS
    ego_state = sce.State()
    
    pass

if __name__ == '__main__':
    main()