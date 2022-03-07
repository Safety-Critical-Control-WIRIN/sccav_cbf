#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

from pygame import Vector2, Vector3

from cbf.geometry import Transform

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.transforms import Affine2D
import cv2
from obstacle_map import ObstacleMap

sys.path.append(os.path.dirname(os.path.abspath('')) +
                "../../")

try:
    from cbf.cbf import KBM_VC_CBF2D
    from cbf.geometry import Rotation, Transform
except:
    raise

# from __ import bbox
IMG_WIDTH = 1280
IMG_HEIGHT = 720

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def draw_image(surface, image, blend=False):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


# class CBF:

# class ObstacleMap:

def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (IMG_WIDTH, IMG_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        start_pose.location.x = -6.5
        start_pose.location.y = 60
        start_pose.location.z = 0.2

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            start_pose)
        ego = vehicle
        actor_list.append(vehicle)
        # vehicle.set_simulate_physics(False)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{IMG_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IMG_HEIGHT}")
        camera_rgb = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        obstacle_map = ObstacleMap(ego, world)

        # The CBF Controller Object
        gamma = 1.0
        cbf_controller = KBM_VC_CBF2D(gamma=gamma)
        cbf_controller.set_model_params(L=ego.bounding_box.extent.z)

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Choose the next waypoint and update the car location.
                # waypoint = random.choice(waypoint.next(1.5))
                # vehicle.set_transform(waypoint.transform)

                # Get the ObstacleMap
                img, obstacles_list = obstacle_map.get_obstacle_map()
                if img is not None:
                    cv2.imshow('obstacle map', img)
                    cv2.waitKey(1)

                
                # obstacles_list is a dictionary consisting of obstacles, 
                # their actor_id being the key and corresponding BoundingBox object being the value
                ## Using obstacles_list to update CBF constraints
                cbf_controller.obstacle_list2d.update_by_bounding_box(bbox_dict=obstacles_list)
                # Updating ego state (global coords) in CBF
                cbf_controller.update_state(p=Vector2(obstacle_map.ego_x, obstacle_map.ego_y), theta=obstacle_map.ego_yaw)

                # Get Tf b/w ego local frame and world
                ego_rotation = ego.get_transform().rotation
                ego_rotation = Rotation(ego_rotation.roll, ego_rotation.pitch, ego_rotation.yaw)
                glob_to_loc_tf = Transform(rotation=ego_rotation)

                # Setting local velocity
                v_ref = 5 # m/s
                delta_ref = 0 # rad/s
                u_ref = np.array([v_ref, delta_ref])

                # Passing u_ref through the CBF Safety Filter
                solver_op, u = cbf_controller.solve_cbf(u_ref)
                u_slv = solver_op['x']
                v_cbf = u[0]
                w_cbf = u_slv[1]

                v_cmd_local = Vector3(v_cbf, 0, 0)
                w_cmd_local = Vector3(0, 0, w_cbf)

                # Tf to world coordinates
                v_cmd_global = glob_to_loc_tf.transform_inverse(v_cmd_local)
                w_cmd_global = glob_to_loc_tf.transform_inverse(w_cmd_local)

                # velocity = carla.Vector3D(5, 0, 0)
                # ang_vel = carla.Vector3D(0, 0, 20)
                # Give command to ego actor
                vehicle.set_velocity(carla.Vector3D(v_cmd_global.x, v_cmd_global.y, v_cmd_global.z))
                vehicle.set_angular_velocity(carla.Vector3D(w_cmd_global.x, w_cmd_global.y, w_cmd_global.z))

                # Monitoring change
                zero_tol = 1e-3
                if abs(v_cbf - v_ref) > zero_tol:
                    obstacle_map.cbf_active = True

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                # draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()
                obstacle_map.refresh(ego, world)

    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
