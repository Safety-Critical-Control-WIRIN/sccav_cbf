#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import math
import os
import sys
import random
import cv2
import pygame
import euclid
import enum

try:
    import queue
except ImportError:
    import Queue as queue

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle, Ellipse
from matplotlib.transforms import Affine2D
from cvxopt import solvers
from euclid import *

try:
    sys.path.append(
        glob.glob('C:\Shyam\CARLA_0.9.13\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Custom Imports
from obstacle_map import ObstacleMap
from bezier_path import Bezier

sys.path.append(os.path.dirname(os.path.abspath('')) +
                "../../")

try:
    from cbf.cbf import KBM_VC_CBF2D, DBM_CBF_2DS
    from cbf.controllers import LateralStanley, PID1
    from cbf.obstacles import Ellipse2D, CollisionCone2D, BoundingBox
    from cbf.geometry import Rotation as Rot
except:
    raise


# from __ import bbox
IMG_WIDTH = 1280
IMG_HEIGHT = 720
DEGREE_TO_RADIANS = math.pi/180

# Suppressing the cvxopt solver output
solvers.options['show_progress'] = False

class CodeOptions(enum.Enum):
    """
    Enumerations used throughout this code to modify and show
    different behaviors.
    """
    
    # Information Print Options
    PRINT_PHYSICS_PROPERTIES = False
    PRINT_EGO_TIRE_FRICTION = False
    PRINT_TORQUE_CURVE = False
    PRINT_REF_CMD_STATE = True
    PRINT_LINE = True
    PRINT_CONTROLLER_OUTPUT = True
    PRINT_OBSTACLE_LIST = True
    PRINT_EGO_FRONT_AXLE_COORDS = False
    
    # Simulation Behavior Options
    INSTANTLY_START_EGO_ENGINE = True
    
    # CBF Options
    VELOCITY_CBF = 0
    ACCELERATION_CBF = 1
    COLLISION_CONE_CBF = 2
    pass

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


def main():
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (IMG_WIDTH, IMG_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    world = client.get_world()

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        start_pose.location.x = -40.1
        start_pose.location.y = 99.5
        start_pose.location.z = 0.1
        start_pose.rotation.yaw = -90

        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.citroen.c3')),
            start_pose)
        ego = vehicle
        actor_list.append(vehicle)
        # vehicle.set_simulate_physics(False)

        ### PHYSICS PARAMETERS OF THE CAR ###
        vehicle_physics = vehicle.get_physics_control()
        vehicle_com = vehicle_physics.center_of_mass
        fl_wheel = vehicle_physics.wheels[0]
        fr_wheel = vehicle_physics.wheels[1]
        
        bl_wheel = vehicle_physics.wheels[2]
        bl_to_fl = fl_wheel.position - bl_wheel.position
        
        front_axle_center = euclid.Vector2((fl_wheel.position.x + fr_wheel.position.x)/2, (fl_wheel.position.y + fr_wheel.position.y)/2)/100

        lf = np.hypot(front_axle_center.x - start_pose.location.x, front_axle_center.y - start_pose.location.y)
        lr = abs(bl_wheel.position.x/100 - start_pose.location.x)

        ego_steer_curve = vehicle_physics.steering_curve
        steer_curve_v = [p.x for p in ego_steer_curve]
        steer_curve_steer = [p.y for p in ego_steer_curve]

        # Each wheel has a different steer angle, therefore
        # we will take an avg of the max of FL and FR wheels.
        max_steer_angle = (fl_wheel.max_steer_angle + fr_wheel.max_steer_angle)/2
        convert_rad_to_steer = 180.0 / max_steer_angle / np.pi
        
        if CodeOptions.PRINT_PHYSICS_PROPERTIES:
            print("L: ", np.sqrt(bl_to_fl.x**2 + bl_to_fl.y**2 + bl_to_fl.z**2)/100)
            print(f"lf: {lf}")
            print(f"steer_curve_x: {steer_curve_v}")
            print(f"steer_curve_y: {steer_curve_steer}")
            print(f"avg_max_steer_angle: {max_steer_angle}")
            print(f"Vehicle COM position: x:{vehicle_com.x}, y:{vehicle_com.y}, z:{vehicle_com.z}")
            print(f"ego bbox extents: x:{ego.bounding_box.extent.x}, y:{ego.bounding_box.extent.y}, z: {ego.bounding_box.extent.z}")
        #####################################

        ### FOR INSTANT START OF VEHICLE ###
        # ref: https://github.com/carla-simulator/carla/issues/1640
        
        cmd_control = carla.VehicleControl()
        
        if CodeOptions.INSTANTLY_START_EGO_ENGINE:
            cmd_control = carla.VehicleControl(manual_gear_shift=True, gear=1)
            vehicle.apply_control(cmd_control)
            
        if CodeOptions.PRINT_EGO_TIRE_FRICTION:
            for i in range(4):
                print(vehicle_physics.wheels[i].tire_friction)
                
        if CodeOptions.PRINT_TORQUE_CURVE:
            for i in range(len(vehicle_physics.torque_curve)):
                print(vehicle_physics.torque_curve[i].x)
                print(vehicle_physics.torque_curve[i].y)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", f"{IMG_WIDTH}")
        camera_bp.set_attribute("image_size_y", f"{IMG_HEIGHT}")
        camera_rgb = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        start_pose.location.x = -40.1
        start_pose.location.y = 40.5
        start_pose.rotation.yaw = -90
        obstacle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.audi.etron')),
            start_pose)
        # obstacle.set_velocity(carla.Vector3D(0, 5, 0))
        actor_list.append(obstacle)
        
        obstacle_initial_ellipse2d = Ellipse2D(obstacle.bounding_box.extent.x,
                                       obstacle.bounding_box.extent.y,
                                       euclid.Vector2(start_pose.location.x,
                                                      start_pose.location.y),
                                       start_pose.rotation.yaw,
                                       buffer=1.0)
        
        ## Create the Bounding Box object here ##
        # of type cbf.obstacles.BoundingBox
        actor_location = obstacle.get_transform().location
        actor_rotation = obstacle.get_transform().rotation
        actor_velocity = obstacle.get_velocity()
        bbox = BoundingBox(extent = Vector3(obstacle.bounding_box.extent.x, obstacle.bounding_box.extent.y, obstacle.bounding_box.extent.z),
                           location = Vector3(actor_location.x, actor_location.y, actor_location.z),
                           rotation = Rot(roll = actor_rotation.roll*DEGREE_TO_RADIANS, pitch = actor_rotation.pitch*DEGREE_TO_RADIANS, yaw = actor_rotation.yaw*DEGREE_TO_RADIANS, right_handed=False),
                           velocity = actor_velocity.length())
        ##
        
        obstacle_ccone2d = CollisionCone2D.from_bounding_box(bbox, buffer=0.5)

        """Defining the trajectory"""
        start_x = -45.1  # [m]
        start_y = 99.5  # [m]
        start_yaw = np.radians(-90.0)  # [rad]

        end_x = -45.1  # [m]
        end_y = -40  # [m]
        end_yaw = np.radians(-90.0)  # [rad]
        # offset = 3.0
        resolution = 100
        velocity = 15

        # bezier = Bezier(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset, resolution=100)
        # curve = bezier.get_trajectory(velocity= velocity)

        straight1 = []
        for t in np.linspace(66.2, 46.2, resolution):
            straight1.append((5.3, t, -math.pi/2, velocity))

        # straight2 = []
        # for t in np.linspace(28.7, 100.0, resolution):
        #     straight2.append((t, 7.0, 0, velocity))

        trajectory = straight1
        obstacle_map = ObstacleMap(ego, world, trajectory, range=60)
        
        # The CBF Controller Object
        CBF_MODE = CodeOptions.COLLISION_CONE_CBF
        
        if CBF_MODE == CodeOptions.VELOCITY_CBF:
            gamma = 5.0
            cbf_controller = KBM_VC_CBF2D()
            L = ego.bounding_box.extent.x * 2
            cbf_controller.set_model_params(L=L)
            
        if CBF_MODE == CodeOptions.ACCELERATION_CBF:
            lateral_stanley = LateralStanley(lr=lr, lf=lf, k=1.2, ks = 10)
            lateral_stanley.set_trajectory(trajectory)

            acc_pid = PID1(kp = 1.0, kd = 0.01, ki = 0.01)
            # acc_pid = PID1(kp = 1.0)        
            
            cbf_controller = DBM_CBF_2DS()
            cbf_controller.set_model_params(lr=lr, lf=lf)
            cbf_controller.obstacle_list2d.update({0: obstacle_initial_ellipse2d})
            
        if CBF_MODE == CodeOptions.COLLISION_CONE_CBF:
            lateral_stanley = LateralStanley(lr=lr, lf=lf, k=1.2, ks = 10)
            lateral_stanley.set_trajectory(trajectory)

            acc_pid = PID1(kp = 1.0, kd = 0.01, ki = 0.01)
            # acc_pid = PID1(kp = 1.0)        
            
            cbf_controller = DBM_CBF_2DS()
            cbf_controller.set_model_params(lr=lr, lf=lf)
            cbf_controller.obstacle_list2d.update({0: obstacle_ccone2d})

        current_t = 0
        previous_t = 0
        throttle = 0
        throttle_previous = 0
        brake = 0
        brake_previous = 0
        delta = 0
        delta_previous = 0

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            while True:
                n = 0
                if should_quit():
                    return
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)
                current_t = snapshot.timestamp.elapsed_seconds

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

                ### VEL CBF PART ... ###

                if CBF_MODE == CodeOptions.VELOCITY_CBF:
                    # Using obstacles_list to update CBF constraints
                    cbf_controller.obstacle_list2d.update_by_bounding_box(bbox_dict=obstacles_list, buffer=8 + L/2)
                    # Updating ego state (global coords) in CBF
                    p = euclid.Vector2(ego.get_transform().location.x, ego.get_transform().location.y)
                    cbf_controller.update_state(p=p, theta=ego.get_transform().rotation.yaw)

                    # Setting local velocity
                    v_ref = 10.0  # m/s
                    delta_ref = 0.0  # rad/s
                    u_ref = np.array([v_ref, delta_ref])

                    if len(cbf_controller.obstacle_list2d) < 1:
                        u = u_ref
                        v_cbf = v_ref
                        w_cbf = v_ref * np.tan(delta_ref) / L
                    else:
                        solver_op, u = cbf_controller.solve_cbf(u_ref)
                        u_slv = solver_op['x']
                        v_cbf = u[0]
                        w_cbf = u_slv[1]

                    # if (n%30 == 0):
                    #     print("v: ", v_cbf, " delta: ", u[1])
                    #     print("old v: ", v_ref, " old delta: ", 0)
                    #     # print(cbf_controller.obstacle_list2d)
                    #     print("L: ", L, " p: ", p)
                    #     print(cbf_controller.obstacle_list2d)
                    #     print("f: ", cbf_controller.obstacle_list2d.f(p).T)

                    vehicle.set_velocity(carla.Vector3D(0, -v_cbf, 0))
                    vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
                    # Monitoring change
                    zero_tol = 1e-3
                    if abs(v_cbf - v_ref) > zero_tol:
                        obstacle_map.cbf_active = True
                    else:
                        obstacle_map.cbf_active = False

                ### VEL CBF PART ^^^ ###

                if CBF_MODE == CodeOptions.ACCELERATION_CBF:
                    
                    # State and Obstacle List Update
                    # cbf_controller.obstacle_list2d.update_by_bounding_box(bbox_dict=obstacles_list)
                    # Updating ego state (global coords) in CBF
                    p = euclid.Vector2(ego.get_transform().location.x, ego.get_transform().location.y)
                        
                    # Lateral Stanley

                    ego_v = obstacle_map.ego_v
                    ego_v = np.sqrt(ego_v.x**2 + ego_v.y**2 + ego_v.z**2)
                    
                    cbf_controller.update_state(p=p, v=ego_v, theta=obstacle_map.ego_yaw * np.pi / 180)
                    u_ref = np.array([0.0, 0.0])
                    
                    front_axle_center = euclid.Vector2((fl_wheel.position.x + fr_wheel.position.x)/2, (fl_wheel.position.y + fr_wheel.position.y)/2)/100
                    
                    bbox_front_center = euclid.Vector2(obstacle_map.ego_x + obstacle_map.ego_width*np.cos(obstacle_map.ego_yaw)/2,\
                        obstacle_map.ego_y + obstacle_map.ego_height*np.sin(obstacle_map.ego_yaw)/2)
                    ### LATERAL STANLEY ###
                    lateral_stanley.update_state(obstacle_map.ego_x, obstacle_map.ego_y,\
                        obstacle_map.ego_yaw * np.pi / 180, ego_v)

                    # delta, target_idx = lateral_stanley.control(front_coords=front_axle_center)
                    # delta, target_idx = lateral_stanley.control(front_coords=bbox_front_center)
                    delta, target_idx = lateral_stanley.control(initial_yaw=-np.pi)
                    # max_steer = steer_curve_steer[get_closest_idx(ego_v, steer_curve_v)]
                    max_steer = 1
                    delta = delta * convert_rad_to_steer
                    u_ref[1] = delta
                    
                    # if delta > 0:
                    #     delta = max(0.0, min(delta, max_steer))
                    #     if delta - delta_previous > 0.1:
                    #         delta = delta_previous + 0.1
                    # else:
                    #     delta = max(-max_steer, min(delta, 0.0))
                    #     if abs(delta - delta_previous) > 0.1:
                    #         delta = delta_previous - 0.1
                    # delta_previous = delta
                    
            
                    #######################

                    ### ACC PID ###
                    max_acc = 20
                    # acc_pid.set_dt(snapshot.timestamp.delta_seconds)
                    acc_pid.set_dt(current_t - previous_t)
                    previous_t = current_t
                    current_ref = trajectory[target_idx]
                    
                    # u_a is the acceleration equivalent o/p of PID
                    u_a = acc_pid.control(ego_v, current_ref[3])
                    u_ref[0] = u_a
                    ###############
                    
                    # Solving the CBF's QP
                    if len(cbf_controller.obstacle_list2d) < 1:
                        u = u_ref
                    else:
                        u = cbf_controller.solve_cbf(u_ref)
                        
                    u_a_cbf = u[0]
                    delta_cbf = u[1]
                                            
                    if u_a_cbf > 0:
                        throttle = np.tanh(u_a_cbf)
                        throttle = max(0.0, min(1.0, throttle)) # saturation
                        if throttle - throttle_previous > 0.1:
                            throttle = throttle_previous + 0.1 # constraining throttle increase rate
                    else:
                        throttle = 0
                        brake = -np.tanh(u_a_cbf)
                        brake = max(0.0, min(1.0, brake)) # saturation
                        if brake - brake_previous > 0.1:
                            brake = brake_previous + 0.1 # constraining throttle increase rate

                    throttle_previous =  throttle
                    brake_previous = brake
                    
                    if delta_cbf > 0:
                        delta_cbf = max(0.0, min(delta_cbf, max_steer))
                    else:
                        delta_cbf = max(-max_steer, min(delta_cbf, 0.0))
                        
                    cmd_control.throttle = throttle
                    cmd_control.steer = delta_cbf
                    cmd_control.brake = brake
                    cmd_control.manual_gear_shift = False
                    
                    vehicle.apply_control(cmd_control)

                if CBF_MODE == CodeOptions.COLLISION_CONE_CBF:
                    
                    # State and Obstacle List Update
                    # cbf_controller.obstacle_list2d.update_by_bounding_box(bbox_dict=obstacles_list)
                    # Updating ego state (global coords) in CBF
                    p = euclid.Vector2(ego.get_transform().location.x, ego.get_transform().location.y)
                        
                    # Lateral Stanley

                    ego_v = obstacle_map.ego_v
                    ego_v = np.sqrt(ego_v.x**2 + ego_v.y**2 + ego_v.z**2)
                    
                    cbf_controller.update_state(p=p, v=ego_v, theta=obstacle_map.ego_yaw * np.pi / 180)
                    u_ref = np.array([0.0, 0.0])
                    
                    front_axle_center = euclid.Vector2((fl_wheel.position.x + fr_wheel.position.x)/2, (fl_wheel.position.y + fr_wheel.position.y)/2)/100
                    
                    bbox_front_center = euclid.Vector2(obstacle_map.ego_x + obstacle_map.ego_width*np.cos(obstacle_map.ego_yaw)/2,\
                        obstacle_map.ego_y + obstacle_map.ego_height*np.sin(obstacle_map.ego_yaw)/2)
                    ### LATERAL STANLEY ###
                    lateral_stanley.update_state(obstacle_map.ego_x, obstacle_map.ego_y,\
                        obstacle_map.ego_yaw * np.pi / 180, ego_v)

                    # delta, target_idx = lateral_stanley.control(front_coords=front_axle_center)
                    # delta, target_idx = lateral_stanley.control(front_coords=bbox_front_center)
                    delta, target_idx = lateral_stanley.control(initial_yaw=-np.pi)
                    # max_steer = steer_curve_steer[get_closest_idx(ego_v, steer_curve_v)]
                    max_steer = 1
                    delta = delta * convert_rad_to_steer
                    u_ref[1] = delta
                    
                    # if delta > 0:
                    #     delta = max(0.0, min(delta, max_steer))
                    #     if delta - delta_previous > 0.1:
                    #         delta = delta_previous + 0.1
                    # else:
                    #     delta = max(-max_steer, min(delta, 0.0))
                    #     if abs(delta - delta_previous) > 0.1:
                    #         delta = delta_previous - 0.1
                    # delta_previous = delta
                    
            
                    #######################

                    ### ACC PID ###
                    max_acc = 20
                    # acc_pid.set_dt(snapshot.timestamp.delta_seconds)
                    acc_pid.set_dt(current_t - previous_t)
                    previous_t = current_t
                    current_ref = trajectory[target_idx]
                    
                    # u_a is the acceleration equivalent o/p of PID
                    u_a = acc_pid.control(ego_v, current_ref[3])
                    u_ref[0] = u_a
                    ###############
                    
                    # Solving the CBF's QP
                    if len(cbf_controller.obstacle_list2d) < 1:
                        u = u_ref
                    else:
                        ## fill s and s_obs here ##
                        # s     -> ego state [ego.x, ego.y, ego.yaw, ego.v]
                        # s_obs -> obs state [obs.x, obs.y, obs.yaw, obs.v]
                        s = np.array([obstacle_map.ego_x, obstacle_map.ego_y, obstacle_map.ego_yaw*DEGREE_TO_RADIANS, obstacle_map.ego_v])
                        obstacle_location = obstacle.get_transform().location
                        obstacle_rotation = obstacle.get_transform().rotation
                        obstacle_velocity = obstacle.get_velocity()
                        s_obs = np.array([obstacle_location.x, obstacle_location.y, obstacle_rotation.yaw*DEGREE_TO_RADIANS, obstacle_velocity])
                        ##
                        
                        for obstacle in cbf_controller.obstacle_list2d.values:
                            obstacle.update(s = s, s_obs = s_obs)
                        
                        u = cbf_controller.solve_cbf(u_ref)
                        
                    u_a_cbf = u[0]
                    delta_cbf = u[1]
                                            
                    if u_a_cbf > 0:
                        throttle = np.tanh(u_a_cbf)
                        throttle = max(0.0, min(1.0, throttle)) # saturation
                        if throttle - throttle_previous > 0.1:
                            throttle = throttle_previous + 0.1 # constraining throttle increase rate
                    else:
                        throttle = 0
                        brake = -np.tanh(u_a_cbf)
                        brake = max(0.0, min(1.0, brake)) # saturation
                        if brake - brake_previous > 0.1:
                            brake = brake_previous + 0.1 # constraining throttle increase rate

                    throttle_previous =  throttle
                    brake_previous = brake
                    
                    if delta_cbf > 0:
                        delta_cbf = max(0.0, min(delta_cbf, max_steer))
                    else:
                        delta_cbf = max(-max_steer, min(delta_cbf, 0.0))
                        
                    cmd_control.throttle = throttle
                    cmd_control.steer = delta_cbf
                    cmd_control.brake = brake
                    cmd_control.manual_gear_shift = False
                    
                    vehicle.apply_control(cmd_control)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                
                if (n % int(fps)) == 0:
                    if CodeOptions.PRINT_EGO_FRONT_AXLE_COORDS:
                        print(f"Front Axle Center: {front_axle_center}")
                        
                    if CodeOptions.PRINT_CONTROLLER_OUTPUT:
                        print(f"PID OUTPUT: {u_a} | CBF acc OUTPUT: {u_a_cbf}")
                        print(f"STANLEY STEER: {delta} | CBF STEER: {delta_cbf}")
                    
                    if CodeOptions.PRINT_OBSTACLE_LIST:
                        print(f"Obstacle List: {cbf_controller.obstacle_list2d}")
                        print(f"Obstacle Eval: {cbf_controller.obstacle_list2d.f(p=p)}")
                        
                    if CodeOptions.PRINT_REF_CMD_STATE:
                        print(f"Current Reference Pt.: {current_ref}")
                        print(f"Throttle: {throttle}, Brake: {brake}, Steer: {delta}")
                        print(f"Vehicle State | x: {obstacle_map.ego_x}, y: {obstacle_map.ego_y},\
                            yaw: {obstacle_map.ego_yaw}, v: {ego_v}")
                
                    if CodeOptions.PRINT_LINE:
                        print("------------------------------------------------------------------------------------------------------------------------")

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
                n += 1

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
