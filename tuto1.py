from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

IM_WIDTH = 640

IM_HEIGHT = 480


import glob
import os
import sys

from gym import spec

from examples.vehicle_gallery import get_transform

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time
import numpy as np
import cv2

def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))


def process_img(image):
    i = np.array(image.raw_data)

    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4)) #reshape the array as 2D image matrix while raw data is RGBA

    i3 = i2[:, :, :3] #Take the necessary data [h,v, rgb]

    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0 #normalize



actor_list = []

try:

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)


    world = client.get_world()



    

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter("model3")[0]

    spawn_point = random.choice(world.get_map().get_spawn_points())

    #print(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)

    vehicle.set_autopilot(True)

    #vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    print("simstart")

    spectator = world.get_spectator()

    spectator.set_transform(get_transform(vehicle.get_location(),vehicle.get_transform().rotation.yaw-180))

    cam_bp = blueprint_library.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")


    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    cam_sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
    actor_list.append(cam_sensor)

    #cam_sensor.listen(lambda data: process_img(data))

    obs_bp = blueprint_library.find("sensor.other.obstacle")
    obs_bp.set_attribute("debug_linetrace", "True")

    obs_sensor = world.spawn_actor(obs_bp, spawn_point, attach_to=vehicle)
    actor_list.append(obs_sensor)


    while True:
        world.wait_for_tick(10.0)
        world.tick()
        world.get_spectator().set_transform(cam_sensor.get_transform())



finally:
    for actor in actor_list:
        actor.destroy()
    print("All clened up")


