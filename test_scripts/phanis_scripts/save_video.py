#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 13:13:15 2022

@author: stoch-lab
"""

import cv2
import os

image_folder = 'figures'
video_name = 'stationay_vehicle.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()


frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


video = cv2.VideoWriter(video_name, 0, 5, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()