#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:42:06 2022

@author: stoch-lab
"""
from random import randint

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

%matplotlib qt
# create empty lists for the x and y data
x = []
y = []

# create the figure and axes objects
fig, ax = plt.subplots()    

# function that draws each frame of the animation
def animate(i):
    pt = randint(1,9) # grab a random integer to be the next y-value in the animation
    x.append(i)
    y.append(pt)    

    ax.clear()
    ax.plot(x, y)
    ax.set_xlim([0,20])
    ax.set_ylim([0,10])

# run the animation
anim = FuncAnimation(fig, animate, frames=20, interval = 1000, repeat=False)

plt.show()

# saving to m4 using ffmpeg writer
writervideo = FFMpegWriter(fps=60)
anim.save('increasingStraightLine.mp4', writer=writervideo)
plt.close()

#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

img = [] # some array of images
frames = [] # for storing the generated images
fig = plt.figure()
for i in range(6):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
# ani.save('movie.mp4')
plt.show()