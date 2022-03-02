#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:00:44 2022

@author: stoch-lab
"""

#%%


import math
import numpy as np
import matplotlib.pyplot as plt

t_param = np.linspace(0, 10, 1000) #
x_values = t_param
y_values = t_param

c_x_t = 7     #x-position of the center
c_y_t = 3     #y-position of the center
r = 1       #radius 

## Plotting
fig, ax = plt.subplots()
plt.gca().set_aspect('equal')

plt.plot(x_values, y_values, color = 'salmon')  # Plotting the required trajectory

plt.grid(color='lightgray',linestyle='--')
plt.title("Desired Path")
plt.xlabel("x(t)")
plt.ylabel("y(t)")

#%%

def point_wrt_circle(x, y, c_x_t, c_y_t, r):
    return pow((x - c_x_t), 2) + pow((y - c_y_t), 2) - pow(r,2)
    

# Initialization of state variables
x_t = 0
y_t = 0
theta_t = math.atan(1)

# Parameters
delta_t = 0.1
gamma = 5.0

# Temporary variables for debugging

c_x_t_dot = -0.5
c_y_t_dot = 0.5

for t in np.arange(0, 10, delta_t):
  
    # Updating state for next time step
    c_x_t = c_x_t + c_x_t_dot*delta_t
    c_y_t = c_y_t + c_y_t_dot*delta_t

    
    t_param = np.linspace(0, 10, 1000)
    x_values_ellipse = c_x_t + r*np.cos(t_param) 
    y_values_ellipse = c_y_t + r*np.sin(t_param)


    ## Plotting
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')

    plt.plot(x_values, y_values, color = 'salmon')  # Plotting the required trajectory

    plt.plot( x_values_ellipse, y_values_ellipse , 'black' )   # Plotting the obstacle 
    plt.fill( x_values_ellipse, y_values_ellipse , 'darkred')

    plt.grid(color='lightgray',linestyle='--')
    plt.title("Desired Path")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    
    plt.show()





