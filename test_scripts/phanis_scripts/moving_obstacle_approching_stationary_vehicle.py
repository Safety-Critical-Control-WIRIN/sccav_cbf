#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:31:29 2022

@author: stoch-lab
"""

#%%
import math
import numpy as np
import matplotlib.pyplot as plt

figures_folder_names = r"figures"
save_figure = False

t = np.linspace(0, 10, 1000)
x_values = t
y_values = t

c_x_t = 6     #x-position of the center
c_y_t = 6     #y-position of the center
r = 1       #radius 


t = np.linspace(0, 10, 1000)
x_values_ellipse = c_x_t + r*np.cos(t) 
y_values_ellipse = c_y_t + r*np.sin(t)


## Plotting
#%matplotlib inline
fig, ax = plt.subplots()
#plt.xlim(-2, 28)
#plt.ylim(-2, 10)
plt.gca().set_aspect('equal')

plt.plot(x_values, y_values, color = 'salmon')  # Plotting the required trajectory

plt.plot( x_values_ellipse, y_values_ellipse , 'black')   # Plotting the obstacle 
plt.fill( x_values_ellipse, y_values_ellipse , 'darkred')

plt.grid(color='lightgray',linestyle='--')
plt.title("Desired Path")
plt.xlabel("x(t)")
plt.ylabel("y(t)")

#%%

def point_wrt_circle(x, y, c_x_t, c_y_t, r):
    return pow((x - c_x_t), 2) + pow((y - c_y_t), 2) - pow(r,2)
    

# Initialization of state variables
x_t = 4
y_t = 4
theta_t = math.atan(1)

# Parameters
gamma = 0.01
delta_t = 0.01
time_start = 0
time_end = 40

# Temporary variables for debugging
x_list = []
y_list = []
x_red_list = []
y_red_list = []
x_green_list = []
y_green_list = []

theta_list = []
psi_list = []
u_s_star_list = []
u_omega_star_list = []
p = 0
n = 0
counter = 0
c_x_t_dot = -0.5
c_y_t_dot = -0.5

for t in np.arange(time_start, time_end, delta_t):

    # Reference controls
    u_s_ref = 0 #math.sqrt(2)
    u_omega_ref = 0
    
    # Terms in QP
    A_1_1 = 2*(x_t - c_x_t)*math.cos(theta_t) + 2*(y_t - c_y_t)*math.sin(theta_t)
    point_dist_wrt_circle = point_wrt_circle(x_t, y_t, c_x_t, c_y_t, r)
    added_term = (2*(x_t - c_x_t)*c_x_t_dot + 2*(y_t - c_y_t)*c_y_t_dot)
    psi =  A_1_1*2*u_s_ref + gamma*point_dist_wrt_circle - added_term
    ratio = (added_term)/point_dist_wrt_circle
    #print(ratio)

    """
    if A_1_1 > 0 and point_dist_wrt_ellipse > 0:
        plt.scatter(x_t, y_t, color = 'green')
    elif A_1_1 > 0 and not (point_dist_wrt_ellipse > 0):
        plt.scatter(x_t, y_t, color = 'blue')        
    elif not (A_1_1 > 0) and point_dist_wrt_ellipse > 0:
        plt.scatter(x_t, y_t, color = 'orange')      
    else:
        plt.scatter(x_t, y_t, color = 'red')
    """  
    
    # Optimal Contorl    
    if psi < 0:
        print("active")
        u_s_star = u_s_ref - psi/(2*A_1_1)
        u_omega_star = u_omega_ref
    else:
        print("Inactive")
        u_s_star = u_s_ref
        u_omega_star = u_omega_ref
    
    # Vehicle Dynamics
    x_dot = u_s_star * math.cos(theta_t)
    y_dot = u_s_star * math.sin(theta_t)
    theta_dot = u_omega_star
    
    # Updating state for next time step
    x_t = x_t + x_dot*delta_t
    y_t = y_t + y_dot*delta_t
    theta_t = theta_t + theta_dot*delta_t
    c_x_t = c_x_t + c_x_t_dot*delta_t
    c_y_t = c_y_t + c_y_t_dot*delta_t
    
    # Plotting
    t_param = np.linspace(0, 10, 1000)
    x_values_ellipse = c_x_t + r*np.cos(t_param) 
    y_values_ellipse = c_y_t + r*np.sin(t_param)

    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')

    plt.plot(x_values, y_values, color = 'salmon')  # Plotting the required trajectory

    plt.plot( x_values_ellipse, y_values_ellipse , 'black' )   # Plotting the obstacle 
    plt.fill( x_values_ellipse, y_values_ellipse , 'darkred')
    
    plt.scatter(x_red_list, y_red_list, color = 'red')  
    plt.scatter(x_green_list, y_green_list, color = 'green')      
    
    plt.grid(color='lightgray',linestyle='--')
    plt.title("Desired Path")
    plt.xlabel("x(t)")
    plt.ylabel("y(t)")
    
    # Saving and showing plot
    if save_figure:
        file_name = figures_folder_names + "/file"+str(counter).zfill(3)+".png" 
        plt.savefig(file_name, bbox_inches='tight')
        counter += 1
    # plt.pause(delta_t)
    # plt.show()
    
    
    # For Debugging
    x_list.append(x_t)
    y_list.append(y_t)
    if psi < 0:
        x_red_list.append(x_t)
        y_red_list.append(y_t)
    else:
        x_green_list.append(x_t)
        y_green_list.append(y_t)
    
    theta_list.append(theta_t)
    psi_list.append(psi)
    u_s_star_list.append(u_s_star)
    u_omega_star_list.append(u_omega_star)

#%%

