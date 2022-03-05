#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Feb  28 11:56:15 2022

@author: madhusudhan
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
x_values = t
y_values = t

c_x = 5     #x-position of the center
c_y = 5     #y-position of the center
r = 1       #radius 


t = np.linspace(0, 10, 1000)
x_values_ellipse = c_x + r*np.cos(t) 
y_values_ellipse = c_y + r*np.sin(t)


## Plotting
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

def point_wrt_circle(x, y, c_x, c_y, r):
    return pow((x - c_x), 2) + pow((y - c_y), 2) - pow(r,2)
    

# Initialization of state variables
x_t = 0
y_t = 0
theta_t = math.atan(1)

# Parameters
delta_t = 0.1
gamma = 2

# Temporary variables for debugging
x_list = []
y_list = []
theta_list = []
psi_list = []
u_s_star_list = []
u_omega_star_list = []
p = 0
n = 0

u_s_ref = math.sqrt(2)
u_s_dot = 0.1

for t in np.arange(0, 10, delta_t):
  
    # Reference controls
    u_s_ref = u_s_ref + u_s_dot*delta_t
    u_omega_ref = 0
    
    # Terms in QP
    A_1_1 = 2*(x_t - c_x)*math.cos(theta_t) + 2*(y_t - c_y)*math.sin(theta_t)
    point_dist_wrt_circle = point_wrt_circle(x_t, y_t, c_x, c_y, r)
    psi =  A_1_1*2*u_s_ref + gamma*point_dist_wrt_circle
   
    
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
    
    if psi < 0:
        plt.scatter(x_t, y_t, color = 'red')  
        
        # Optimal Contorl    
        u_s_star = u_s_ref - psi/(2*A_1_1)
        u_omega_star = u_omega_ref 
        
    else:
        plt.scatter(x_t, y_t, color = 'green')      
        
        # Optimal Contorl    
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
    
    # For Plotting and debugging
    x_list.append(x_t)
    y_list.append(y_t)
    theta_list.append(theta_t)
    psi_list.append(psi)
    u_s_star_list.append(u_s_star)
    u_omega_star_list.append(u_omega_star)

plt.show()

#%%






