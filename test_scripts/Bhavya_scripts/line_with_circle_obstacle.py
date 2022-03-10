#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:56:15 2022

@author: madhusudhan
Modified by: BGG
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt



t = np.linspace(0, 10, 1000)
x_values = t
y_values = t

x_goal = 9
y_goal = 10
phi_goal = math.pi/4

c_x = 5     #x-position of the center
c_y = 5     #y-position of the center
#r = 2       #radius 
a = 3 #Semi-Major axis
b = 2 #Semi-Minor axis

p = np.linspace(0, 6.3, 630)
x_values_ellipse = c_x + a*np.cos(p) 
y_values_ellipse = c_y + b*np.sin(p)


## Plotting
fig, ax = plt.subplots(2,2)
#plt.xlim(-2, 28)
#plt.ylim(-2, 10)
plt.gca().set_aspect('equal')

plt.plot(x_values, y_values, color = 'salmon')  # Plotting the required trajectory

plt.plot( x_values_ellipse, y_values_ellipse)   # Plotting the obstacle 
plt.fill( x_values_ellipse, y_values_ellipse , 'lightgreen')

plt.grid(color='lightgray',linestyle='--')
plt.title("Desired Path")
plt.xlabel("x(t)")
plt.ylabel("y(t)")

#%%

def point_wrt_ellipse(x, y, c_x, c_y, a, b):
    return pow(((x - c_x)/a), 2) + pow(((y - c_y)/b), 2) - 1
    

# Initialization of state variables
x_t = 0.5
y_t = 0
theta_t = math.atan(0)

# Parameters
delta_t = 0.01
gamma = 1

# Temporary variables for debugging
x_list = []
y_list = []
theta_list = []
psi_list = []
u_s_star_list = []
u_omega_star_list = []
p = 0
n = 0

#PID Tuning Parameters
kp_theta = 2
kd_theta = 5
old_e_theta = 0

#kp_s = 2
kd_s = 5
old_e_s = 0

alpha = 2

for t in np.arange(0, 10, delta_t):
  
    # Reference controls
    theta_d = math.atan2(y_goal - y_t, x_goal - x_t)
    e_s = math.sqrt((x_goal - x_t)**2 + (y_goal - y_t)**2)
    e_theta = theta_d - theta_t
    
    kp_s = 2*((1-math.exp(-alpha*(e_s**3)))/e_s)
    
    e_s_dot = e_s - old_e_s
    e_theta_dot = e_theta - old_e_theta
    
    u_s_ref = kp_s*e_s + kd_s*e_s_dot
    u_omega_ref = kp_theta*e_theta + kd_theta*e_theta_dot
    #u_s_ref = 2
    #u_omega_ref = 0
    
    old_e = e_theta
    old_e_s = e_s
    
    
    # Terms in QP
    A_1_1 = (2*(x_t - c_x)*math.cos(theta_t))/(a**2) + (2*(y_t - c_y)*math.sin(theta_t))/(b**2)
    point_dist_wrt_ellipse = point_wrt_ellipse(x_t, y_t, c_x, c_y, a, b)
    psi =  A_1_1*u_s_ref + gamma*point_dist_wrt_ellipse
   

    
    if psi < 0:
        plt.scatter(x_t, y_t, color = 'red')  
        
        # Optimal Control    
        u_s_star = u_s_ref - psi/(A_1_1)
        
        u_omega_star = np.pi/2
        
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

t_list = np.arange(0, 10, delta_t)

ax[0, 0].plot(t_list, u_s_star_list)
ax[0, 0].set_xlabel("Time step")
ax[0, 0].set_ylabel("Us* (Speed)")
#ax[0, 0].set_title("Us*")

ax[1, 0].plot(t_list, u_omega_star_list)
ax[1, 0].set_xlabel("Time step")
ax[1, 0].set_ylabel("Uw*")
#ax[1, 0].set_title("Uw*")

ax[0, 1].plot(t_list, theta_list)
ax[0, 1].set_xlabel("Time step")
ax[0, 1].set_ylabel("Theta (Heading direction)")
#ax[1, 0].set_title("Theta")

plt.show()

#Continuous Uref
#Incoorporate Bicycle model with acceleration
#Incoorporate Path tracking 
#Incoorporate generalized Uref for circum-navigation
#Side Obstacle
#Us dependent gamma (Shaping of CBF)?
#Area around obstacle where QP kicks in




