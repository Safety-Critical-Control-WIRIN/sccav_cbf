# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 02:10:25 2022

@author: My PC
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
x_values = t
y_values = t

c_x = 7     #x-position of the center
c_y = 7     #y-position of the center
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
plt.title("Acceleration Control")
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
gamma = 0.13
time_start = 0
time_end = 40

# Temporary variables for debugging
x_list = []
y_list = []
theta_list = []
v_list = []

psi_list = []
u_a_star_list = []
u_omega_star_list = []
p = 0
n = 0

v_0 = 0
v_t = v_0
v_t_minus_1 = v_0

for t in np.arange(time_start, time_end, delta_t):
  
    # Reference controls
    u_a_ref = 0.01
    u_omega_ref = 0
    
    # Terms in QP
    A_1_1 = (x_t - c_x)*math.cos(theta_t) + (y_t - c_y)*math.sin(theta_t)
    point_dist_wrt_circle = point_wrt_circle(x_t, y_t, c_x, c_y, r)
    psi =  2*A_1_1*delta_t*u_a_ref + 2*A_1_1*v_t_minus_1 + gamma*point_dist_wrt_circle
   
    
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
        u_a_star = u_a_ref - psi/(2*A_1_1*delta_t)
        u_omega_star = u_omega_ref 
        
    else:
        plt.scatter(x_t, y_t, color = 'green')      
        
        # Optimal Contorl    
        u_a_star = u_a_ref
        u_omega_star = u_omega_ref 
    
    # Vehicle Dynamics
    x_dot = v_t * math.cos(theta_t)
    y_dot = v_t * math.sin(theta_t)
    theta_dot = u_omega_star
    v_dot = u_a_star
    
    # Updating state for next time step
    x_t = x_t + x_dot*delta_t
    y_t = y_t + y_dot*delta_t
    theta_t = theta_t + theta_dot*delta_t
    v_t_minus_1 = v_t
    v_t = v_t + v_dot*delta_t
    
    # For Plotting and debugging
    x_list.append(x_t)
    y_list.append(y_t)
    theta_list.append(theta_t)
    v_list.append(v_t)
    psi_list.append(psi)
    u_a_star_list.append(u_a_star)
    u_omega_star_list.append(u_omega_star)

plt.show()

#%%

# Plotting Velocity
plt.grid(color='lightgray',linestyle='--')
plt.title("Velocity Profile")
plt.xlabel("time")
plt.ylabel("Velocity")
plt.plot(v_list)

# Plotting Acceleration
plt.plot(u_a_star_list)