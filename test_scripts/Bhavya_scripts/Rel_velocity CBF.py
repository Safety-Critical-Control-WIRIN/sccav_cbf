# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 14:19:03 2022

@author: CHESTER
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt



t = np.linspace(0, 10, 1000)
x_values = t
y_values = t

#Vehicle axle length
lr = 0.5
lf = 0.5
l = lr + lf 

x_goal = 9
y_goal = 10

c_x = 4     #x-position of the center
c_y = 6     #y-position of the center
c_x_dot = 0
c_y_dot = 0
vo = math.sqrt(c_x_dot**2 + c_y_dot**2) #Velocity of the obstacle
a = 3 #Semi-Major axis
b = 3 #Semi-Minor axis

p = np.linspace(0, 6.3, 630)
x_values_ellipse = c_x + a*np.cos(p) 
y_values_ellipse = c_y + b*np.sin(p)


## Plotting
fig, ax = plt.subplots(3,2)
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

    

# Initialization of state variables
x_t = 0
y_t = 0
theta_t = math.atan(1)
v_t = 1
delta = 0

# Parameters
delta_t = 0.01
gamma = 0.5
k = 0.01

# Temporary variables for debugging

x_list = []
y_list = []
theta_list = []
v_list = []
steering_delta = []
psi_list = []
u_a_star_list = []
u_beta_star_list = []

for t in np.arange(0, 10, delta_t):
    
    # Reference controls
    
    u_a_ref = 0
    u_beta_ref = 0
    x_d = (x_t - c_x)/(a**2)
    y_d = (y_t - c_y)/(b**2)
    # Terms in QP
    p1 = -k*u_a_ref - 2*u_beta_ref*(x_d*v_t*math.sin(theta_t) - y_d*v_t*math.cos(theta_t))
    p2 = 2*(x_d*v_t*math.cos(theta_t) + y_d*v_t*math.sin(theta_t))
    p3 = -2*(x_d*c_x_dot + y_d*c_y_dot)
    p4 = gamma*(x_d*(x_t - c_x) + y_d*(y_t - c_y) - 1 - k*(v_t - vo))
    psi =  p1 + p2 + p3 + p4
    
    AAt = k**2 + (-2*x_d*v_t*math.sin(theta_t) + 2*y_d*v_t*math.cos(theta_t))**2
   

    
    if psi < 0:
        plt.scatter(x_t, y_t, color = 'red', s = 0.5)  
        
        # Optimal Control    
        u_a_star = u_a_ref + (k/AAt)*psi
        u_beta_star = u_beta_ref + (2*(x_d*v_t*math.sin(theta_t) - y_d*v_t*math.cos(theta_t)))*psi
        
    else:
        plt.scatter(x_t, y_t, color = 'green', s = 0.5)      
        
        # Optimal Control    
        u_a_star = u_a_ref
        u_beta_star = u_beta_ref
    
    # Vehicle Kinematics
    x_dot = v_t*math.cos(theta_t) - v_t*math.sin(theta_t)*u_beta_star
    y_dot = v_t*math.sin(theta_t) + v_t*math.cos(theta_t)*u_beta_star
    theta_dot = (v_t/(lr))*u_beta_star
    v_dot = u_a_star
    
    # Updating state for next time step
    x_t += x_dot*delta_t
    y_t += y_dot*delta_t
    theta_t += theta_dot*delta_t
    v_t += v_dot*delta_t
    delta = math.atan2((lr + lf), lr**math.tan(u_beta_star))
    
    # For Plotting and debugging
    x_list.append(x_t)
    y_list.append(y_t)
    theta_list.append(theta_t)
    v_list.append(v_t)
    steering_delta.append(delta)
    psi_list.append(psi)
    u_a_star_list.append(u_a_star)
    u_beta_star_list.append(u_beta_star)

t_list = np.arange(0, 10, delta_t)

ax[0, 0].plot(t_list, u_a_star_list)
ax[0, 0].set_xlabel("Time step")
ax[0, 0].set_ylabel("Acceleration*")
#ax[0, 0].set_title("Us*")

ax[1, 0].plot(t_list, u_beta_star_list)
ax[1, 0].set_xlabel("Time step")
ax[1, 0].set_ylabel("Beta*")
#ax[1, 0].set_title("Uw*")

ax[0, 1].plot(t_list, theta_list)
ax[0, 1].set_xlabel("Time step")
ax[0, 1].set_ylabel("Theta (Heading direction)")
#ax[1, 0].set_title("Theta")

ax[1, 1].plot(t_list, steering_delta)
ax[1, 1].set_xlabel("Time step")
ax[1, 1].set_ylabel("Steering Angle")
#ax[1, 0].set_title("Theta")

ax[2, 0].plot(t_list, v_list)
ax[2, 0].set_xlabel("Time step")
ax[2, 0].set_ylabel("Velocity")
#ax[1, 0].set_title("Theta")

plt.show()

        

#Continuous Uref
#Acceleration Requires new CBF
#Incoorporate Path tracking and animation
#Incoorporate generalized Uref for circum-navigation for decision making
#Side Obstacle
#Us dependent gamma (Shaping of CBF)?
#Area around obstacle where QP kicks in
#CBF based on distance with the closest point on the obstacle
#Different types of CBF



