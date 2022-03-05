#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 17:28:42 2022

@author: madhusudhan
"""
#%%
import math
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 5, 1000)
x_values = pow(t,2)
y_values = t

c_x = 9     #x-position of the center
c_y = 3   #y-position of the center
a = 3    #radius on the x-axis
b = 2   #radius on the y-axis

t = np.linspace(0, 10, 1000)
x_values_ellipse = c_x + a*np.cos(t) 
y_values_ellipse = c_y + b*np.sin(t)

## Plotting
fig, ax = plt.subplots()
plt.xlim(-2, 28)
plt.ylim(-2, 10)
plt.gca().set_aspect('equal')

plt.plot(x_values, y_values, color = 'salmon')
plt.plot( x_values_ellipse, y_values_ellipse)


plt.fill( x_values_ellipse, y_values_ellipse , 'lightgreen')
plt.grid(color='lightgray',linestyle='--')

plt.xlabel("x(t)")
plt.ylabel("y(t)")


#%%

def point_wrt_ellipse(x, y, c_x, c_y, a, b):
    return pow((x - c_x)/a, 2) + pow((y - c_y)/b, 2) - 1
    
q = math.radians(0.5)
theta = 0
x_t = 0
y_t = 0
theta_t = math.pi/2
delta_t = 0.5
gamma = 0.05

# Temporary variables
x_list = []
y_list = []
theta_list = []
p = 0
n = 0
for t in np.arange(1, 10, delta_t):
    
    # Reference controls
    u_s_ref = math.sqrt(4*math.pow(t,2) + 1)
    u_omega_ref = -2/(4*math.pow(t,2) + 1)
    
    # Theta
    #theta_t = theta_t + u_omega_ref * delta_t
    theta_t = math.atan(1/(2*t))
    #print(theta_t)
    #continue
    theta_list.append(theta_t)
    
    #print(theta_t)
    A_1_1 = ((x_t - c_x)/2)*math.cos(theta_t) + ((y_t - c_y)/2)*math.sin(theta_t)
    psi =  A_1_1*2*u_s_ref + gamma*point_wrt_ellipse(x_t, y_t, c_x, c_y, a, b)
    
    if psi > 0:
        p += 1
        #u_s = u_s_ref - (psi/(2*A_1_1))
        plt.scatter(x_t, y_t, color = 'red')
        u_s = u_s_ref
        
    else:
        n += 1
        plt.scatter(x_t, y_t, color = 'green')
        #print(A_1_1)
        u_s = u_s_ref
        
    
    x_dot = u_s * math.cos(theta_t)
    y_dot = u_s * math.sin(theta_t)
    
    x_t = x_t + x_dot*delta_t
    y_t = y_t + y_dot*delta_t
        
    x_list.append(x_t)
    y_list.append(y_t)
    


#plt.show()
plt.plot(theta_list)




