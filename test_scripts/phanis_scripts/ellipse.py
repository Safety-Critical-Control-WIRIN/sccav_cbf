#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:21:22 2022

@author: madhusudhan
"""

import numpy as np
import matplotlib.pyplot as plt
from math import pi

c_x = 3     #x-position of the center
c_y = 4    #y-position of the center
a=2.     #radius on the x-axis
b=1.5    #radius on the y-axis

t = np.linspace(0, 2*pi, 100)
x_values = c_x + a*np.cos(t) 
y_values = c_y + b*np.sin(t)

## Plotting
plt.plot( x_values, y_values)
plt.fill( x_values, y_values , 'lightgreen')
plt.grid(color='lightgray',linestyle='--')
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.show()