#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:48:03 2022

@author: madhusudhan
"""

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
x_values = pow(t,2)
y_values = t

## Plotting
fig, ax = plt.subplots()
ax.plot(x_values, y_values)
#x_values = pow(t,2)
#y_values = t
ax.plot(x_values, y_values - 2)
ax.plot(x_values, y_values + 2)
plt.fill(np.append(x_values, x_values[::-1]), np.append(y_values + 2, y_values[::-1] - 2), 'lightblue')

plt.grid(color='lightgray',linestyle='--')
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.show()