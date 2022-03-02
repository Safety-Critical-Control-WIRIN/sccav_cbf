#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:10:46 2022

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

plt.grid(color='lightgray',linestyle='--')
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.show()