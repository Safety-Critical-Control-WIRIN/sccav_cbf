#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:26:23 2022

@author: madhusudhan
"""
import matplotlib.pyplot as plt

point1 = [1, 2]
point2 = [3, 2]
point3 = [3, 3]
point4 = [1, 3]

x_values = [point1[0], point2[0], point3[0], point4[0], point1[0]]
y_values = [point1[1], point2[1], point3[1], point4[1], point1[1]]

## Plotting
plt.plot(x_values, y_values)

#plt.autoscale(False)
plt.xlim((0, 4))
plt.ylim((1, 4))
plt.fill(x_values, y_values, 'lightcoral')

plt.grid(color='lightgray',linestyle='--')
plt.xlabel("x(t)")
plt.ylabel("y(t)")
plt.show()