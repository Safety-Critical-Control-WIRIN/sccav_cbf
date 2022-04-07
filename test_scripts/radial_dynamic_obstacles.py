#!/bin/python3
"""

Generates uniformly distributed obstacles over the periphery of a circle
with fixed radius and attached to the ego vehicle for testing of the TV-CBFs.
The vehicles are treated as obstacles through the cbf library and the CBFs
are solved through the cvxopt.qp solver, instead of the cp solver used by 
other programs.

author: Neelaksh Singh [https://www.github.com/TheGodOfWar007]

"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import stanley_controller_ellipse as sce

from cvxopt import solvers, matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    from cbf.obstacles import Ellipse2D
except:
    raise

# Global Variables used throughout the code.
OBSTACLE_COUNT = 1
OBSTACLE_SPAWN_INTERVAL = 10.0

def main():
    
    pass

if __name__ == '__main__':
    main()