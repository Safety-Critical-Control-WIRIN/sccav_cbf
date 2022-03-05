#!/bin/python3
"""

Implementation of the Control Barrier Functions using a numerical
solver. Uses the obstacle list classes defined in the obstacles
module for the corresponding CBF representations.

author: Neelaksh Singh

"""

import sys
import os
import errno

import numpy as np
from cvxopt import matrix, solvers

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "../")

try:
    from obstacles import Ellipse2D, ObstacleList2D
except:
    raise

