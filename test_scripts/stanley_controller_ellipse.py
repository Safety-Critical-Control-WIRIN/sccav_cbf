#!/bin/python3
"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

modified by: Neelaksh Singh
for: Safety Critical Control of Self Driving Cars

"""
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import imageio
import sys
import os

from matplotlib.patches import Ellipse
from cvxopt import matrix, solvers, spdiag

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise


k = 0.5  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
max_steer = np.radians(30.0)  # [rad] max steering angle

show_animation = True


class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt


def pid_control(target, current):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, last_target_idx):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param last_target_idx: (int)
    :return: (float, int)
    """
    current_target_idx, error_front_axle = calc_target_index(state, cx, cy)

    if last_target_idx >= current_target_idx:
        current_target_idx = last_target_idx

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[current_target_idx] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * error_front_axle, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, current_target_idx


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    target_idx = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

    return target_idx, error_front_axle

def CBF(s, u_des, cx, cy, a, b, gamma):
    m = 1 # No. of Constraints
    n = 2 # Dimension of x0 i.e. u
    u_des = matrix(u_des)
    def F(x=None, z=None):
        # if x is None: return m, matrix(0.0, (n, 1))
        if x is None: return m, u_des    
        # for 1 objective function and 1 constraint and 3 state vars
        f = matrix(0.0, (m+1, 1))
        Df = matrix(0.0, (m+1, n))
        f[0] = (x - u_des).T * (x - u_des)
        # temp vars are written as tn
        t1 = matrix([2*(s[0] - cx)/a, 2*(s[1] - cy)/b, 0], (1,3))
        t2 = matrix([ [np.cos(s[2]), np.sin(2), 0], [0, 0, 1] ])
        t3 = gamma * ( ((s[0] - cx)/a)**2 + ((s[1] - cy)/b)**2 - 1 )

        f[1] = -(t1 * t2 * x + t3)

        Df[0, :] = 2 * (x - u_des).T
        Df[1, :] = -1 * (t1 * t2)

        if z is None: return f, Df
        H = z[0] * 2 * matrix(np.eye(n))
        return f, Df, H
    return solvers.cp(F)['x']


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    target_speed = 30.0 / 3.6  # [m/s]

    max_simulation_time = 40.0

    # Initial state
    state = State(x=-0.0, y=5.0, yaw=np.radians(20.0), v=0.0)

    last_idx = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    target_idx, _ = calc_target_index(state, cx, cy)

    # Elliptical Obstacle on Track
    a = 20
    b = 10
    obs_idx = int(last_idx*0.75) # Obstacle on 75% of the trajectory
    o_cx = cx[obs_idx]
    o_cy = cy[obs_idx]

    USE_CBF = True
    ZERO_TOL = 1e-3
    # params for animation
    i = 0
    fnames = []
    while max_simulation_time >= time and last_idx > target_idx:
        # We will assume that the velocity control based CBF is modifying the
        # target velocity.
        v_ = target_speed
        ai = pid_control(target_speed, state.v)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)

        # Implementing the CBF
        if USE_CBF:
            u_des = np.array([v_, v_ * np.tan(state.yaw)/L])
            gamma = 3.0
            s = np.array([ state.x, state.y, state.yaw ])
            u = CBF(s, u_des, o_cx, o_cy, a, b, gamma)
            v_cbf = u[0]
            w_cbf = u[1]
            # di_cbf = np.arctan(w_cbf * L / v_cbf)
            di_cbf = di
            print("v: ", v_cbf, " delta: ", di_cbf)
            print("old v: ", v_, " old delta: ", di)
            print("time: ", time)
            ai_cbf = pid_control(v_cbf, state.v)
            state.update(ai_cbf, di_cbf)
        else:
            state.update(ai, di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        
        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(cx, cy, ".r", label="course")
            plt.plot(x, y, "-b", label="trajectory")
            plt.plot(cx[target_idx], cy[target_idx], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.xlabel("x[m]")
            plt.ylabel("y[m]")
            plt.legend(loc='lower left')
            if(abs(state.v) > ZERO_TOL):
                plt.title("Speed[m/s]:" + str(state.v)[:4])
            else:
                plt.title("Speed[m/s]: 0")
            
            if(abs(v_cbf) > ZERO_TOL):
                plt.text(0, 30, "CBF V[m/s]:" + str(v_cbf)[:4])
            else:
                plt.text(0, 30, "CBF V[m/s]: 0")
            
            if(abs(target_speed) > ZERO_TOL):
                plt.text(75, 30, "Planner V[m/s]:" + str(target_speed)[:4])
            else:
                plt.text(75, 30, "Planner V[m/s]: 0")
            
            if abs(v_cbf - target_speed) < ZERO_TOL:
                cbf_active = False
                plt.text(0, 25, "CBF Status: Dormant")
            else:
                cbf_active = True
                plt.text(0, 25, "CBF Status: Triggered")
            plt.text(0, 20, "Gamma: " + str(gamma)[:4])            

            ax = plt.gca()
            obs_ellipse = Ellipse(xy=(o_cx, o_cy), width=a, height=b, ec='b', fc='g', lw=2)
            ax.add_patch(obs_ellipse)

            # im = plt.imshow(animated=True)
            # ims.append([im])
            plt.pause(0.001)
            fname = os.getcwd() + "/temp/ts1_{0}.png".format(i)
            i = i + 1
            fnames.append(fname)
            plt.savefig(fname)
        

    # Test
    assert last_idx >= target_idx, "Cannot reach goal"

    # ani = animation.ArtistAnimation(plt.get_current_fig_manager, ims, interval=50, blit=True)
    # ani.save("stanley_ellipse_cbf.mp4")
    writer =  imageio.get_writer("stanley_ellipse_cbf.mp4", fps=20)
    for filename in fnames:
        image = imageio.imread(filename)
        writer.append_data(image)
    writer.close()
    
    for filename in set(fnames):
        os.remove(filename)

    if show_animation:  # pragma: no cover
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(x, y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(t, [iv * 3.6 for iv in v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()
