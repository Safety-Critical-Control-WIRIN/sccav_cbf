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
import errno

# from matplotlib.patches import Ellipse, Circle
import matplotlib.patches as patches
from cvxopt import matrix, solvers, spdiag, sqrt
from euclid import *

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/PathPlanning/CubicSpline/")

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../")

try:
    import cubic_spline_planner
    from cbf.obstacles import Ellipse2D
    from cbf.cbf import KBM_VC_CBF2D, DBM_CBF_2DS
except:
    raise


k = 0.5  # control gain
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time difference
L = 2.9  # [m] Wheel base of vehicle
lr = L/2
lf = L - lr
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
    
    def update_by_vel(self, v_, delta):
        """
        Update the state of the vehicle. But
        instead of using an acceleration based
        control, use direct velocity control.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v = v_
    
    def update_com(self, acceleration, delta):
        delta = np.clip(delta, -max_steer, max_steer)
        beta = np.arctan2(lr * np.tan(delta), lf + lr)
        
        self.x += (self.v * np.cos(self.yaw) - self.v * np.sin(self.yaw) * beta) * dt
        self.y += (self.v * np.sin(self.yaw) + self.v * np.cos(self.yaw) * beta) * dt
        self.yaw += (self.v * beta/lr) * dt
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
        t1 = matrix([2*(s[0] - cx)/(a**2), 2*(s[1] - cy)/(b**2), 0], (1,3))
        t2 = matrix([ [np.cos(s[2]), np.sin(s[2]), 0], [0, 0, 1] ])
        t3 = gamma * ( ((s[0] - cx)/a)**2 + ((s[1] - cy)/b)**2 - 1 )

        f[1] = -(t1 * t2 * x + t3)

        Df[0, :] = 2 * (x - u_des).T
        Df[1, :] = -1 * (t1 * t2)

        if z is None: return f, Df
        H = z[0] * 2 * matrix(np.eye(n))
        return f, Df, H
    return solvers.cp(F)['x']

def D_CBF(s, u_des, cx, cy, Ds, gamma):
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
        # CBFs
        # Distance based CBF and its derivatives
        h1 = sqrt((s[0] - cx)**2 + (s[1] - cy)**2) - Ds
        h1_dx = 2 * (s[0] - cx)/(h1 + Ds)
        h1_dy = 2 * (s[1] - cy)/(h1 + Ds)
        # h1_dxx = 2 * ( (h1 + Ds) - (s[0] - cx) * h1_dx )/( (h1 + Ds)**2 )
        # h1_dxy = -2 * (s[0] - cx) * (s[1] - cy)/( (h1 + Ds)**3 )
        # h1_dyx = h1_dxy
        # h1_dyy = 2 * ( (h1 + Ds) - (s[1] - cy) * h1_dy )/( (h1 + Ds)**2 )

        # temp vars are written as tn or tnm or tnf where f is the func it
        # corresponds to
        t11 = matrix([h1_dx, h1_dy, 0], (1,3))
        t12 = matrix([ [np.cos(s[2]), np.sin(s[2]), 0], [0, 0, 1] ])
        t13 = -gamma * h1

        f[1] = -(t11 * t12 * x + t13)

        Df[0, :] = 2 * (x - u_des).T
        Df[1, :] = -1 * (t11 * t12)

        if z is None: return f, Df
        H = z[0] * 2 * matrix(np.eye(n))
        return f, Df, H
    return solvers.cp(F)['x']

def CBF_A(s, u_des, cx, cy, a, b, alpha):
    m = 1 # No. of Constraints
    n = 2 # Dimension of x0 i.e. u
    u_des = matrix(u_des)

    def F(x=None, z=None):
        if x is None: return m, matrix(0.0, (n, 1))
        # if x is None: return m, u_des    
        # for 1 objective function and 1 constraint and 3 state vars
        f = matrix(0.0, (m+1, 1))
        Df = matrix(0.0, (m+1, n))
        f[0] = (x - u_des).T * (x - u_des)
        # CBFs
        h1 = ((s[0] - cx)/a)**2 + ((s[1] - cy)/b)**2 - 1
        h1_dx = 2*(s[0] - cx)/(a**2)
        h1_dy = 2*(s[1] - cy)/(b**2)
        # temp vars are written as tn
        Dh1 = matrix([h1_dx, h1_dy, 0, 0], (1, 4))
        f_c = matrix([ s[3] * np.cos(s[2]), s[3] * np.sin(s[2]), 0, 0], (4, 1))
        g_c = matrix([ [0, 0, 0, 1], [-s[3] * np.sin(s[2]), s[3] * np.cos(s[2]), s[3]/lr, 0] ])

        f[1] = -(Dh1 * (f_c + g_c * x) + alpha * h1)

        Df[0, :] = 2 * (x - u_des).T
        Df[1, :] = -1 * (Dh1 * g_c)

        if z is None: return f, Df
        H = z[0] * 2 * matrix(np.eye(n))
        return f, Df, H

    # Enforcing linear constraints on control i/p
    # a_max = 2.23 # m/s^2
    # a_min = 0

    # G = matrix([ [-1, 1],
    #              [ 0, 0] ])
    # h = matrix([ -a_min, a_max ])

    # dims = {'l': 0, 'q': [], 's':  [3]}

    # return solvers.cp(F, G=G, h=h, dims=dims)['x']
    return solvers.cp(F)['x']

def saturation(x, x_min, x_max):
    if x >= x_max:
        return x_max
    elif x <= x_min:
        return x_min
    else:
        return x


# The accelerating obstacle case.


def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    #  target course
    ax = [0.0, 100.0, 100.0, 50.0, 60.0]
    ay = [0.0, 0.0, -30.0, -20.0, 0.0]

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=0.1)

    target_speed = 30.0 / 3.6  # [m/s]

    max_simulation_time = 30

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
    obs_idx = int(last_idx*0.50) # Obstacle on 75% of the trajectory
    o_cx = cx[obs_idx]
    o_cy = cy[obs_idx]

    # FLAGS and IMP. CONSTANTS
    USE_CBF = True
    ZERO_TOL = 1e-3
    CBF_TYPE = 2 # 0: Ellipse, 1: Distance, 2: Ellipse - Acceleration Controlled
                 # 3: Ellipse - API
    a_max = 2.29 # m/s^2
    a_min = -2.29
    # params for animation
    i = 0
    fnames = []
    delta_stanley = np.zeros(int(max_simulation_time/dt) + 1)
    delta_cbf = np.zeros_like(delta_stanley)
    while max_simulation_time >= time and last_idx > target_idx:
        # We will assume that the velocity control based CBF is modifying the
        # target velocity.
        v_ = target_speed
        ai = pid_control(target_speed, state.v)
        di, target_idx = stanley_control(state, cx, cy, cyaw, target_idx)
        delta_stanley[i] = di

        # Implementing the CBF
        if USE_CBF:
            gamma = 1
            
            Dbuffer = 1
            Ds = max(a, b)/2 + Dbuffer
            if CBF_TYPE == 0:
                u_des = np.array([v_, v_ * np.tan(di)/L])
                s = np.array([ state.x, state.y, state.yaw ])
                u = CBF(s, u_des, o_cx, o_cy, a, b, gamma)
                v_cbf = u[0]
                w_cbf = u[1]
                di_cbf = np.arctan(w_cbf * L / v_cbf)
                delta_cbf[i] = di_cbf
                state.update_by_vel(v_cbf, di_cbf)
                print("v: ", v_cbf, " delta: ", di_cbf)
                print("old v: ", v_, " old delta: ", di)

            if CBF_TYPE == 1:
                u_des = np.array([v_, v_ * np.tan(di)/L])
                s = np.array([ state.x, state.y, state.yaw ])
                u = D_CBF(s, u_des, o_cx, o_cy, Ds, gamma)
                v_cbf = u[0]
                w_cbf = u[1]
                di_cbf = np.arctan(w_cbf * L / v_cbf)
                delta_cbf[i] = di_cbf
                state.update_by_vel(v_cbf, di_cbf)
                print("v: ", v_cbf, " delta: ", di_cbf)
                print("old v: ", v_, " old delta: ", di)

            if CBF_TYPE == 2:
                a_ = pid_control(target_speed, state.v)

                ## Without Class ##
                # beta_ = np.arctan2(lr * np.tan(di), lf + lr)
                # u_des = np.array([a_, beta_])
                # s = np.array([ state.x, state.y, state.yaw, state.v ])
                # u = CBF_A(s, u_des, o_cx, o_cy, a, b, gamma)
                # # a_cbf = saturation(u[0], a_min, a_max)
                # a_cbf = u[0]
                # beta_cbf = u[1]
                # di_cbf = np.arctan2((lf + lr)*np.tan(beta_cbf), lr)

                ## With Class ##
                cbf_controller = DBM_CBF_2DS(alpha=gamma)
                cbf_controller.set_model_params(lr=lr, lf=lf)
                cbf_controller.obstacle_list2d.update({
                    0: Ellipse2D(a=a, b=b, center=Point2(o_cx, o_cy))
                })
                p=Vector2(state.x, state.y)
                cbf_controller.update_state(p, state.v, state.yaw)
                cbf_controller.set_qp_cost_weight(np.diag([0.5, 20000.0]))
                u = cbf_controller.solve_cbf(np.array([a_, di]))
                a_cbf = u[0]
                di_cbf = u[1]
                delta_cbf[i] = di_cbf

                state.update_com(a_cbf, di_cbf)
                print(" a: ", a_cbf, " delta: ", di_cbf)
                print(" v: ", v_, " old a: ", a_, " old delta: ", di)
            
            if CBF_TYPE == 3:
                cbf_controller = KBM_VC_CBF2D(gamma=gamma)
                cbf_controller.set_model_params(L=L)
                cbf_controller.obstacle_list2d.update({
                    0: Ellipse2D(a=a, b=b, center=Point2(o_cx, o_cy), buffer=20.0)
                })
                p=Vector2(state.x, state.y)
                cbf_controller.update_state(p, theta=state.yaw)
                solver_op, u = cbf_controller.solve_cbf(np.array([v_, di]))
                v_cbf = u[0]
                di_cbf = u[1]
                state.update_by_vel(v_cbf, di_cbf)
                print("v: ", v_cbf, " delta: ", di_cbf)
                print("old v: ", v_, " old delta: ", di)
                print("f: ", cbf_controller.obstacle_list2d.f(p))
            
            print("time: ", time)
            
        else:
            state.update(ai, di)

        time += dt

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # Creating the temp directory to store temporary data
        try:
            if not os.path.exists('temp'):
                os.makedirs('temp')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
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
                        
            if CBF_TYPE in [0, 1, 3]:
                if(abs(v_cbf) > ZERO_TOL):
                    plt.text(0, 30, "CBF V[m/s]:" + str(v_cbf)[:4])
                else:
                    plt.text(0, 30, "CBF V[m/s]: 0")
                
                if(abs(target_speed) > ZERO_TOL):
                    plt.text(75, 30, "Planner V[m/s]:" + str(target_speed)[:4])
                else:
                    plt.text(75, 30, "Planner V[m/s]: 0")
                
                if USE_CBF and (CBF_TYPE == 0):
                    plt.text(75, 25, "CBF Type: Ellipse")
                if USE_CBF and (CBF_TYPE == 1):
                    plt.text(75, 25, "CBF Type: Distance")
                if USE_CBF and (CBF_TYPE == 3):
                    plt.text(75, 25, "CBF Type: Ellipse (API)")

                if abs(v_cbf - target_speed) < ZERO_TOL:
                    cbf_active = False
                    plt.text(0, 25, "CBF Status: Dormant")
                else:
                    cbf_active = True
                    plt.text(0, 25, "CBF Status: Triggered")

            if CBF_TYPE == 2:
                if(abs(a_cbf) > ZERO_TOL):
                    plt.text(0, 30, "CBF a[m/s]:" + str(a_cbf)[:4])
                else:
                    plt.text(0, 30, "CBF a[m/s]: 0")
                
                if(abs(a_) > ZERO_TOL):
                    plt.text(75, 30, "PID a[m/s]:" + str(a_)[:4])
                else:
                    plt.text(75, 30, "PID a[m/s]: 0")

                if USE_CBF:
                    plt.text(75, 25, "CBF Type: Ellipse|A")

                trig_flag_a = abs(a_cbf - a_) >= ZERO_TOL
                trig_flag_d = abs(di_cbf - di) >= ZERO_TOL
                if trig_flag_a and trig_flag_d:
                    plt.text(0, 25, "CBF Status: a, delta Triggered")
                elif trig_flag_a: 
                    plt.text(0, 25, "CBF Status: a Triggered")
                elif trig_flag_d:
                    plt.text(0, 25, "CBF Status: delta Triggered")
                else:
                    plt.text(0, 25, "CBF Status: Dormant")

            plt.text(0, 20, "Gamma: " + str(gamma)[:4])            

            ax = plt.gca()
            obs_ellipse = patches.Ellipse(xy=(o_cx, o_cy), width=a, height=b, ec='b', fc=(0,1,0,0.5), lw=2, ls='-.')
            obs_dist_circle = patches.Circle(xy=(o_cx, o_cy), radius=Ds, ls='--', lw=2, ec='k', fc=(0,1,0,0))
            ax.add_patch(obs_ellipse)
            ax.add_patch(obs_dist_circle)
            # im = plt.imshow(animated=True)
            # ims.append([im])
            plt.pause(0.001)
            fname = os.getcwd() + "\\temp\\ts1_{0}.png".format(i)
            i = i + 1
            fnames.append(fname)
            plt.savefig(fname)
        
    delta_diff = delta_cbf - delta_stanley
    # zero padding
    if delta_diff.size < len(t):
        delta_diff = np.append(delta_diff, np.zeros(len(t) - delta_diff.shape[0]))

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
        plt.figure(2)
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

        if CBF_TYPE in [0, 1, 3]:
            plt.figure(4)
            plt.plot(t, delta_diff, "-g")
            plt.xlabel("Time[s]")
            plt.ylabel("CBF Modification in delta[rad]")
            plt.grid(True)
            plt.show()

if __name__ == '__main__':
    main()
