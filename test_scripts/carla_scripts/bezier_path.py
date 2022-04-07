"""
Path planning with Bezier curve.
author: Atsushi Sakai(@Atsushi_twi)
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import math

show_animation = True

class Bezier:

    def __init__(self, sx, sy, syaw, ex, ey, eyaw, offset, resolution = 100):
        # self.sx = sx
        # self.sy = sy
        # self.syaw = syaw
        # self.ex = ex
        # self.ey = ey
        # self.eyaw = eyaw
        # self.offset = offset
        self.resolution = resolution
        self.path, self.control_points = self.calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset)

    def calc_4points_bezier_path(self,sx, sy, syaw, ex, ey, eyaw, offset):
        """
        Compute control points and path given start and end position.
        :param sx: (float) x-coordinate of the starting point
        :param sy: (float) y-coordinate of the starting point
        :param syaw: (float) yaw angle at start
        :param ex: (float) x-coordinate of the ending point
        :param ey: (float) y-coordinate of the ending point
        :param eyaw: (float) yaw angle at the end
        :param offset: (float)
        :return: (numpy array, numpy array)
        """
        dist = np.hypot(sx - ex, sy - ey) / offset
        control_points = np.array(
            [[sx, sy],
             [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
             [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],
             [ex, ey]])

        path = self.calc_bezier_path(control_points)

        return path, control_points


    def calc_bezier_path(self, control_points):
        """
        Compute bezier path (trajectory) given control points.
        :param control_points: (numpy array)
        :return: (numpy array)
        """
        traj = []
        for t in np.linspace(0, 1, self.resolution):
            traj.append(self.bezier(t, control_points))

        return np.array(traj)


    def bernstein_poly(self, n, i, t):
        """
        Bernstein polynom.
        :param n: (int) polynom degree
        :param i: (int)
        :param t: (float)
        :return: (float)
        """
        return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)


    def bezier(self, t, control_points):
        """
        Return one point on the bezier curve.
        :param t: (float) number in [0, 1]
        :param control_points: (numpy array)
        :return: (numpy array) Coordinates of the point
        """
        n = len(control_points) - 1
        return np.sum([self.bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

    def bezier_derivatives_control_points(self, control_points, n_derivatives):
        """
        Compute control points of the successive derivatives of a given bezier curve.
        A derivative of a bezier curve is a bezier curve.
        See https://pomax.github.io/bezierinfo/#derivatives
        for detailed explanations
        :param control_points: (numpy array)
        :param n_derivatives: (int)
        e.g., n_derivatives=2 -> compute control points for first and second derivatives
        :return: ([numpy array])
        """
        w = {0: control_points}
        for i in range(n_derivatives):
            n = len(w[i])
            w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j])
                                 for j in range(n - 1)])
        return w


    def curvature(self, dx, dy, ddx, ddy):
        """
        Compute curvature at one point given first and second derivatives.
        :param dx: (float) First derivative along x axis
        :param dy: (float)
        :param ddx: (float) Second derivative along x axis
        :param ddy: (float)
        :return: (float)
        """
        return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

    def plot_arrow(self, x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
        """Plot arrow."""
        if not isinstance(x, float):
            for (ix, iy, iyaw) in zip(x, y, yaw):
                self.plot_arrow(ix, iy, iyaw)
        else:
            plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                      fc=fc, ec=ec, head_width=width, head_length=width)
            plt.plot(x, y)

    def heading_angle(self, t):
        derivatives_cp = self.bezier_derivatives_control_points(self.control_points, 2)
        dt = self.bezier(t, derivatives_cp[1])
        dt /= np.linalg.norm(dt, 2)

        dy = dt[1]
        dx = dt[0]

        angle = np.arctan(dy / dx)
        # angle = angle*180 / math.pi

        return angle

    def get_trajectory(self, velocity = 5):
        traj = []
        for t in np.linspace(0, 1, self.resolution):
            point = self.bezier(t,self.control_points)
            angle = self.heading_angle(t)
            velocity = velocity
            waypoint = (point[0], point[1], angle, velocity)
            traj.append(waypoint)
        return traj

def main():
    """Plot an example bezier curve."""
    start_x = -88.8  # [m]
    start_y = 108.5  # [m]
    start_yaw = np.radians(90.0)  # [rad]

    end_x = -58.6  # [m]
    end_y = 139.0  # [m]
    end_yaw = np.radians(0.0)  # [rad]
    offset = 3.0
    curve = Bezier(start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset, resolution=100)
    # path, control_points = calc_4points_bezier_path(
    #     start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)

    # Note: alternatively, instead of specifying start and end position
    # you can directly define n control points and compute the path:
    # control_points = np.array([[5., 1.], [-2.78, 1.], [-11.5, -4.5], [-6., -8.]])
    # path = calc_bezier_path(control_points, n_points=100)

    # Display the tangent, normal and radius of cruvature at a given point
    t = 0.5  # Number in [0, 1]
    x_target, y_target = curve.bezier(t, curve.control_points)
    derivatives_cp = curve.bezier_derivatives_control_points(curve.control_points, 2)
    point = curve.bezier(t, curve.control_points)
    dt = curve.bezier(t, derivatives_cp[1])
    ddt = curve.bezier(t, derivatives_cp[2])
    # Radius of curvature
    radius = 1 / curve.curvature(dt[0], dt[1], ddt[0], ddt[1])
    # Normalize derivative
    dt /= np.linalg.norm(dt, 2)
    tangent = np.array([point, point + dt])
    normal = np.array([point, point + [- dt[1], dt[0]]])
    curvature_center = point + np.array([- dt[1], dt[0]]) * radius
    circle = plt.Circle(tuple(curvature_center), radius,
                        color=(0, 0.8, 0.8), fill=False, linewidth=1)

    assert curve.path.T[0][0] == start_x, "path is invalid"
    assert curve.path.T[1][0] == start_y, "path is invalid"
    assert curve.path.T[0][-1] == end_x, "path is invalid"
    assert curve.path.T[1][-1] == end_y, "path is invalid"

    print(curve.get_trajectory())

    if show_animation:  # pragma: no cover
        fig, ax = plt.subplots()
        ax.plot(curve.path.T[0], curve.path.T[1], label="Bezier Path")
        ax.plot(curve.control_points.T[0], curve.control_points.T[1],
                '--o', label="Control Points")
        ax.plot(x_target, y_target)
        ax.plot(tangent[:, 0], tangent[:, 1], label="Tangent")
        # ax.plot(normal[:, 0], normal[:, 1], label="Normal")
        # ax.add_artist(circle)
        curve.plot_arrow(start_x, start_y, start_yaw)
        curve.plot_arrow(end_x, end_y, end_yaw)
        ax.invert_yaxis()
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        plt.show()


# def main2():
#     """Show the effect of the offset."""
#     start_x = 10.0  # [m]
#     start_y = 1.0  # [m]
#     start_yaw = np.radians(180.0)  # [rad]
#
#     end_x = -0.0  # [m]
#     end_y = -3.0  # [m]
#     end_yaw = np.radians(-45.0)  # [rad]
#
#     for offset in np.arange(1.0, 5.0, 1.0):
#         path, control_points = calc_4points_bezier_path(
#             start_x, start_y, start_yaw, end_x, end_y, end_yaw, offset)
#         assert path.T[0][0] == start_x, "path is invalid"
#         assert path.T[1][0] == start_y, "path is invalid"
#         assert path.T[0][-1] == end_x, "path is invalid"
#         assert path.T[1][-1] == end_y, "path is invalid"
#
#         if show_animation:  # pragma: no cover
#             plt.plot(path.T[0], path.T[1], label="Offset=" + str(offset))
#
#     if show_animation:  # pragma: no cover
#         plot_arrow(start_x, start_y, start_yaw)
#         plot_arrow(end_x, end_y, end_yaw)
#         plt.legend()
#         plt.axis("equal")
#         plt.grid(True)
#         plt.show()


if __name__ == '__main__':
    main()
    #  main2()