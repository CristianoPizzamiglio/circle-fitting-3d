# -*- coding: utf-8 -*-
"""
@author: Cristiano Pizzamiglio

"""

import matplotlib.pyplot as plt
import numpy as np

from skspatial.objects import Points

from circle_fitting_3d import Circle3D


def main():

    radius = 1.0
    center = np.array([-1.2, -1.5, 0.2])
    theta = 45 / 180 * np.pi
    phi = -30 / 180 * np.pi
    t = np.linspace(0, 1 * np.pi, 80)
    points = generate_points_by_angles(t, center, radius, theta, phi)

    # a = Circle3D([[1, 0, 0], [0, 1, 0], [-1, 0, 0]])
    # circle_3d = Circle3D([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    circle_3d = Circle3D(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    circle_3d.plot(ax)


def generate_points_by_angles(t, center, radius, theta, phi):
    """
    Generate test points given center, radius, azimuth (theta) and zenith (phi).

    References
    ----------
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

    """
    n = np.array(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
    )
    u = np.array([-np.sin(phi), np.cos(phi), 0])

    circle = (
        radius * np.cos(t)[:, np.newaxis] * u
        + radius * np.sin(t)[:, np.newaxis] * np.cross(n, u)
        + center
    )
    return Points(circle) + np.random.normal(size=circle.shape) * 0.1


if __name__ == "__main__":
    main()
