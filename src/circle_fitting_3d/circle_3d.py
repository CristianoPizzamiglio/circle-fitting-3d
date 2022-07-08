# -*- coding: utf-8 -*-
"""
@author: Cristiano Pizzamiglio

"""
from __future__ import annotations

from functools import cached_property
from typing import Sequence
from typing import Union

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from skspatial.objects import Circle
from skspatial.objects import Plane
from skspatial.objects import Point
from skspatial.objects import Points
from skspatial.objects import Vector


class Circle3D:
    """
    Best fit circle given three-dimensional points.

    Parameters
    ----------
    points : Union[np.ndarray, Sequence]
        Three-dimensional points.

    Attributes
    ----------
    points : Points
        Three-dimensional points.
    center : Point
        Circle center.
    radius : float
        Circle radius.
    normal : Vector
        Fitting plane normal.

    Raises
    ------
    ValueError
        If points are not three-dimensional.
        If there are fewer than three points.
        If points are collinear.

    Notes
    -----
    Duplicate points are removed.

    References
    ----------
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

    Example
    -------
    >>> from circle_fitting_3d import Circle3D

    >>> circle_3d = Circle3D([[3, 0, 0], [0, 3, 0], [0, 0, 3]])

    >>> circle_3d.center
    Point([1., 1., 1.])

    >>> circle_3d.radius
    2.449489742783178

    """

    def __init__(self, points: Union[np.ndarray, Sequence]) -> None:

        self.points = Points(points).unique()

    @property
    def points(self) -> Points:
        """
        Three-dimensional points.

        Returns
        -------
        Points

        """
        return self._points

    @points.setter
    def points(self, points_: Points) -> None:

        if points_.dimension != 3:
            raise ValueError("The points must be 3D.")

        if points_.shape[0] < 3:
            raise ValueError("There must be at least 3 points.")

        if points_.are_collinear():
            raise ValueError("The points must not be collinear.")

        self._points = points_

    @property
    def center(self) -> Point:
        """
        Center

        Returns
        -------
        Point

        """
        center = np.append(self._best_fit_circle_2d.point, 0.0)

        return Point(
            np.matmul(self._transformation, center) + self._projected_points.centroid()
        )

    @property
    def radius(self) -> float:
        """
        Radius

        Returns
        -------
        float

        """
        return self._best_fit_circle_2d.radius

    @property
    def normal(self) -> Vector:
        """
        Plane normal

        Returns
        -------
        Vector

        """
        return self._plane.normal.unit()

    def equation(self, t: Union[np.ndarray, Sequence]) -> np.ndarray:
        """
        Parametric equation.

        Parameters
        ----------
        t : Union[np.ndarray, Sequence]
            .. math:: 0 \le t \le 2\pi

        Returns
        -------
        np.ndarray
            Coordinates of points lying on the circle.

        """
        if isinstance(t, Sequence):
            t = np.array(t)

        u = Vector.from_points(
            self._plane.to_points()[0], self._plane.to_points()[1]
        ).unit()
        t = t.reshape(-1, 1)

        return (
            self.radius * np.cos(t) * u.reshape(1, -1)
            + self.radius * np.sin(t) * self.normal.cross(u)
            + self.center
        )

    @property
    def _plane(self) -> Plane:
        """
        Fitting plane, i.e., plane where the circle lies.

        Returns
        -------
        Plane

        """
        return Plane.best_fit(self.points)

    @property
    def _projected_points(self) -> Points:
        """
        Points projected onto the fitting plane.

        Returns
        -------
        Points

        """
        return Points([self._plane.project_point(point) for point in self.points])

    @property
    def _centered_projected_points(self) -> Points:
        """
        Mean-centered projected_points.

        Returns
        -------
        Points

        """
        return self._projected_points.mean_center()

    @cached_property
    def _transformation(self) -> np.ndarray:
        """
        Transformation matrix to map 3D co-planar points to 2D coordinates.

        Returns
        -------
        np.ndarray

        References
        ----------
        https://stackoverflow.com/questions/49769459/convert-points-on-a-3d-plane-to-2d-coordinates

        """
        u_basis = Vector.from_points(
            self._centered_projected_points[0], self._centered_projected_points[1]
        ).unit()
        v_basis = u_basis.cross(self.normal)

        return np.column_stack((u_basis, v_basis, self.normal))

    @cached_property
    def _best_fit_circle_2d(self) -> Circle:
        """
        2D best fit circle

        Returns
        -------
        Circle

        """
        points_2d = self._centered_projected_points.dot(self._transformation)
        return Circle.best_fit(points_2d[:, :2])

    def plot(
        self, ax: Axes3D, plot_points: bool = True, plot_plane: bool = False
    ) -> None:
        """
        Plot the circle in space.

        Parameters
        ----------
        ax : Axes3D
            Instance of :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`.
        plot_points : bool, optional
            The default is True.
        plot_plane : bool, optional
            The default is False.

        """
        t = np.linspace(0.0, 2 * np.pi, 1000)
        points = self.equation(t)
        ax.plot(points[:, 0], points[:, 1], points[:, 2])

        if plot_points:
            self.points.plot_3d(ax, c="r")

        if plot_plane:
            x_max = max(points[:, 0])
            y_max = max(points[:, 1])
            self._plane.plot_3d(
                ax, alpha=0.2, lims_x=(-x_max, x_max), lims_y=(-y_max, y_max)
            )

        ax.scatter(0.0, 0.0, 0.0, s=20)
        ax.quiver(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, length=0.5)
        ax.quiver(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, length=0.5)
        ax.quiver(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, length=0.5)
        text_position = 0.52
        ax.text(text_position, 0.0, 0.0, "x")
        ax.text(0.0, text_position, 0.0, "y")
        ax.text(0.0, 0.0, text_position, "z")

        ax.set_box_aspect((1, 1, 1))
        scaling = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
        minbound = min(scaling[:, 0])
        maxbound = max(scaling[:, 1])
        ax.auto_scale_xyz(*[[minbound, maxbound]] * 3)

    def __repr__(self) -> str:

        repr_points = np.array_repr(self.points)

        return f"Circle3D({repr_points})"
