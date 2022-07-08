# -*- coding: utf-8 -*-
"""
@author: Cristiano Pizzamiglio

"""

import math
from fractions import Fraction

import numpy as np
import pytest

from circle_fitting_3d import Circle3D


@pytest.mark.parametrize(
    ("points", "message_expected"),
    [
        ([[1, 0], [-1, 0], [0, 1]], "The points must be 3D."),
        ([[2, 0, 1], [-2, 0, -3]], "There must be at least 3 points."),
        ([[0, 0, 0], [1, 1, 1], [2, 2, 2]], "The points must not be collinear."),
    ],
)
def test_failure(points, message_expected):

    with pytest.raises(ValueError, match=message_expected):
        Circle3D(points)


@pytest.mark.parametrize(
    ("points", "center_expected", "radius_expected"),
    [
        ([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], [0, 0, 0], 1),
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [Fraction(1, 3)] * 3, math.sqrt(2 / 3)),
        ([[2, 0, 0], [0, 1, 0], [-2, 0, 0], [0, -1, 0]], [0, 0, 0], math.sqrt(2.5)),
    ],
)
def test_best_fit(points, center_expected, radius_expected):

    circle_3d = Circle3D(points)

    assert circle_3d.center.is_close(center_expected, abs_tol=1e-9)
    assert math.isclose(circle_3d.radius, radius_expected)


@pytest.mark.parametrize(
    ("circle_3d", "t", "array_expected"),
    [
        (
            Circle3D([[1, 0, 0], [0, 1, 0], [-1, 0, 0]]),
            [0, math.pi / 2, math.pi],
            np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]]),
        ),
    ],
)
def test_equation(circle_3d, t, array_expected):

    assert np.allclose(circle_3d.equation(t), array_expected)
