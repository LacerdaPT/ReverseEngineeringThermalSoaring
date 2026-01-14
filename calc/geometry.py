import numpy as np
from numpy.linalg import LinAlgError


def get_curvature_from_trajectory(vx, vy, ax, ay):

    # (vx.ay - vy.ax) / (vx^2 + vy^2)^1.5
    if np.isclose(vx, 0) and np.isclose(vy, 0): # vx**2 + vy**2 == 0:
        return 0
    return (vx * ay - vy * ax) / (vx**2 + vy**2)**1.5


def get_radius_from_curvature(curvature):
    if np.isclose(curvature, 0):
        return np.inf
    else:
        #return 1/np.abs(curvature)
        return 1 / curvature


def radial_velocity_from_cartesian(x, y, vx, vy):
    return (x * vx + y * vy) / np.linalg.norm([x, y])


def angular_velocity_from_cartesian(x, y, vx, vy):
    return (vy * x - vx * y) / np.linalg.norm([x, y])


def get_cartesian_velocity_on_rotating_frame_from_inertial_frame(x, y, vx, vy):
    v_radial = radial_velocity_from_cartesian(x, y, vx, vy)
    v_angular = angular_velocity_from_cartesian(x, y, vx, vy)
    return v_radial, v_angular


def radial_acceleration_from_cartesian(x, y, vx, vy, ax, ay):
    rho = np.linalg.norm([x, y])
    radial_velocity = radial_velocity_from_cartesian(x, y, vx, vy)
    return (vx ** 2 + vy ** 2 + x * ax + y * ay - radial_velocity ** 2) / rho


def calculate_circle_lulu(x_array, y_array):
    if np.any(np.isnan(x_array)) or np.any(np.isnan(y_array)):
        return np.nan, np.nan, np.nan
    n = len(x_array)
    A = np.ones(shape=(n, 3))
    A[:, 0] = x_array
    A[:, 1] = y_array
    B = A[:, 0] ** 2 + A[:, 1] ** 2
    try:
        A_inv = np.linalg.pinv(A)
    except LinAlgError as e:
        print(e)
        return np.nan, np.nan, np.nan

    (a, b, c) = A_inv@B

    xc = a / 2
    yc = b / 2
    radius = np.sqrt(4 * c + a ** 2 + b ** 2) / 2

    radius_sign = np.sign(np.cross([x_array[1] - x_array[0], y_array[1] - y_array[0]],
                                   [x_array[-1] - x_array[0], y_array[-1] - y_array[0]])
                          )

    signed_radius = radius * radius_sign
    return xc, yc, signed_radius
