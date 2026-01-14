import numpy as np

from importlib import __import__

from scipy.interpolate import CubicSpline


def quadratic(r, theta, z, t, A_rotation, radius):
    import numpy as np
    if r > radius:
        return [0, 0]
    else:
        # K is the constant so that magnitude is A_rotation at r=radius/2
        K = 4 * A_rotation  / radius ** 2
        magnitude = K * r * (radius - r)

        return [-magnitude * np.sin(theta),
                magnitude * np.cos(theta)]


def gaussian(r, theta, z, t, A, radius):
    import numpy as np
    return A * np.exp(-np.power(r, 2.) / (2 * np.power(radius, 2.)))

def random_walk_gaussian(r, theta, z, t, A, radius_array, z_array):
    import numpy as np
    radius_spline = CubicSpline(z_array, radius_array)

    R = radius_spline(z)
    #magnitude = A * (1 + 0.3 * np.sin(2 * np.pi * z / 100.0))
    magnitude = np.clip(A * (30.0 / R) ** 2, 3, 6)
    return magnitude * np.exp(-np.power(r, 2.) / (2 * np.power(R, 2.)))

def expanding_gaussian(r, theta, z, t, A, radius, epsilon=1):
    import numpy as np
    magnitude = A
    R = (radius
         * (1 +  1 / 1000 * z )
         )
    return magnitude * np.exp(-np.power(r, 2.) / (2 * np.power(R, 2.)))

def fat_gaussian(r, theta, z, t, A, radius, epsilon=1):
    import numpy as np
    magnitude = A
    R = (radius
         * (1 +  1 / 1000 * z )
         * (1 + epsilon * np.cos(theta - np.pi / 4))
         )
    return magnitude * np.exp(-np.power(r, 2.) / (2 * np.power(R, 2.)))

def rotating_wind(X, t, magnitude, period):
    import numpy as np
    z = X[-1]
    return [magnitude * np.cos(2 * np.pi * z / period),
            magnitude * np.sin(2 * np.pi * z / period)]


def test_function(a,b):
    return a+b
