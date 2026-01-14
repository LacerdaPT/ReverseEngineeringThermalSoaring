from typing import Iterable, List

import matplotlib
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, Colormap
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator

from calc.auxiliar import parse_projection_string


class ColorBar2D(Colormap):
    def __init__(self, cmaps: List[matplotlib.colors.Colormap],
                 norms: List[matplotlib.colors.Normalize], name: str, X: Iterable = None,
                 boundary: int = None, negative_boundary: int = None):
        super().__init__(name)
        self._isinit = False
        assert len(cmaps) == len(norms), "length of cmaps must match length of norms"
        self.cmaps = [None] * 3
        self.norms = [None] * 3
        if len(cmaps) == 2:
            self.cmaps[1:] = cmaps
            self.norms[1:] = norms
        else:
            self.cmaps = cmaps
            self.norms = norms
        self.boundary = boundary
        self.negative_boundary = negative_boundary
        self.X = X
        self._interpolators = []
        if self.boundary is None:
            self.boundary = 1.0

        if self.X is not None:
            self._init()

    def _init(self):
        # if self.negative_boundary is None:
        self.normed_boundary = self.norms[2](self.boundary)
        self.normed_zero = self.norms[2](0)


        my_points = np.stack(np.meshgrid(np.linspace(0, 1, self.N // 4),
                                          np.linspace(0, 1, self.N // 4)
                                          ),
                              axis=-1).reshape(-1,2)
        my_values = np.empty(shape=my_points.shape[:1] + (4,))
        negative_mask = np.full(my_points.shape[0], fill_value=False)
        if self.negative_boundary:
            self.normed_negative_boundary = self.norms[2](self.negative_boundary)
            negative_mask = (self.normed_negative_boundary <= my_points[:,1]) & (my_points[:,1] < self.normed_zero)
            negative_weights = 1 - (my_points[negative_mask,1] - self.normed_negative_boundary)/ (self.normed_zero - self.normed_negative_boundary)
            # my_values[my_mask] = self.cmap1(my_points[my_mask,0] / self.normed_boundary)
            my_values[negative_mask] = (np.multiply(self.cmaps[0](my_points[negative_mask,0]).T, negative_weights).T
                                  + np.multiply(self.cmaps[1](my_points[negative_mask,0]).T, 1 - negative_weights).T)


        my_mask = (self.normed_zero <= my_points[:,1]) & (my_points[:,1] < self.normed_boundary)
        my_weights = (self.normed_boundary - my_points[my_mask,1])/ (self.normed_boundary - self.normed_zero)
        # my_values[my_mask] = self.cmap1(my_points[my_mask,0] / self.normed_boundary)
        my_values[my_mask] = (np.multiply(self.cmaps[1](my_points[my_mask,0]).T, my_weights).T
                              + np.multiply(self.cmaps[2](my_points[my_mask,1]).T, 1 - my_weights).T)

        vertical_only_mask = (~my_mask) & (~negative_mask)
        my_values[vertical_only_mask] = self.cmaps[2](my_points[vertical_only_mask,1])

        my_values = np.array(my_values)
        self._interpolators = [LinearNDInterpolator(my_points, my_values[:,i]) for i in range(4)]
        self._isinit = True

    def __call__(self, X, alpha=None, bytes=False):
        r"""
        Parameters
        ----------
        X : float or int, `~numpy.ndarray` or scalar
            The data value(s) to convert to RGBA.
            For floats, *X* should be in the interval ``[0.0, 1.0]`` to
            return the RGBA values ``X*100`` percent along the Colormap line.
            For integers, *X* should be in the interval ``[0, Colormap.N)`` to
            return RGBA values *indexed* from the Colormap with index ``X``.
        alpha : float or array-like or None
            Alpha must be a scalar between 0 and 1, a sequence of such
            floats with shape matching X, or None.
        bytes : bool
            If False (default), the returned RGBA values will be floats in the
            interval ``[0, 1]`` otherwise they will be `numpy.uint8`\s in the
            interval ``[0, 255]``.

        Returns
        -------
        Tuple of RGBA values if X is scalar, otherwise an array of
        RGBA values with a shape of ``X.shape + (4, )``.
        """
        if not self._isinit:
            self._init()

        X_normed = np.empty(X.shape)
        negative_mask = X[:, 1] < 0
        X_normed[negative_mask] = np.column_stack((self.norms[0](X[negative_mask,0]),
                                                   self.norms[2](X[negative_mask,1]))
                                                  )
        positive_mask = (0 <= X[:, 1]) & (X[:, 1] < self.boundary)
        X_normed[positive_mask] = np.column_stack((self.norms[1](X[positive_mask,0]),
                                                    self.norms[2](X[positive_mask,1]))
                                                   )
        vertical_mask = (~negative_mask) & (~positive_mask)

        X_normed[vertical_mask] = np.column_stack((self.norms[1](X[vertical_mask,0]),
                                                    self.norms[2](X[vertical_mask,1]))
                                                   )

        xa = np.array(X_normed, copy=True)
        if not xa.dtype.isnative:
            # Native byteorder is faster.
            xa = xa.byteswap().view(xa.dtype.newbyteorder())
        if xa.dtype.kind == "f":
            xa *= self.N
            # xa == 1 (== N after multiplication) is not out of range.
            xa[xa == self.N] = self.N - 1
        # Pre-compute the masks before casting to int (which can truncate
        # negative values to zero or wrap large floats to negative ints).
        mask_under = xa < 0
        mask_over = xa >= self.N
        # If input was masked, get the bad mask from it; else mask out nans.
        mask_bad = X.mask if np.ma.is_masked(X) else np.isnan(xa)
        with np.errstate(invalid="ignore"):
            # We need this cast for unsigned ints as well as floats
            xa = xa.astype(int)
        xa[mask_under] = self._i_under
        xa[mask_over] = self._i_over
        xa[mask_bad] = self._i_bad

        rgba = np.column_stack([self._interpolators[i](np.clip(X_normed, 0, 1)) for i in range(4)])

        if alpha is not None:
            alpha = np.clip(alpha, 0, 1)
            if bytes:
                alpha *= 255  # Will be cast to uint8 upon assignment.
            if alpha.shape not in [(), xa.shape]:
                raise ValueError(
                    f"alpha is array-like but its shape {alpha.shape} does "
                    f"not match that of X {xa.shape}")
            rgba[..., -1] = alpha

        if not np.iterable(X):
            rgba = tuple(rgba)
        return rgba

    def get_color_bar(self,ax, extent=None):
        if extent is None:
            horizontal_min = np.min([self.norms[0].vmin, self.norms[1].vmin])
            horizontal_max = np.max([self.norms[0].vmax, self.norms[1].vmax])
            vertical_min = self.norms[2].vmin
            vertical_max = self.norms[2].vmax
        else:
            horizontal_min, horizontal_max,vertical_min,vertical_max = extent
        x,y = np.meshgrid(np.linspace(horizontal_min, horizontal_max, 30, endpoint=True),
                             np.linspace(vertical_min, vertical_max, 30, endpoint=True))
        xy = np.stack((x,y), axis=-1)
        c = self(xy.reshape(-1,2)).reshape(xy.shape[:-1] + (4,))
        twod_cbar = ax.imshow(c, origin='lower', interpolation='bicubic', aspect='auto',
                              extent=(horizontal_min, horizontal_max,
                                      vertical_min, vertical_max))
        return twod_cbar

def get_2d_colorbar(ax, norms, cmaps, mixing_function=None):
    if mixing_function is None:
        mixing_function = get_2d_colormap_angular_mixing
    norm_horizontal, norm_vertical = norms

    cx, cy = np.meshgrid(np.linspace(norm_horizontal.vmin, norm_horizontal.vmax, 30, endpoint=True),
                         np.linspace(norm_vertical.vmin, norm_vertical.vmax, 30, endpoint=True))
    c, (norm_horizontal, norm_vertical) = mixing_function(cx, cy, norms=norms, cmaps=cmaps)
    twod_cbar = ax.imshow(c, origin='lower', interpolation='bicubic', extent=(norm_horizontal.vmin, norm_horizontal.vmax,
                                                                     norm_vertical.vmin, norm_vertical.vmax))
    return twod_cbar

def get_2d_colormap_angular_mixing(x, y, norms, cmaps):
    norm_horizontal, norm_vertical = norms
    cmap_horizontal, cmap_vertical = cmaps

    angle_mg = np.abs(np.arctan2(y, x))

    color_horizontal = plt.colormaps[cmap_horizontal](norm_horizontal(x))
    color_vertical = plt.colormaps[cmap_vertical](norm_vertical(y))

    color_norm = np.linalg.norm(np.stack([color_horizontal[..., :3], color_vertical[..., :3]], axis=-1), axis=-1)

    color_horizontal[..., :3] = np.where(color_norm == 0,
                                         0,
                                         color_horizontal[..., :3] / color_norm)
    color_vertical[..., :3] = np.where(color_norm == 0,
                                       0,
                                       color_vertical[..., :3] / color_norm)
    color_both = np.ones_like(color_horizontal)
    for i in range(3):
        color_both[..., i] = (np.cos(angle_mg) ** 2 * color_horizontal[..., i]
                              + np.sin(angle_mg) ** 2 * color_vertical[..., i])
        color_both[..., i] = color_both[..., i] / color_both[..., i].max()

    return color_both, (norm_horizontal, norm_vertical)


def get_parameters_title(df_parameters, bird_name):
    current_parameters = df_parameters[df_parameters['bird_name'] == bird_name].iloc[0]

    distance_to_core = np.linalg.norm([current_parameters['x_center'] - 0,
                                       current_parameters['y_center'] - 0])

    title = f'bird={current_parameters["bird_name"]}, ' \
            f'Ravg={round(current_parameters["radius_avg"], 2)}, ' \
            f'Rstd={round(current_parameters["radius_sigma"], 2)}, ' \
            f'A={round(current_parameters["thermal"]["A"], 2)}' \
            f', sigma={round(current_parameters["thermal"]["sigma"], 2)}, ' \
            f'rotation={round(current_parameters["rotation"], 2)}, ' \
            f'noise={round(current_parameters["noise_level"], 2)}, ' \
            f'distance_to_core={round(distance_to_core, 2)}'

    return title


def fig_ax_handler(fig, ax, ax_kwargs=None):
    if ax_kwargs is None:
        ax_kwargs = {}
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, **ax_kwargs)

    return fig, ax


def get_color_with_custom_alphas(color_array, alpha_array, norm=None, cmap=None):
    alpha_array = np.array(alpha_array.copy())
    color_array = np.array(color_array.copy())

    assert color_array.shape == alpha_array.shape, 'color_array and alpha_array must have the same shape'
    if norm is None:
        norm = Normalize(vmin=color_array.min(),
                         vmax=color_array.max())
    if cmap is None:
        cmap = None  # Defaults to virids

    alpha_array = (alpha_array - alpha_array.min()) / (alpha_array.max() - alpha_array.min())
    cmap = mpl.colormaps[cmap]

    color_array = norm(color_array)
    color_array = cmap(color_array)

    color_array[..., -1] = alpha_array

    return color_array


def get_cross_section_meshgrid(limits, n_points=20, section_value=0, cross_section_type='XY'):

    if np.isscalar(n_points):
        n_points = [n_points, n_points]

    (first_var, second_var,
     first_index, second_index,
     section_index, section_var) = parse_projection_string(cross_section_type)

    plotting_vars = [first_var, second_var]
    plotting_indices = [first_index, second_index]
    plotting_meshgrids = np.meshgrid(*[np.linspace(limits[i][0], limits[i][1], n_points[i]) for i in range(2)])
    meshgrid_shape = plotting_meshgrids[0].shape
    section_meshgrid = np.ones(shape=meshgrid_shape) * section_value

    meshgrids = np.empty(shape=(meshgrid_shape[0], meshgrid_shape[1], 3))

    meshgrids[:, :, first_index] = plotting_meshgrids[0]
    meshgrids[:, :, second_index] = plotting_meshgrids[1]
    meshgrids[:, :, section_index] = section_meshgrid

    XYZ_meshgrid = np.dstack([meshgrids[:, :, 0], meshgrids[:, :, 1], meshgrids[:, :, 2]])

    return XYZ_meshgrid, plotting_vars, plotting_indices, section_var, section_index


def get_interpolation(*args, method='rbf', evalute=False):
    if method == 'rbf':
        interpolation = interpolate.Rbf(*args, function='linear')
    elif method == 'nearest':
        interpolation = interpolate.NearestNDInterpolator(x=args[:-1], y=args[-1])
    else:
        interpolation = interpolate.LinearNDInterpolator(points=args[:-1], values=args[-1])

    return interpolation


def get_mesh_grid_from_polar(df, rho, phi, depend_key, limits=None, resolution=30, interpolation_method=None,
                             polar=True):
    independent_keys = [rho, phi]
    df_calc = df.copy()
    if limits is None:
        limits = {key: {'min': df_calc[key].min(), 'max': df_calc[key].max()} for key in independent_keys}
    else:
        for key in independent_keys:
            if key not in limits.keys():
                limits = {key: {'min': df_calc[key].min(), 'max': df_calc[key].max()} for key in independent_keys}
            else:
                if 'min' not in limits[key].keys():
                    limits[key]['min'] = df_calc[key].min()
                if 'max' not in limits[key].keys():
                    limits[key]['max'] = df_calc[key].max()

    discretization_dict = {key: np.linspace(limits[key]['min'], limits[key]['max'], resolution)
                           for key in independent_keys}

    meshgrid_tuple = np.meshgrid(*discretization_dict.values())
    meshgrid_dict = {key: meshgrid_tuple[i] for i, key in enumerate(discretization_dict.keys())}

    if interpolation_method is not None:
        if polar:
            meshgrid_dict['X'] = meshgrid_dict[rho] * np.cos(meshgrid_dict[phi])
            meshgrid_dict['Y'] = meshgrid_dict[rho] * np.sin(meshgrid_dict[phi])
            df_calc['Xi'] = df_calc[rho] * np.cos(df_calc[phi])
            df_calc['Yi'] = df_calc[rho] * np.sin(df_calc[phi])

            del meshgrid_dict[rho]
            del meshgrid_dict[phi]

            keys_for_interpolation = ['Xi', 'Yi']
        else:
            keys_for_interpolation = independent_keys
            # Interpolate
        interpolation_obj = get_interpolation(*[df_calc[key].values for key in keys_for_interpolation + [depend_key]],
                                              method=interpolation_method)
        meshgrid_dict[depend_key] = interpolation_obj(*meshgrid_dict.values())

    return meshgrid_dict


def polygon_under_graph(xlist, ylist, bottom=0.0, **kwargs):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
    """

    poly_artist = PolyCollection([[(xlist[0], bottom), *zip(xlist, ylist), (xlist[-1], bottom)]], **kwargs)
    return poly_artist



def simple_axis(ax: plt.Axes):

    ax.spines[['right', 'top']].set_visible(False)
    visible_ticks = {
        "top":   False,
        "right": False
    }
    ax.tick_params(axis="x", which="both", **visible_ticks)
    ax.tick_params(axis="y", which="both", **visible_ticks)