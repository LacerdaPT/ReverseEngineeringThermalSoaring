import logging
import os

import matplotlib
import matplotlib as mpl
from itertools import product
from types import LambdaType

from matplotlib.collections import LineCollection

from calc.auxiliar import get_regular_grid_from_irregular_data

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from plotting.auxiliar import fig_ax_handler, get_mesh_grid_from_polar


def line_plot3D(ax, x, y, z, color=None, norm=None, cmap=None):
    if norm is None:
        norm = Normalize()
    if cmap is None:
        cmap = mpl.colormaps['viridis']

    color_array = cmap(norm(color))
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, colors=color_array)
    line_array = []
    for i, seg in enumerate(segments):
        line, = ax.plot3D(seg[:, 0], seg[:, 1], seg[:, 2], color=color_array[i])
        line.set_solid_capstyle('round')
        line_array.append(line)

    x_lims = ax.get_xlim()
    ax.set_ylim(np.mean(y) - (x_lims[1] - x_lims[0]) / 2, np.mean(y) + (x_lims[1] - x_lims[0]) / 2)

    return line_array, lc


def line_plot2D(ax, x, y, color=None, norm=None, cmap=None):
    if norm is None:
        norm = Normalize()
    if cmap is None:
        cmap = mpl.colormaps['viridis']

    color_array = cmap(norm(color))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=color_array)
    line_array = []
    for i, seg in enumerate(segments):
        line, = ax.plot(seg[:, 0], seg[:, 1], color=color_array[i])
        line.set_solid_capstyle('round')
        line_array.append(line)

    x_lims = ax.get_xlim()
    # ax.set_ylim(np.mean(y) - (x_lims[1] - x_lims[0]) / 2, np.mean(y) + (x_lims[1] - x_lims[0]) / 2)

    return line_array, lc


def plot_stream_interpolated(ax, Xi, Yi, vx_i, vy_i, color_array='no_color', **kwargs):

    if color_array is False:
        artist = ax.streamplot(x=Xi, y=Yi, u=vx_i, v=vy_i, **kwargs)
    else:
        if isinstance(color_array, (list, np.ndarray)):
            artist = ax.streamplot(x=Xi, y=Yi, u=vx_i, v=vy_i, color=color_array, **kwargs)
        else:
            if color_array == 'no_color':
                color_array = 'k'
            artist = ax.streamplot(x=Xi, y=Yi, u=vx_i, v=vy_i, color=color_array, **kwargs)
    # artist_dict[plot_type] = ax_dict[plot_type].contourf(xi, yi, zi[plot_type], levels=30, cmap=cmap)
    return artist


def plot_stem_arrows(ax, x, y, **kwargs):

    my_cmap = kwargs['cmap']
    if isinstance(my_cmap, str):
        my_cmap = matplotlib.colormaps[my_cmap]
    my_norm = kwargs['norm']
    artist_lines = ax.hlines(x, np.zeros_like(y), y,
                             colors=my_cmap(my_norm(y)))

    artist_arrows = ax.scatter(y, x, marker='>', c=y, **kwargs)
    return ax, artist_lines, artist_arrows


def plot_quiver_interpolated(ax, Xi, Yi, vx_i, vy_i, color_array=None, resolution=30,
                             constant_length=None, **kwargs):

    if constant_length is not None:
        norm = np.linalg.norm(np.dstack([vx_i.data, vy_i.data]), axis=-1)

        vx_i_to_plot = vx_i / norm * constant_length
        vy_i_to_plot = vy_i / norm * constant_length
        plotting_kwargs = {'scale_units': 'xy',
                           'scale': 0.2,
                           'units': 'xy'}

        plotting_kwargs.update(kwargs)
    else:
        plotting_kwargs = kwargs.copy()
        vx_i_to_plot = vx_i
        vy_i_to_plot = vy_i

    if color_array is False:
        artist = ax.quiver(Xi, Yi, vx_i_to_plot, vy_i_to_plot, **plotting_kwargs)
    else:
        artist = ax.quiver(Xi, Yi, vx_i_to_plot, vy_i_to_plot, color_array, **plotting_kwargs)

    return artist


def plot_contour_tri(ax, x_array, y_array, color_array, **kwargs):
    # if kwargs is None:
    #     kwargs = {}
    # if ('norm' in kwargs) and ('levels' in kwargs):
    #     norm = kwargs['norm']
    #     levels = kwargs['levels']
    #     if (norm.vmin is not None) and (norm.vmax is not None):
    #         kwargs['levels'] = np.linspace(norm.vmin, norm.vmax, levels)
    artist = ax.contourf(x_array, y_array, color_array, **kwargs)
    return artist


def plot_interpolated(ax, plot_type, x_array, y_array, vx_array, vy_array=None, color_array='norm',
                      background_contour_kwargs=None, resolution=30, **kwargs):
    if (kwargs is None) or (kwargs == {}):
        kwargs = {'kwargs': {}}
    if np.all(vx_array == 0) and (vy_array is not None) and np.all(vy_array == 0):
        return None, None
    plot_type = plot_type.lower().strip()

    Xi, Yi, v_i = get_regular_grid_from_irregular_data(x_array, y_array, vx_array, vy_array, resolution=resolution)

    if isinstance(color_array, LambdaType):
        color_array = np.apply_along_axis(color_array,-1, np.dstack(v_i) )
    elif isinstance(color_array, str):
        kwargs.update({'color': color_array})
        color_array = False

    if background_contour_kwargs is not False:
        bg_kwargs = {'alpha':       1,
                     'zorder':      1,
                     'antialiased': True,
                     'levels':      resolution}
        if background_contour_kwargs:
            bg_kwargs.update(background_contour_kwargs)

        contour_color = np.linalg.norm(np.dstack(v_i), axis=-1)
        contour_artist = ax.contourf(Xi, Yi, contour_color,
                                     **bg_kwargs)
        color_array = False
    else:
        contour_artist = None

    if plot_type == 'quiver':
        artist = plot_quiver_interpolated(ax, Xi, Yi, *v_i, color_array, **(kwargs['kwargs']))
    elif plot_type == 'stream':
        artist = plot_stream_interpolated(ax, Xi, Yi, *v_i, color_array, **(kwargs['kwargs']))
        artist = artist.lines
    elif plot_type == 'contour':
        if 'levels' not in kwargs:
            kwargs['levels'] = resolution
        if not np.isscalar(color_array):
            contour_color = color_array
        else:
            if len(v_i) > 1:
                contour_color = np.linalg.norm(np.dstack(v_i), axis=-1)
            else:
                contour_color = v_i[0]
        artist = plot_contour_tri(ax, Xi, Yi, contour_color.data, **kwargs['kwargs'])

    return artist, contour_artist


def get_outlier_idx(arr, n_sigmas=3):
    mean = np.nanmean(arr)
    sigma = np.nanstd(arr, )
    print('sigma', sigma)

    mask = np.abs(arr - mean) < n_sigmas * sigma
    indices_to_remove = np.argwhere(np.invert(mask)).flatten()
    indices_to_keep = np.argwhere(mask).flatten()

    # good_array = arr[indices_to_keep]

    return indices_to_remove, indices_to_keep


def plot_tracks_scatter(ax, x_data, y_data, z_data, color_data=None, with_projection=False, **plot_args):
    for data in [x_data, y_data, z_data]:
        if isinstance(data, pd.Series):
            data = data.values

    color_is_categorical = False
    if color_data is not None:
        if isinstance(color_data, pd.Series):
            color_data = color_data.values

        color_is_categorical = (color_data.dtype == object) or (color_data.dtype == str)
        if color_is_categorical:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            color_data = le.fit_transform(color_data)
            # This 0.5 simply puts the labels in the center of the color patch
            color_data = color_data + 0.5
            unique_labels = le.classes_
            n_unique_labels = len(unique_labels)

            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            cmap = ListedColormap(cycle[:n_unique_labels])
            norm = BoundaryNorm(np.arange(n_unique_labels), cmap.N)
            plot_args['norm'] = norm
            plot_args['cmap'] = cmap


    if color_data is not None:
        artist = ax.scatter3D(x_data,
                              y_data,
                              z_data,
                              c=color_data,
                              **plot_args)
    else:
        artist = ax.scatter3D(x_data,
                              y_data,
                              z_data,
                              **plot_args)


    x_lims = ax.get_xlim()
    ax.set_ylim(np.mean(y_data) - (x_lims[1] - x_lims[0]) / 2,
                np.mean(y_data) + (x_lims[1] - x_lims[0]) / 2)
    ax.set_anchor('C')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    if color_is_categorical:
        color_bar = plt.colorbar(artist, ax=ax, )
        colorbar_ticks = np.arange(n_unique_labels) + 0.5
        label_mask = (colorbar_ticks > color_bar.ax.get_ylim()[0]) & (colorbar_ticks < color_bar.ax.get_ylim()[1])
        color_bar.set_ticks(colorbar_ticks[label_mask])
        color_bar.ax.set_yticklabels(unique_labels[label_mask])

    artists_array = [artist]
    if with_projection:
        z_level = np.min(z_data) # np.min([np.min(z_data), 0])
        projection_artist = ax.plot(x_data, y_data, zs=z_level, zdir='z', c='k', alpha=0.3, )
        artists_array.append(projection_artist)

    return ax, artists_array


def plot_tracks_line2d(ax, x_data, y_data, color_data=None, thickness_data=None, with_projection=False, **plot_args):
    for data in [x_data, y_data]:
        if isinstance(data, pd.Series):
            data = data.values

    if thickness_data is not None:
        if isinstance(thickness_data, pd.Series):
            thickness_data = thickness_data.values

        # thickness_data = 5*(thickness_data - np.min(thickness_data)) / (np.max(thickness_data) - np.min(thickness_data)) + 1
    else:
        if 'lw' in plot_args:
            thickness_data = plot_args['lw'] * np.ones(len(x_data))
            plot_args.pop('lw')
        elif 'linewidth' in plot_args:
            thickness_data = plot_args['linewidth']* np.ones(len(x_data))
            plot_args.pop('linewidth')
        elif 'linewidths' in plot_args:
            thickness_data = plot_args['linewidths']* np.ones(len(x_data))
            plot_args.pop('linewidths')
        else:
            thickness_data = plt.rcParams['lines.linewidth']* np.ones(len(x_data))


    if color_data is not None:
        if isinstance(color_data, pd.Series):
            color_data = color_data.values

        if 'norm' in plot_args:
            norm = plot_args['norm']
            plot_args.pop('norm')
        else:
            norm = Normalize()

        if 'cmap' in plot_args:
            if isinstance(plot_args['cmap'], str):
                cmap = mpl.colormaps[plot_args['cmap']]
            else:
                cmap = plot_args['cmap']
            plot_args.pop('cmap')
        else:
            cmap = mpl.colormaps['viridis']
        # color_data = norm(color_data)
        # color_data = cmap(color_data)

    if color_data is not None:

        N=len(x_data)

        artist = LineCollection(segments=[[(xi, yi), (xi1, yi1)]
                                 for xi, yi, xi1, yi1 in zip(x_data[:-1],y_data[:-1],
                                                             x_data[1:],y_data[1:])],
                                array=color_data,
                                linewidths=thickness_data,
                                cmap=cmap,
                                norm=norm, **plot_args)
        ax.add_artist(artist)
        # for i in np.arange(N - 1):
        #     a = ax.plot(x_data[i:i + 2],
        #                 y_data[i:i + 2],
        #                 color=color_data[i],
        #                 linewidth=thickness_data[i],
        #                 **plot_args)
        #     artist.append(a)
    else:
        artist = ax.plot(x_data,
                         y_data,
                         **plot_args)

    artists_array = [artist]

    return ax, artists_array


def plot_tracks_line(ax, x_data, y_data, z_data=None, color_data=None, with_projection=False,projection_kwargs=None, **plot_args):
    if z_data is None:
        return plot_tracks_line_2d(ax, x_data, y_data, color_data=color_data, **plot_args)
    else:
        return plot_tracks_line_3d(ax, x_data, y_data, z_data, color_data=color_data, with_projection=with_projection,
                                   projection_kwargs=projection_kwargs,
                                   **plot_args)



def plot_tracks_line_2d(ax, x_data, y_data, color_data=None, **plot_args):
    for data in [x_data, y_data]:
        if isinstance(data, pd.Series):
            data = data.values

    color_is_categorical = False
    if color_data is not None:
        if isinstance(color_data, pd.Series):
            color_data = color_data.values

        color_is_categorical = (color_data.dtype == object) or (color_data.dtype == str)
        if color_is_categorical:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            color_data = le.fit_transform(color_data)
            # This 0.5 simply puts the labels in the center of the color patch
            color_data = color_data + 0.5
            unique_labels = le.classes_
            n_unique_labels = len(unique_labels)

            cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            cmap = ListedColormap(cycle[:n_unique_labels])
            norm = BoundaryNorm(np.arange(n_unique_labels), cmap.N)
            plot_args['norm'] = norm
            plot_args['cmap'] = cmap

        if 'norm' in plot_args:
            norm = plot_args['norm']
            plot_args.pop('norm')
        else:
            norm = Normalize()

        if 'cmap' in plot_args:
            if isinstance(plot_args['cmap'], str):
                cmap = mpl.colormaps[plot_args['cmap']]
            else:
                cmap = plot_args['cmap']
            plot_args.pop('cmap')
        else:
            cmap = mpl.colormaps['viridis']
        color_data = norm(color_data)
        color_data = cmap(color_data)

    if color_data is not None:
        artist = []
        N=len(x_data)
        for i in np.arange(N - 1):
            a = ax.plot(x_data[i:i + 2],
                        y_data[i:i + 2],
                        color=color_data[i],
                        **plot_args)
            artist.append(a)
    else:
        artist = ax.plot(x_data,
                         y_data,
                         **plot_args)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    if color_is_categorical:
        color_bar = plt.colorbar(artist, ax=ax, )
        colorbar_ticks = np.arange(n_unique_labels) + 0.5
        label_mask = (colorbar_ticks > color_bar.ax.get_ylim()[0]) & (colorbar_ticks < color_bar.ax.get_ylim()[1])
        color_bar.set_ticks(colorbar_ticks[label_mask])
        color_bar.ax.set_yticklabels(unique_labels[label_mask])

    artists_array = [artist]
    return ax, artists_array



def plot_tracks_line_3d(ax, x_data, y_data, z_data, color_data=None, with_projection=False, projection_kwargs=None, **plot_args):
    for data in [x_data, y_data, z_data]:
        if isinstance(data, pd.Series):
            data = data.values

    if projection_kwargs is None:
        projection_kwargs = { 'c': 'k', 'alpha': 0.05}
    else:
        projection_kwargs = { 'c': 'k', 'alpha': 0.05} | projection_kwargs
    color_is_categorical = False
    if color_data is not None:
        if isinstance(color_data, str):
            color_data = [color_data] * len(x_data)
        else:
            if isinstance(color_data, pd.Series):
                color_data = color_data.values


            if 'norm' in plot_args:
                norm = plot_args['norm']
                plot_args.pop('norm')
            else:
                norm = Normalize()

            if 'cmap' in plot_args:
                if isinstance(plot_args['cmap'], str):
                    cmap = mpl.colormaps[plot_args['cmap']]
                else:
                    cmap = plot_args['cmap']
                plot_args.pop('cmap')
            else:
                cmap = mpl.colormaps['viridis']
            color_data = norm(color_data)
            color_data = cmap(color_data)

    if color_data is not None:
        artist = []
        N=len(x_data)
        for i in np.arange(N - 1):
            a = ax.plot(x_data[i:i + 2],
                        y_data[i:i + 2],
                        z_data[i:i + 2],
                        color=color_data[i],
                        **plot_args)
            artist.append(a)
    else:
        artist = ax.plot(x_data,
                         y_data,
                         z_data,
                         **plot_args)

    ax.set_aspect('equalxy')
    # ax.set_anchor('C')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    artists_array = [artist]
    if with_projection:
        projection_artist = ax.plot(x_data, y_data, zs=0, # - (np.nanmax(z_data) - np.nanmin(z_data)) * 0.2,
                                    zdir='z', **projection_kwargs)
        artists_array.append(projection_artist)

    return ax, artists_array


def plot_contour3D(df, rho, phi, Z, offset=None, fig=None, ax=None, alpha=0.8, cmap='RdYlBu_r', norm=None,
                   interpolation_method='nearest'):
    if offset is None:
        offset = df[Z].min() - 1

    fig, ax = fig_ax_handler(fig, ax, ax_kwargs={'projection': '3d'})
    # Surface
    meshgrid = get_mesh_grid_from_polar(df, rho, phi, depend_key=Z, limits={rho: {'min': 0},
                                                                            phi: {'min': -np.pi,
                                                                                  'max': np.pi
                                                                                  }},
                                        interpolation_method=interpolation_method
                                        )

    rhoi = meshgrid[rho]
    phii = meshgrid[phi]

    Zi = meshgrid[Z]

    Xi, Yi = rhoi * np.cos(phii), rhoi * np.sin(phii)

    # norm = Normalize(vmin=0, vmax=5)

    artist = ax.contourf(X=Xi, Y=Yi, Z=Zi, cmap=cmap, offset=offset, levels=10, alpha=alpha)
    fig.tight_layout()

    sm = plt.cm.ScalarMappable(cmap=mpl.colormaps[cmap])
    sm.set_array(Zi)

    return fig, ax, sm, artist

def plot_surface3D_from_polar(df, rho, phi, Z, color=None, offset=None, fig=None, ax=None, cmap='RdYlBu_r',
                              interpolation_method='nearest'):
    if color is None:
        color = Z
    if offset is None:
        offset = df[Z].min() - 1

    fig, ax = fig_ax_handler(fig, ax, ax_kwargs={'projection': '3d'})
    # Surface

    meshgrid = get_mesh_grid_from_polar(df, rho, phi, depend_key=Z,
                                        limits={rho: {'min': 0},
                                                phi: {'min': -np.pi,
                                                      'max': np.pi
                                                      }
                                                }, interpolation_method=interpolation_method
                                        )

    rhoi = meshgrid[rho]
    phii = meshgrid[phi]
    colori = meshgrid[color]
    Zi = meshgrid[Z]

    Xi, Yi = rhoi * np.cos(phii), rhoi * np.sin(phii)

    norm = Normalize(vmin=0, vmax=5)

    m = ax.plot_surface(X=Xi, Y=Yi, Z=Zi,
                        facecolors=mpl.colormaps[cmap](norm(colori)), alpha=0.4)

    # ax.plot_surface(X, Y, Z, cmap="autumn_r", lw=0.5, rstride=1, cstride=1,alpha=0.5)
    # Lines on the bottom
    # ax.contour(X, Y, Z, 10, cmap="autumn_r", linestyles="solid", offset=offset)
    # Lines on the surface
    # ax.contour(X, Y, Z, 10, colors="k", linestyles="solid")

    return fig, ax, m


def plot_cuboid(ax, dimensions, at, position_type='bottom', kwargs=None, plot_vertices=False):
    if kwargs is None:
        kwargs = {'alpha': 0.1,
                  'color': 'b'}

    if position_type == 'bottom':
        points = [[at[0] - dimensions[0] / 2, at[0] + dimensions[0] / 2],
                  [at[1] - dimensions[1] / 2, at[1] + dimensions[1] / 2],
                  [at[2], at[2] + dimensions[2]]]
    else:
        points = [[at[i] - dimensions[i] / 2, at[i] + dimensions[i] / 2] for i in range(3)]

    from itertools import product

    xx, yy = np.meshgrid(points[0], points[1])
    artists = []
    art = ax.plot_surface(xx, yy, points[2][0] * np.ones(shape=xx.shape), **kwargs)
    artists.append(art)
    art = ax.plot_surface(xx, yy, points[2][1] * np.ones(shape=xx.shape), **kwargs)
    artists.append(art)

    yy, zz = np.meshgrid(points[1], points[2])
    art = ax.plot_surface(points[0][0] * np.ones(shape=yy.shape), yy, zz, **kwargs)
    artists.append(art)
    art = ax.plot_surface(points[0][1] * np.ones(shape=yy.shape), yy, zz, **kwargs)
    artists.append(art)

    xx, zz = np.meshgrid(points[0], points[2])
    art = ax.plot_surface(xx, points[1][0] * np.ones(shape=xx.shape), zz, **kwargs)
    artists.append(art)
    art = ax.plot_surface(xx, points[1][1] * np.ones(shape=xx.shape), zz, **kwargs)
    artists.append(art)

    if plot_vertices:
        for p in product(*points):
            art = ax.scatter3D(*p)
            artists.append(art)

    return artists


def plot_scatter3D(ax, X_array, Y_array, Z_array, color_array=None, kwargs=None, aspect='equal'):
    default_kwargs = {'cmap': 'bwr',
                      'alpha': 1,
                      's': 4}
    if kwargs is not None:
        default_kwargs.update(kwargs)  # 'norm': Normalize(vmin=0, vmax=np.pi / 3),
    if color_array is not None:
        default_kwargs.update({'c': color_array})

    kwargs = default_kwargs

    artist = ax.scatter(xs=X_array,
                        ys=Y_array,
                        zs=Z_array,
                        **kwargs
                        )

    if aspect == 'equal':
        ax.set_aspect('equalxy')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    return artist


def plot_contour_levels_polar_to_polar(df, rho, phi, Z, fig=None, ax=None):
    fig, ax = fig_ax_handler(fig, ax, ax_kwargs={'projection': 'polar'})

    meshgrid = get_mesh_grid_from_polar(df, rho, phi, depend_key=Z, limits={rho: {'min': 0},
                                                                            phi: {'min': -np.pi,
                                                                                  'max': np.pi}
                                                                            }
                                        )

    rhoi = meshgrid[rho]
    phii = meshgrid[phi]
    Zi = meshgrid[Z]

    try:
        # Set up a regular grid of interpolation points
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        artist = ax.contourf(phii, rhoi, Zi, levels=30)
        fig.colorbar(artist)
    except ValueError as e:
        print(e)
        return fig, ax
    else:
        return fig, ax, artist


def plot_contour_levels_polar_to_cart(df, rho, phi, Z, ax=None, offset=None, interpolation_method='nearest',
                                      contourplot_kwargs=None):
    df_calc = df.copy()

    # df_calc['Xi'] = df_calc[rho] * np.cos(df_calc[phi])
    # df_calc['Yi'] = df_calc[rho] * np.cos(df_calc[phi])
    if contourplot_kwargs is None:
        contourplot_kwargs = {}
    meshgrid = get_mesh_grid_from_polar(df_calc, rho, phi, depend_key=Z, limits={rho: {'min': 0},
                                                                                 phi: {'min': -np.pi,
                                                                                       'max': np.pi}
                                                                                 },
                                        interpolation_method=interpolation_method
                                        )

    xi = meshgrid['X']
    yi = meshgrid['Y']
    Zi = meshgrid[Z]

    if offset is None:
        offset = 3
    try:
        # Set up a regular grid of interpolation points
        # Create the mesh in polar coordinates and compute corresponding Z.

        artist_obj = ax.contourf(xi, yi, Zi, levels=30, **contourplot_kwargs)
        # plt.colorbar(artist_obj, ax=ax)
    except ValueError as e:
        print(e)
        return ax
    else:
        return ax, artist_obj


def plot_quiver_3d(X, Y, Z, Vx, Vy, Vz, color_array=None, cmap=None, norm=None, ax_kwargs=None, plot_kwargs=None,
                   fig=None, ax=None):
    fig, ax = fig_ax_handler(fig, ax, ax_kwargs)
    if plot_kwargs is None:
        plot_kwargs = {}

    if norm is None:
        norm = Normalize()
    if color_array is None:
        artist = ax.quiver(X, Y, Z, Vx, Vy, Vz,
                           **plot_kwargs)
    else:
        color = mpl.colormaps[cmap](norm(color_array))
        artist = ax.quiver(X, Y, Z, Vx, Vy, Vz, colors=color,
                           **plot_kwargs)
    # if color is not None:
    #    artist.set_array()

    return fig, ax, artist


def plot_control_top_view(ax, air_obj, x_array, y_array, z_array, color_array, thermal_core_estimate, window_size, thermal_kwargs=None,
                          track_kwargs=None):
    if track_kwargs is None:
        track_kwargs = {'s': 7,
                        'norm': Normalize(),
                        'alpha': 1}

    if thermal_kwargs is None:
        thermal_kwargs = {'levels':20,
                          'antialiased':False,
                          'cmap': 'Reds'
                          }

    artists = {}
    # ====================================================
    #                   CALCULATIONS
    # ====================================================
    Z_level = z_array[-1]
    # Thermal Core
    core = air_obj.get_thermal_core(Z_level)

    # Thermal Profile
    rho_mg, phi_mg = np.meshgrid(np.linspace(0, 50, 15), np.linspace(-np.pi, np.pi, 20))

    xy_mg = np.array([rho_mg * np.cos(phi_mg) + core[0],
                      rho_mg * np.sin(phi_mg) + core[1]])

    thermal_mg = []
    for i, j in product(np.arange(xy_mg[1].shape[0]), np.arange(xy_mg[1].shape[1])):
        thermal_mg.append(air_obj.get_velocity([xy_mg[0][i, j],
                                                xy_mg[1][i, j],
                                                Z_level],
                                               include='thermal')[-1])

    thermal_mg = np.array(thermal_mg)
    thermal_mg = thermal_mg.reshape(xy_mg[1].shape)

    # Wind
    wind = air_obj.get_velocity([x_array[-1],
                                 y_array[-1],
                                 z_array[-1]],
                                include='wind')[:2]

    # ====================================================
    #                   PLOTTING
    # ====================================================

    # ====================================================
    #                   THERMAL PROFILE
    # ====================================================

    artists['thermal_profile'] = ax.contourf(xy_mg[0], xy_mg[1], thermal_mg, **thermal_kwargs)

    # This for loop removes the white lines between contours
    for c in artists['thermal_profile'].collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.000000000001)

    # ====================================================
    #                   BIRD TRACK
    # ====================================================
    artists['track'] = ax.scatter(x_array,
                                  y_array,
                                  c=color_array,
                                  **track_kwargs)


    # ====================================================
    #               THERMAL CORES
    # ====================================================
    artists['real_core'] = ax.scatter(core[0], core[1], marker='+', label='Real\nCore')

    artists['estimated_core'] = ax.scatter(*thermal_core_estimate,
                                           s=80, marker=(5, 1),
                                           label='Estimated\nCore')

    # ====================================================
    #               WIND PLOT
    # ====================================================

    #gridspec = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[3,1],hspace=0.1,wspace=0.1)
    #ax = fig.add_subplot(gridspec[:,1])
    #ax_wind = fig.add_subplot(gridspec[1,0])

    # wind = wind / np.linalg.norm(wind)
    # ax_wind.quiver(0, 0, wind[0], wind[1],
    #               pivot='middle',
    #               units='height',
    #               scale=2,
    #               scale_units='x',
    #               width=0.01)
    # ax_wind.set_xlim(-0.5, 0.5)
    # ax_wind.set_ylim(-0.5, 0.5)
    # ax_wind.axis('off')

    # Put the quiver plot on the top left corner
    artists['wind'] = ax.quiver(x_array[-1] - window_size / 2 + 0.10 * window_size,
                                y_array[-1] - window_size / 2 + 0.90 * window_size,
                                wind[0], wind[1],
                                pivot='middle',
                                units='height',
                                scale=0.5,
                                scale_units='x',
                                width=0.005)

    artists['wind_quiverkey'] = ax.quiverkey(artists['wind'], 0.1, 1.04, 2, label=f'${2} m/s$', labelpos='E', )

    # ====================================================
    #                   PLOT STYLE
    # ====================================================

    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    # Keeps the most recent track point in the center of the plot
    ax.set_xlim(x_array[-1] - window_size / 2,
                x_array[-1] + window_size / 2)
    ax.set_ylim(y_array[-1] - window_size / 2,
                y_array[-1] + window_size / 2)

    return artists


def inspect_flock(df, X_col='X', Y_col='Y', Z_col='Z', time_col='time', bird_label_col='bird_name', color=None,
                  air_velocity_field=None, plot_thermal_core_estimate=True, path_to_save=None):
    exclude_list = []
    for bird in df[bird_label_col].unique():
        df_bird = df[df[bird_label_col] == bird]
        if df_bird.empty:
            continue

        if color is None:
            color_data = None
        else:
            if isinstance(color, str):
                color_data = df_bird[color]
            elif isinstance(color, LambdaType):
                color_data = df_bird.apply(color)

        fig = plt.figure(figsize=(15,12), tight_layout=True)

        gridspec_wrapper = fig.add_gridspec(1, 2, width_ratios=[3,2])
        ax3d = fig.add_subplot(gridspec_wrapper[0], projection='3d')
        # ===============================        LEFT PANEL    ================================
        gridspec_line_plots = GridSpecFromSubplotSpec(3, 1, hspace=0.4, height_ratios=[1, 1, 1],
                                                      subplot_spec=gridspec_wrapper[1])

        # ===============================     RIGHT PANEL =====================================
        ax_line_plots = [fig.add_subplot(gridspec_line_plots[0]),
                         fig.add_subplot(gridspec_line_plots[1]),
                         fig.add_subplot(gridspec_line_plots[2])]


        # Bird track
        _, im = plot_tracks_scatter(ax=ax3d,
                                    x_data=df_bird[X_col],
                                    y_data=df_bird[Y_col],
                                    z_data=df_bird[Z_col],
                                    color_data=color_data,
                                    s=7,
                                    alpha=1, with_projection=True)

        if air_velocity_field:
            # Real Core
            zi = np.linspace(0.1, df_bird['Z'].max(), 100)
            ti = np.linspace(0.1, df_bird[time_col].max(), 100)
            core = air_velocity_field.get_thermal_core(zi, t=ti)
            ax3d.plot3D(xs=core[:, 0], ys=core[:, 1], zs=zi, alpha=0.9,  # linewidth=2,
                      label='Real\nCore')

        if plot_thermal_core_estimate:# Core Estimate
            if not df_bird['thermal_core_estimate_X_avg'].dropna().empty:
                ax3d.plot3D(xs=df_bird['thermal_core_estimate_X_avg'],
                            ys=df_bird['thermal_core_estimate_Y_avg'],
                            zs=df_bird['thermal_core_estimate_Z_avg'],
                            alpha=0.9,
                            linewidth=2,
                            label='Estimated\nCore')

        ax3d.set_aspect('equalxy')
        ax3d.set_anchor('C')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')

        if color:
            cbar = fig.colorbar(im[0], ax=ax3d, label='Bank Angle (rad)')
        else:
            cbar = None

        for i, coord in enumerate([X_col, Y_col, Z_col]):
            ax_line_plots[i].plot(df_bird[time_col], df_bird[coord])
            ax_line_plots[i].set_xlabel('time')
            ax_line_plots[i].set_ylabel(coord + ' (m)')
            ax_line_plots[i].grid()

        fig.suptitle(bird)
        fig.tight_layout()
        if path_to_save:
            fig.canvas.draw()
            fig.savefig( os.path.join(path_to_save, bird + '.png'))
            plt.close(fig)
        else:
            def on_press(event):
                if event.key == 'e':
                    exclude_list.append(bird)
                    print(f'{bird} excluded..')
            fig.canvas.mpl_connect('key_press_event', on_press)
            plt.show(block=True)

    return fig, [ax3d, ax_line_plots], im, cbar, exclude_list
