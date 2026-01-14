from copy import deepcopy
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import ConstantInputWarning

from calc.auxiliar import get_regular_grid_from_irregular_data
from object.air import ReconstructedAirVelocityField, AirVelocityField
from plotting.auxiliar import ColorBar2D
#import deprecated.calc.thermal
from plotting.plot import plot_interpolated

#
# def plot_curtain2d(ax, avf, mg_along_tc, data_resolution, plotting_resolution=None, background_contour_kwargs=None,
#                    quiver_kwargs=None, include_turbulence=False, thermal_mask=False):
#     if plotting_resolution is None:
#         plotting_resolution = data_resolution
#     constant_length = 4
#     width = 2
#
#     if background_contour_kwargs is None:
#         background_contour_kwargs = {'levels': 3 * plotting_resolution}
#     else:
#         background_contour_kwargs = {'levels': 3 * plotting_resolution} | background_contour_kwargs
#     if quiver_kwargs is None:
#         quiver_kwargs = {'constant_length': constant_length,
#                          'width': width,
#                          'scale_units': 'y', 'pivot': 'mid'
#                          }
#     else:
#         quiver_kwargs = quiver_kwargs
#         #| {'constant_length': constant_length,
#                                       #  'width':           width,
#                                       #  'scale_units':     'y',
#                                       #  'pivot': 'mid'
#                                       #  }
#
#
#
#     # inset Axes....
#     my_slice_x = slice(int(0.9 * data_resolution), int(1 * data_resolution)) # slice(data_resolution // 4, 3 * (data_resolution // 4))
#     my_slice_y = slice(int(0.8 * data_resolution), int(0.9 * data_resolution))
#     x1, x2, y1, y2 = plot_x[my_slice_x,my_slice_y].min(), plot_x[my_slice_x,my_slice_y].max(), plot_z[my_slice_x,my_slice_y].min(), plot_z[my_slice_x,my_slice_y].max()  # subregion of the original image
#     axins = ax.inset_axes(
#         [0.6, 0.1, 0.3, 0.3],
#         xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
#
#     _, _ = plot_interpolated(axins, 'quiver',
#                                                       plot_x[my_slice_x,my_slice_y], plot_z[my_slice_x,my_slice_y],
#                                                       plot_vx[my_slice_x,my_slice_y], plot_vy[my_slice_x,my_slice_y],
#                                                       color_array=False,
#                                                       background_contour_kwargs=background_contour_kwargs,
#                                                       kwargs=quiver_kwargs,
#                                                       resolution=plotting_resolution // 5
#                                                       )
#
#
#     ax.indicate_inset_zoom(axins, edgecolor="black")
#     return m_stream, m_contour

def plot_curtain3d(ax, avf, mg_along_tc, data_resolution, plotting_resolution=None, include_turbulence=False,
                    **plot_surface_kwargs):
    if plotting_resolution is None:
        plotting_resolution = data_resolution
    constant_length = 4
    width = 2

    plot_x = np.linalg.norm(mg_along_tc[..., :2], axis=-1)
    plot_z = mg_along_tc[..., 2]

    v_wind = avf.get_velocity(mg_along_tc.reshape(-1,3), t=0, include='wind').reshape((data_resolution,data_resolution,3))
    v_thermal = avf.get_velocity(mg_along_tc.reshape(-1,3), t=0, include=['thermal']).reshape((data_resolution,data_resolution,3))
    if include_turbulence:
        v_turbulence = avf.get_velocity(mg_along_tc, t=0, include=['turbulence'])
        v_turbulence[..., 0] = np.where(np.abs(v_thermal[..., 2]) > 0.2, v_turbulence[..., 0], 0)
        v_turbulence[..., 1] = np.where(np.abs(v_thermal[..., 2]) > 0.2, v_turbulence[..., 1], 0)
        v_turbulence[..., 2] = np.where(np.abs(v_thermal[..., 2]) > 0.2, v_turbulence[..., 2], 0)
    else:
        v_turbulence = np.zeros(shape=v_thermal.shape)
    wind_thermal = v_wind + np.nan_to_num(v_thermal) + np.nan_to_num(v_turbulence)
    plot_vx = np.linalg.norm(wind_thermal[..., :2], axis=-1)
    plot_vy = wind_thermal[..., -1]
    my_cmap = plot_surface_kwargs.pop('cmap')
    my_norm = plot_surface_kwargs.pop('norm')
    m_surface = ax.plot_surface(X=mg_along_tc[..., 0].reshape((data_resolution,data_resolution)),
                                Y=mg_along_tc[..., 1].reshape((data_resolution,data_resolution)),
                                Z=mg_along_tc[..., 2].reshape((data_resolution,data_resolution)),
                                facecolors=my_cmap(my_norm(np.linalg.norm(wind_thermal, axis=-1))),
                                **plot_surface_kwargs
                                )
    return m_surface

def plot_synthetic_curtain3d_syn_vs_decomposed(ax_arr, avf_real, avf, z_array, data_resolution, plotting_resolution=None,
                                             plot_surface_kwargs=None
                                             ):
    if plotting_resolution is None:
        plotting_resolution = data_resolution
    constant_length = 4
    width = 2

    my_cmap = 'jet'
    quiver_kwargs = {'constant_length': constant_length,
                     'width': width,
                     'scale_units': 'y', 'pivot': 'mid'
                     }

    tc = avf.get_thermal_core(z_array, t=0)
    tc_arc_length = np.linalg.norm(tc, axis=-1)
    mg_along_tc = np.empty(shape=(data_resolution * data_resolution, 3))
    for i, z in enumerate(z_array):
        mg_along_tc[i * data_resolution:(i + 1) * data_resolution, :2] = tc
        mg_along_tc[i * data_resolution:(i + 1) * data_resolution, 2] = z

    mg_along_tc = mg_along_tc.reshape((data_resolution, data_resolution,3))

    ax_real, ax_decomposed = ax_arr
    m_stream_real, m_contour_real =plot_curtain3d(ax_real, avf_real, mg_along_tc, data_resolution=data_resolution,
                                                  plotting_resolution=plotting_resolution,
                                                  include_turbulence=True,**plot_surface_kwargs)

    m_stream_decomposed, m_contour_decomposed = plot_curtain3d(ax_decomposed, avf, mg_along_tc, data_resolution=data_resolution,
                                                  plotting_resolution=plotting_resolution,
                                                  include_turbulence=True,**plot_surface_kwargs)

    cbar = plt.colorbar(ScalarMappable(norm=m_contour_decomposed.norm, cmap=m_contour_decomposed.cmap,
                                       ax=ax_decomposed,
                                       label='$ \\vert V_{{Air}} \\vert (m/s)$'))
    cbar.vmin = m_contour_decomposed.norm.vmin
    cbar.vmax = m_contour_decomposed.norm.vmax

    ax_real.set_ylabel('Z (m)')
    ax_decomposed.set_ylabel('Z (m)')

    ax_real.set_xlabel('X\' (m)')
    ax_decomposed.set_xlabel('X\' (m)')

    ax_decomposed.yaxis.set_visible(False)

    ax_real.set_title('Synthetic Data')
    ax_decomposed.set_title('Decomposed Data')

    return (m_stream_real, m_contour_real), (m_stream_decomposed, m_contour_decomposed)

def get_air_velocity_curtain(ax, avf, include_turbulence,
                             background_contour_kwargs,
                             quiver_kwargs,
                             plotting_resolution,thermal_mask=None, z_array=None, mg_along_tc=None,
                             inset_limits=None, inset_zoom_factor=3
                             ):
    if mg_along_tc is None:
        mg_along_tc,_ = get_curtain_array(avf=avf, z_array=z_array, data_resolution=len(z_array))
    mg_original_shape = mg_along_tc.shape
    v_wind = avf.get_velocity(mg_along_tc.reshape(-1, 3), t=0, relative_to_ground=True, include='wind').reshape(mg_original_shape)
    v_thermal = avf.get_velocity(mg_along_tc.reshape(-1, 3), t=0, relative_to_ground=True, include=['thermal']).reshape(mg_original_shape)
    if include_turbulence:
        v_turbulence = avf.get_velocity(mg_along_tc, t=0, relative_to_ground=True, include=['turbulence'])
    else:
        v_turbulence = np.zeros(shape=v_thermal.shape)

    if thermal_mask is not None:
        v_turbulence[..., 0] = np.where(thermal_mask, v_turbulence[..., 0], 0)
        v_turbulence[..., 1] = np.where(thermal_mask, v_turbulence[..., 1], 0)
        v_turbulence[..., 2] = np.where(thermal_mask, v_turbulence[..., 2], 0)

        v_thermal[..., 0] = np.where(thermal_mask, v_thermal[..., 0], 0)
        v_thermal[..., 1] = np.where(thermal_mask, v_thermal[..., 1], 0)
        v_thermal[..., 2] = np.where(thermal_mask, v_thermal[..., 2], 0)
    thermal_mask = ~np.any(np.isnan(v_thermal), axis=-1)

    wind_thermal = v_wind + np.nan_to_num(v_thermal) + np.nan_to_num(v_turbulence)
    plot_x = np.vstack(mg_original_shape[0]
                       * [np.append([0],
                                    np.cumsum(
                                        np.linalg.norm(
                                            np.diff(
                                                np.vstack([mg_along_tc[0, :, 0], mg_along_tc[1, :, 1]]).T,
                                                axis=0),
                                            axis=-1)
                                    )
                                    )
                          ]
                       )
    plot_y = mg_along_tc[...,2]
    plot_vx = np.linalg.norm(wind_thermal[..., :2], axis=-1)
    plot_vy = wind_thermal[..., -1]

    m_stream, m_contour = plot_interpolated(ax, 'quiver',
                                            plot_x, plot_y,
                                            plot_vx, plot_vy,
                                            color_array=False,
                                            background_contour_kwargs=background_contour_kwargs,
                                            kwargs=quiver_kwargs,
                                            resolution=plotting_resolution
                                            )

    if inset_limits is not None:
        inset_kwargs = deepcopy(quiver_kwargs)
        x1, x2, y1, y2 = inset_limits
        inset_mask = np.logical_and(np.logical_and((plot_x >= x1), (plot_x <= x2)),
                                    np.logical_and((plot_y >= y1), (plot_y <= y2)))
        plot_x_inset = np.ma.masked_array(plot_x, mask=~inset_mask)
        plot_z_inset = np.ma.masked_array(plot_y, mask=~inset_mask)
        plot_vx_inset = np.ma.masked_array(plot_vx, mask=~inset_mask)
        plot_vy_inset = np.ma.masked_array(plot_vy, mask=~inset_mask)

        bbb = ax.dataLim
        delta = bbb.max - bbb.min
        x1_axis = (x1 - bbb.min[0]) / delta[0]
        x2_axis = (x2 - bbb.min[0]) / delta[0]
        y1_axis = (y1 - bbb.min[1]) / delta[1]
        y2_axis = (y2 - bbb.min[1]) / delta[1]
        inset_size = [inset_zoom_factor * (x2_axis - x1_axis), inset_zoom_factor * (y2_axis - y1_axis)]
        axins = ax.inset_axes(
            [0.95 - inset_size[0], 0.05, inset_size[0], inset_size[1]],
            xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
        inset_kwargs['constant_length'] = inset_kwargs['constant_length'] / 2
        _, _ = plot_interpolated(axins, 'quiver',
                                 plot_x_inset, plot_z_inset,
                                 plot_vx_inset, plot_vy_inset,
                                 color_array=False,
                                 background_contour_kwargs=background_contour_kwargs,
                                 kwargs=inset_kwargs,
                                 resolution=plotting_resolution // 5 if isinstance(plotting_resolution, int) else [
                                     r // 5 for r in plotting_resolution]
                                 )
        axins.autoscale()

        ax.indicate_inset_zoom(axins, edgecolor="black")
    return m_stream, m_contour, thermal_mask


def get_curtain_array(avf: Union[AirVelocityField, ReconstructedAirVelocityField], z_array: np.ndarray, data_resolution: int):
    tc = avf.get_thermal_core(z_array, t=0)
    tc_arc_length = np.append([0],np.cumsum(np.linalg.norm(np.diff(tc, axis=0), axis=-1))) # np.linalg.norm(tc, axis=-1)

    x_mg, z_mg = np.meshgrid(tc_arc_length, z_array)
    mg_along_arc_length = np.stack([x_mg, z_mg], axis=-1)
    mg_along_tc = np.empty(shape=(data_resolution * data_resolution, 3))

    for i, z in enumerate(z_array):
        mg_along_tc[i * data_resolution:(i + 1) * data_resolution, :2] = tc
        mg_along_tc[i * data_resolution:(i + 1) * data_resolution, 2] = z

    mg_along_tc = mg_along_tc.reshape((data_resolution, data_resolution, 3))

    return mg_along_tc, mg_along_arc_length


def plot_curtain_2d_colorbar(ax: Axes, avf:ReconstructedAirVelocityField, data_resolution: int,
                             mg_along_tc_array=None, ax_colorbar: Axes=None,
                             colorbar: ColorBar2D=None, do_inset=False, **config_kwargs):

    zlims = config_kwargs['zlims']
    if zlims is None:
        zlims = avf.wind_spline['X'].x_min, avf.wind_spline['X'].x_max
    z_array = np.linspace(zlims[0], zlims[1], data_resolution)

    if mg_along_tc_array is None:
        mg_along_tc, mg_arc_length = get_curtain_array(avf=avf, z_array=z_array, data_resolution=len(z_array))
    else:
        mg_along_tc, mg_arc_length = mg_along_tc_array

    quiver_kwargs = config_kwargs['quiver_kwargs']
    v_wind = avf.get_velocity(mg_along_tc.reshape(-1, 3),
                              t=0,
                              relative_to_ground=True,
                              include='wind').reshape(mg_along_tc.shape)
    v_thermal = avf.get_velocity(mg_along_tc.reshape(-1, 3),
                                 t=0,
                                 relative_to_ground=True,
                                 include=['thermal']).reshape(mg_along_tc.shape)

    wind_thermal = v_wind + np.nan_to_num(v_thermal)

    plot_x = mg_arc_length[..., 0]
    plot_y = mg_arc_length[..., 1]
    plot_vx = np.linalg.norm(wind_thermal[..., :2], axis=-1)
    plot_vy = wind_thermal[..., -1]
    plot_Xi, plot_Yi, (plot_Vxi, plot_Vyi) = get_regular_grid_from_irregular_data(plot_x, plot_y,
                                                                                  plot_vx, plot_vy,
                                                                                  resolution=data_resolution)
    plot_Vxi = plot_Vxi.data
    plot_Vyi = plot_Vyi.data

    quiver_plot_vx_i = plot_Vxi / np.linalg.norm(np.stack([plot_Vxi, plot_Vyi], axis=-1), axis=-1)
    quiver_plot_vy_i = plot_Vyi / np.linalg.norm(np.stack([plot_Vxi, plot_Vyi], axis=-1), axis=-1)

    quiver_downsample = quiver_kwargs.pop('downsample')
    quiver_plot_vx_i = quiver_plot_vx_i[::quiver_downsample, ::quiver_downsample]
    quiver_plot_vy_i = quiver_plot_vy_i[::quiver_downsample, ::quiver_downsample]
    quiver_plot_Xi = plot_Xi[::quiver_downsample, ::quiver_downsample]
    quiver_plot_Yi = plot_Yi[::quiver_downsample, ::quiver_downsample]

    if colorbar is None:
        norm_horizontal_i = config_kwargs['norm_horizontal_i']
        if norm_horizontal_i is None:
            norm_horizontal_i = Normalize(-plot_Vxi.max() / 3, np.linalg.norm(v_wind, axis=-1).max())

        norm_vertical_i = config_kwargs['norm_vertical_i']
        if norm_vertical_i is None:
            norm_vertical_i = Normalize(min(plot_Vyi.min(), -plot_Vyi.max() / 4), plot_Vyi.max())

        negative_boundary = config_kwargs['negative_boundary']
        boundary = config_kwargs['boundary']
        colorbar = ColorBar2D(cmaps=[plt.colormaps['Purples'], plt.colormaps['Blues'], plt.colormaps['hot_r']],
                             norms=[norm_horizontal_i, norm_horizontal_i, norm_vertical_i],
                             negative_boundary=negative_boundary, boundary=boundary,
                             name='myname')
    else:
        norm_horizontal_i = colorbar.norms[1]
        norm_vertical_i = colorbar.norms[2]

    my_X = np.stack((plot_Vxi, plot_Vyi), axis=-1)
    my_colors = colorbar(my_X.reshape(-1, 2)).reshape(my_X.shape[:-1] + (4,))

    ax.imshow(my_colors, aspect='auto', interpolation='bicubic',
              extent=(plot_Xi.min(), plot_Xi.max(), plot_Yi.min(), plot_Yi.max()),
              origin='lower')
    ax.quiver(quiver_plot_Xi, quiver_plot_Yi, quiver_plot_vx_i, quiver_plot_vy_i,
              **quiver_kwargs)
    if ax_colorbar is not None:
        colorbar.get_color_bar(ax_colorbar, extent=[max(norm_horizontal_i.vmin, 0),
                                                    norm_horizontal_i.vmax,
                                                    config_kwargs['negative_boundary'], plot_Vyi.max()])
    if do_inset:

        ax3d = ax.inset_axes(bounds=config_kwargs['inset_bounds'], projection='3d')
        ax3d.plot(mg_along_tc[0, ..., 0], mg_along_tc[0, ..., 1], zs=mg_along_tc[..., 2].min(), zdir='z', c='k',
                  linewidth=2)

        ax3d.plot_surface(mg_along_tc[..., 0], mg_along_tc[..., 1], mg_along_tc[..., 2], facecolors=my_colors,
                          rstride=1, cstride=1, )
        ax3d.set_aspect('equalxy')
        # ax3d.set_axis_off()
        ax3d.set_facecolor((1, 1, 1, 0.6))
        #
        ax3d.set_xticklabels([], visible=False)
        ax3d.set_yticklabels([], visible=False)
        ax3d.set_zticklabels([], visible=False)
        ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # # # make the grid lines transparent
        ax3d.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax3d.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax3d.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax3d.view_init(elev=config_kwargs['elev'], azim=config_kwargs['azim'])
    else:
        ax3d = None
    return ax, ax_colorbar,ax3d, mg_along_tc, mg_arc_length, z_array, colorbar



def plot_synthetic_curtain_syn_vs_decomposed(ax_arr, avf_real: AirVelocityField, avf: ReconstructedAirVelocityField,
                                             z_array,
                                             data_resolution, plotting_resolution=None,
                                             background_contour_kwargs=None,
                                             quiver_kwargs=None, include_turbulence=None, inset_limits=None, inset_zoom_factor = 3
                                             ):
    ax_real, ax_decomposed = ax_arr
    if plotting_resolution is None:
        plotting_resolution = data_resolution
    if include_turbulence is None:
        include_turbulence = {'decomposed': False, 'real': True}
    mg_along_tc, _ = get_curtain_array(avf=avf, z_array=z_array, data_resolution=data_resolution)
    thermal_mask = None
    m_contour_dict = {}
    m_stream_dict = {}
    for label, current_axis, current_avf in zip(['decomposed', 'real'], [ax_decomposed, ax_real], [avf, avf_real]):
        m_stream_dict[label], m_contour_dict[label], thermal_mask = get_air_velocity_curtain(current_axis, avf=current_avf,
                                                                               mg_along_tc=mg_along_tc,
                                                                               thermal_mask=thermal_mask,
                                                                               include_turbulence=include_turbulence[label],
                                                                               plotting_resolution=plotting_resolution,
                                                                               background_contour_kwargs=background_contour_kwargs,
                                                                               quiver_kwargs=quiver_kwargs,
                                                                                             inset_limits=inset_limits,
                                                                                             inset_zoom_factor=inset_zoom_factor
        )

    #inset Axes....
    if 'norm' in background_contour_kwargs:
        cbar = plt.colorbar(ScalarMappable(norm=m_contour_dict['decomposed'].norm, cmap=m_contour_dict['decomposed'].cmap),
                                           ax=ax_decomposed,
                                           label='$ \\vert V_{{Air}} \\vert (m/s)$')
        cbar.vmin = m_contour_dict['decomposed'].norm.vmin
        cbar.vmax = m_contour_dict['decomposed'].norm.vmax

    ax_real.set_ylabel('Z (m)')
    ax_decomposed.set_ylabel('Z (m)')

    ax_real.set_xlabel('X\' (m)')
    ax_decomposed.set_xlabel('X\' (m)')

    ax_decomposed.yaxis.set_visible(False)

    ax_real.set_title('Ground Truth')
    ax_decomposed.set_title('Decomposed Data')

    return m_stream_dict, m_contour_dict


def get_pearson_and_RMS_from_path(avf_real, df_real, avf, df_dec, N, do_plots=False, lims=None, extrapolate=False):

    df_merge = pd.merge(df_dec, df_real[['bird_name', 'time', 'dXdT_bird_real', 'dYdT_bird_real', 'dZdT_bird_real']], on=['bird_name', 'time'], how='left')

    if lims is None:
        x_min = -40
        x_max = 40
        y_min = -40
        y_max = 40
        z_min = avf.df_bins['Z_thermal_min'].min()
        z_max = avf.df_bins['Z_thermal_max'].max()
    else:
        x_min = lims['x_min']
        x_max = lims['x_max']
        y_min = lims['y_min']
        y_max = lims['y_max']
        z_min = lims['z_min']
        z_max = lims['z_max']


    if not np.isscalar(N):
        Nx, Ny, Nz = N
    else:
        Nx, Ny, Nz = N, N, N

    mgs = np.meshgrid(np.linspace(x_min, x_max,Nx),
                      np.linspace(y_min, y_max,Ny),
                      np.linspace(z_min, z_max,Nz),
                      indexing='ij'
                      )
    XYZ = np.stack(mgs, axis=-1)

    velocity_decomposed, velocity_decomposed_components = avf.get_velocity_agg(X=XYZ, t=0, relative_to_ground=False, return_components=True, velocity_kwargs={'thermal': {'extrapolate': extrapolate}})
    velocity_decomposed_components['turbulence'] = avf.get_velocity_fluctuations(XYZ, 0)
    velocity_real, velocity_real_components = avf_real.get_velocity(X=XYZ, t=0, relative_to_ground=False, return_components=True)
    velocity_real_components['thermal'][:,:,:,0] = velocity_real_components['rotation'][:,:,:,0]
    velocity_real_components['thermal'][:,:,:,1] = velocity_real_components['rotation'][:,:,:,1]
    velocity_decomposed_thermal = velocity_decomposed_components['thermal']
    velocity_decomposed_wind = velocity_decomposed_components['wind']

    if extrapolate:
        for bin_z_idx in np.arange(Nz):
            for i_comp in range(3):
                na_indices = np.argwhere(np.isnan(velocity_decomposed_thermal[...,bin_z_idx, i_comp]))
                for na_idx in na_indices:
                    i1, i2 = na_idx[0], na_idx[1]
                    x_idx_min = max([i1 - 1, 0])
                    x_idx_max = min([i1 + 2, Nx - 1])
                    y_idx_min = max([i2 - 1, 0])
                    y_idx_max = min([i2 + 2, Ny - 1])
                    velocity_decomposed_thermal[i1, i2, bin_z_idx, i_comp] =np.nanmean(velocity_decomposed_thermal[x_idx_min: x_idx_max, y_idx_min : y_idx_max, bin_z_idx, i_comp])




    comparison_dict_per_component = {}

    for comp in velocity_decomposed_components.keys():
        for i_coord in range(3):
            good_mask = ~np.isnan(velocity_decomposed_components[comp][..., i_coord])
            comparison_dict_per_component[(comp, i_coord)] ={}
            rms = np.linalg.norm(velocity_real_components[comp][good_mask, i_coord] - velocity_decomposed_components[comp][good_mask, i_coord])
            rms = rms / np.sqrt(velocity_real_components[comp][good_mask, i_coord].size)

            comparison_dict_per_component[(comp, i_coord)]['RMS'] = rms

            try:
                pearson_result = stats.pearsonr(velocity_real_components[comp][good_mask, i_coord].flatten(),
                                                      velocity_decomposed_components[comp][good_mask, i_coord].flatten())
            except ConstantInputWarning as e:
                print(comp, i_coord)
                print(e)
                pearson_result = (np.nan, np.nan)

            comparison_dict_per_component[(comp, i_coord)]['pearson_r'] = pearson_result[0]
            comparison_dict_per_component[(comp, i_coord)]['pearson_p'] = pearson_result[1]
            comparison_dict_per_component[(comp, i_coord)]['pearson_N'] = velocity_real_components[comp][good_mask, i_coord].size

    for i_coord, coord in enumerate(['X', 'Y', 'Z']):
        dec_values = df_merge[f'd{coord}dT_bird_4'].values
        real_values = df_merge[f'd{coord}dT_bird_real'].values
        good_mask = ~np.isnan(dec_values)
        good_mask = np.logical_and(good_mask, ~np.isnan(real_values))
        dec_values = dec_values[good_mask]
        real_values = real_values[good_mask]
        rms = np.linalg.norm(real_values - dec_values)
        rms = rms / np.sqrt(dec_values.size)

        pearson_result = stats.pearsonr(dec_values,
                                              real_values)
        comparison_dict_per_component[('bird', i_coord)] ={}
        comparison_dict_per_component[('bird', i_coord)]['RMS'] = rms
        comparison_dict_per_component[('bird', i_coord)]['pearson_r'] = pearson_result[0]
        comparison_dict_per_component[('bird', i_coord)]['pearson_p'] = pearson_result[1]
        comparison_dict_per_component[('bird', i_coord)]['pearson_N'] = df_merge[f'd{coord}dT_bird_real'].values.size


    if do_plots:

        fig, ax_arr = plt.subplots(4,3,  #layout='tight'
                                   )
        for i_coord in range(3):
            coord = 'XYZ'[i_coord]
            for i_comp, (comp) in enumerate(velocity_decomposed_components.keys()):

                good_mask = ~np.isnan(velocity_decomposed_components['thermal'][..., 2])

                ax_arr[i_comp, i_coord].scatter(velocity_real_components[comp][..., i_coord].flatten(),
                                            velocity_decomposed_components[comp][..., i_coord].flatten(), s=2)
                ax_arr[i_comp, i_coord].set_title(f'{comp}_{coord}\n'
                                                  + ', '.join([ f'{k}={v:.3g}' for k,v in comparison_dict_per_component[(comp, i_coord)].items()]))
                ax_arr[i_comp, i_coord].set_xlabel('Real (m/s)')
                ax_arr[i_comp, i_coord].set_ylabel('Decomposed (m/s)')

            comp = 'bird'
            ax_arr[-1, i_coord].scatter(df_merge[f'd{coord}dT_bird_4'].values, df_merge[f'd{coord}dT_bird_real'].values, s=2)
            ax_arr[-1, i_coord].set_xlabel('Real (m/s)')
            ax_arr[-1, i_coord].set_ylabel('Decomposed (m/s)')
            ax_arr[-1, i_coord].set_title(f'{comp}_{coord}\n'
                                              + ', '.join([ f'{k}={v:.3g}' for k,v in comparison_dict_per_component[(comp, i_coord)].items()]))
        plt.show(block=True)

    return comparison_dict_per_component
