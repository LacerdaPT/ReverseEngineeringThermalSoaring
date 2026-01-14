from copy import deepcopy
from typing import List

import numpy as np
from matplotlib import gridspec, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from object.air import AirVelocityField
from plotting.auxiliar import get_cross_section_meshgrid, polygon_under_graph
from plotting.plot import plot_stem_arrows


def plot_summary_from_avf(ax, avf: AirVelocityField, section_z_value: float,
                          limits_xy=None,
                          limits_xz=None, section_y_value: float = 0,
                          resolution: int =51, t_value: float = 0, kwargs_list: List[dict]=None,
                          cax: List[Axes]=None
                          ):
    if limits_xy is None:
        limits_xy = [[-100, 100],
                     [-100, 100]]
    if limits_xz is None:
        limits_xz = [[10, 1000],
                 [0.1, 1000]]

    _kwargs_list = [{'cmap': 'turbulence_cmap_r', 'norm': Normalize(-2, 5)},
                   {'cmap': 'hot_r', 'norm': Normalize(-2.5, 5), 'linewidth': 0, 'antialiased': False},
                   {},
                   {'cmap': 'turbulence_cmap_r', 'norm': Normalize(-3, 4),'density':2,}
                                    ]
    if kwargs_list is not None:

        for i in range(4):
            _kwargs_list[i].update(deepcopy(kwargs_list[i]))

    kwargs_list = deepcopy(_kwargs_list)
    del _kwargs_list

    ax_wind, ax_profile, ax_thermalcore, ax_rotation = ax


    (XYZ_meshgrid_wind,
     plotting_vars, plotting_indices,
     section_var, section_index) = get_cross_section_meshgrid(limits=limits_xz,
                                                              cross_section_type='XZ',
                                                              n_points=resolution,
                                                              section_value=section_y_value)

    wind_array = avf.get_velocity(XYZ_meshgrid_wind, t=t_value, include=['wind'], relative_to_ground=True)



    ########################################################################################################################
    ##########################################              WIND                ############################################
    ########################################################################################################################

    plot_stem_arrows(ax_wind, XYZ_meshgrid_wind[:, 0, 2], np.linalg.norm(wind_array[:, 0, :], axis=-1), **(kwargs_list[0]))


    ########################################################################################################################
    ##########################################         THERMAL PROFILE          ############################################
    ########################################################################################################################

    (XYZ_meshgrid_profile, _, _, _, _) = get_cross_section_meshgrid(limits=limits_xy,
                                                                    cross_section_type='XY',
                                                                    n_points=2 * resolution,
                                                                    section_value=section_z_value)

    thermal_array = avf.get_velocity(XYZ_meshgrid_profile, t=t_value, include=['thermal', 'turbulence'],
                                     relative_to_ground=False)

    ax_profile.add_collection3d(polygon_under_graph(XYZ_meshgrid_profile[resolution + 1, :, 0],
                                                    np.max(thermal_array[..., 2], axis=0),
                                                    facecolors=['k'], alpha=0.25, zorder=0), zs=np.max(XYZ_meshgrid_profile[..., 1]), zdir='y')
    ax_profile.add_collection3d(polygon_under_graph(XYZ_meshgrid_profile[resolution + 1, :, 0],
                                                    np.min(thermal_array[..., 2], axis=0),
                                                    facecolors=['k'], alpha=0.25, zorder=0), zs=np.max(XYZ_meshgrid_profile[..., 1]), zdir='y')

    ax_profile.add_collection3d(polygon_under_graph(XYZ_meshgrid_profile[:, resolution + 1, 1],
                                                    np.max(thermal_array[..., 2], axis=1),
                                                    facecolors=['k'], alpha=0.25, zorder=0), zs=np.min(XYZ_meshgrid_profile[..., 0]), zdir='x')
    ax_profile.add_collection3d(polygon_under_graph(XYZ_meshgrid_profile[:, resolution + 1, 1],
                                                    np.min(thermal_array[..., 2], axis=1),
                                                    facecolors=['k'], alpha=0.25, zorder=0), zs=np.min(XYZ_meshgrid_profile[..., 0]), zdir='x')
    ax_profile.plot_surface(XYZ_meshgrid_profile[..., 0], XYZ_meshgrid_profile[..., 1],
                            thermal_array[..., 2], **(kwargs_list[1]), zorder=1000)

    ax_profile.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_profile.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_profile.set_xlabel('X~(\\mathrm{m})')
    ax_profile.set_ylabel('Y~(\\mathrm{m})')
    ax_profile.set_zlabel('$Vz~(\\mathrm{m~s}^{-1})$')

    ########################################################################################################################
    ##########################################          THERMAL CORE            ############################################
    ########################################################################################################################

    tc_z_array = np.linspace(limits_xz[1][0], limits_xz[1][1], 200)
    XY_core = avf.get_thermal_core(z=tc_z_array, t=0)

    core = np.hstack([XY_core, tc_z_array.reshape((tc_z_array.shape[0], 1))])

    artist = ax_thermalcore.plot(xs=core[:, 0],
                                 ys=core[:, 1],
                                 zs=core[:, 2], **kwargs_list[2])

    ax_thermalcore.plot(core[:, 0], core[:, 1], zs=np.min(core[:, -1]), zdir='z', color='k', alpha=0.3)

    ########################################################################################################################
    ##########################################         THERMAL ROTATION         ############################################
    ########################################################################################################################

    (XYZ_meshgrid_rotation, _, _, _, _) = get_cross_section_meshgrid(limits=limits_xy,
                                                                     cross_section_type='XY',
                                                                     n_points=5 * resolution,
                                                                     section_value=section_z_value)

    rotation_array = avf.get_velocity(XYZ_meshgrid_rotation, t=t_value, include=['rotation', 'turbulence'],
                                      relative_to_ground=False)

    color_array_rotation = np.linalg.norm(rotation_array[..., :2], axis=-1)
    lw_array_rotation = 2 * plt.rcParams['lines.linewidth'] * (color_array_rotation / np.max(color_array_rotation)
                                                               ) + plt.rcParams['lines.linewidth']
    artist = ax_rotation.streamplot(x=XYZ_meshgrid_rotation[:, :, 0],
                                    y=XYZ_meshgrid_rotation[:, :, 1],
                                    u=rotation_array[..., 0],
                                    v=rotation_array[..., 1],
                                    linewidth=lw_array_rotation,
                                    color=color_array_rotation,
                                    # minlength=0.01,
                                    # broken_streamlines=False,
                                    # maxlength=0.1,
                                    **(kwargs_list[3])

                                    )
    artist.lines.set_capstyle('round')
    # ax_wind.set_aspect('equal')
    ax_wind.set_xlabel('$V_\\mathrm{Wind,x} ~ ~ (\\mathrm{m ~ s}^{-1})$')
    ax_wind.set_ylabel('$Z ~ ~ (\\mathrm{m})$')

    ax_profile.set_xlabel('$X (\\mathrm{m})$')
    ax_profile.set_ylabel('$Y (\\mathrm{m})$')
    ax_profile.set_zlabel('$V_\\mathrm{Thermal,V} ~ ~ (\\mathrm{m ~ s}^{-1})$')

    ax_thermalcore.set_xlabel('$X (\\mathrm{m})$')
    ax_thermalcore.set_ylabel('$Y (\\mathrm{m})$')
    ax_thermalcore.set_zlabel('$Z (\\mathrm{m})$')
    ax_thermalcore.set_aspect('equalxy', adjustable='datalim')

    ax_rotation.set_aspect('equal')
    ax_rotation.set_xlabel('$X (\\mathrm{m})$')
    ax_rotation.set_ylabel('$Y (\\mathrm{m})$')
    if cax is not None:
        cbar0 = plt.colorbar(ScalarMappable(norm=kwargs_list[0]['norm'], cmap=kwargs_list[0]['cmap']), cax=cax[0],
                     label='$V_\\mathrm{Wind,x} ~ ~ (\\mathrm{m ~ s}^{-1})$')
        cbar1 = plt.colorbar(ScalarMappable(norm=kwargs_list[1]['norm'], cmap=kwargs_list[1]['cmap']), cax=cax[1],
                     label='$V_\\mathrm{Thermal,V} ~ ~ (\\mathrm{m ~ s}^{-1})$')
        cbar3 = plt.colorbar(ScalarMappable(norm=kwargs_list[3]['norm'], cmap=kwargs_list[3]['cmap']), cax=cax[3],
                     label='$V_\\mathrm{Thermal, H} ~ ~ (\\mathrm{m ~ s}^{-1})$')

        cbar0.outline.set_edgecolor('black')
        cbar1.outline.set_edgecolor('black')
        cbar3.outline.set_edgecolor('black')



