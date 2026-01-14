import os.path

import numpy as np
import pandas as pd

import matplotlib


from misc.config import science_matplotlib_config

save = True

figsize_multiplier = 2
science_matplotlib_config(figsize_multiplier, save=save)

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, CenteredNorm, TwoSlopeNorm

marker_list = 'o*+xd^v'

pd.options.mode.chained_assignment = None
root_path = '/home/pedro/PycharmProjects/ThermalModelling'
base_url = os.path.join(root_path, f'results/turbulence/storks/small_scale/air')


# ==================================================================================================================== #
# ==========================================          PLOTTING         =============================================== #
# ==================================================================================================================== #

list_of_mappable = []
df_all_sigmas = pd.read_csv(f'{base_url}/turbulence_sigmas.csv', index_col=False)
df_all_fluctuations = pd.read_csv(f'{base_url}/fluctuations.csv', index_col=False)
# df_all_spheres = pd.read_csv(f'{base_url}/stats_per_radius_per_sphere.csv', index_col=False)
#
# df_new_spheres = df_all_fluctuations.groupby(['n', 'radius', 'center_x', 'center_y', 'center_z',
#        'noise', 'turbulence', 'realization', 'datatype']).agg(avg_x=('x', 'mean'),
#                                                               avg_y=('y', 'mean'),
#                                                               avg_z=('z', 'mean'),
#                                                               avg_xyz=('xyz', 'mean'),
#                                                               sigma_x=('x', 'std'),
#                                                               sigma_y=('y', 'std'),
#                                                               sigma_z=('z', 'std'),
#                                                               sigma_xyz=('xyz', 'std'),
#                                                               median_x=('x', 'median'),
#                                                               median_y=('y', 'median'),
#                                                               median_z=('z', 'median'),
#                                                               median_xyz=('xyz', 'median'),).reset_index()



my_norm = TwoSlopeNorm(0, vmin=-6, vmax=6)
my_cmap = matplotlib.colormaps['seismic']
my_norm = Normalize(vmin=0, vmax=1.5)
my_cmap = matplotlib.colormaps['gnuplot']
# df_all_sigmas = df_all_sigmas[df_all_sigmas['noise'] < 0.6]
df_all_sigma_grouped = df_all_sigmas.groupby(['noise', 'turbulence', 'radius', 'datatype']
                                             ).agg(x_mean=('x', 'mean'),
                                                   y_mean=('y', 'mean'),
                                                   z_mean=('z', 'mean'),
                                                   xyz_mean=('xyz', 'mean'),
                                                   x_std=('x', 'std'),
                                                   y_std=('y', 'std'),
                                                   z_std=('z', 'std'),
                                                   xyz_std=('xyz', 'std'),
                                                   x_median=('x', 'median'),
                                                   y_median=('y', 'median'),
                                                   z_median=('z', 'median'),
                                                   xyz_median=('xyz', 'median'),
                                                   x_count=('x', 'count'),
                                                   y_count=('y', 'count'),
                                                   z_count=('z', 'count'),
                                                   xyz_count=('xyz', 'count'),
                                                   ).reset_index()
df_all_sigma_grouped = df_all_sigma_grouped[df_all_sigma_grouped['noise'] <=1.0]


list_of_noises = df_all_sigma_grouped['noise'].unique()
list_of_turbulences = df_all_sigma_grouped['turbulence'].unique()
list_of_radii = df_all_sigma_grouped['radius'].unique()
list_of_realizations = df_all_sigmas['realization'].unique()

color_list = [f'C{i + 1}' for i in range(len(list_of_noises))]
#fig, ax_all = plt.subplots(len(list_of_radii), 3, figsize=(14,12), sharey='all', sharex='all', layout='constrained')
my_cmap = 'gnuplot'
my_norm = Normalize(0, 2)
my_levels = 30
fig, ax_array = plt.subplots(4,4,figsize=(14,14),layout='constrained',
                                               #sharey=True,
                                               #sharex=True,

                             )

for i_radius, current_radius in enumerate(list_of_radii[:]):
    current_sigma_grouped = df_all_sigma_grouped[df_all_sigma_grouped['radius'] == current_radius].copy()
    for i_coord, coord in enumerate( ['x', 'y', 'z', 'xyz']):
        current_ax = ax_array[i_coord,i_radius]

        m=current_ax.tricontourf(current_sigma_grouped['turbulence']*0.632,
                                 current_sigma_grouped['noise'] * np.sqrt(3),
                                 current_sigma_grouped[f'{coord}_median'],
                               cmap=my_cmap, norm=my_norm, levels=my_levels)

        ax_array[i_coord,0].set_ylabel('noise')
        if i_radius == 0:
            plt.colorbar(ScalarMappable(cmap=my_cmap, norm=my_norm), ax=ax_array[i_coord,-1],label=f'$\\delta v_{coord}\\ (ms^{{-1}}$')
    ax_array[-1,i_radius].set_xlabel('turbulence')
    ax_array[0, i_radius].set_title(f'R={current_radius}')


for i_radius, current_radius in enumerate(list_of_radii[:]):
    fig, ax_all = plt.subplots(4,2,figsize=(14,6),
                                     sharey=True,
                                     sharex=True,
                                     layout='constrained',
                                     width_ratios=[1,1])
    fig.suptitle(f'radius={current_radius}')
    ax_dec = ax_all[:,0]
    ax_gt = ax_all[:,1]

    #ax_all = np.array(list(ax_all.values())).reshape(2,2)
    for i_noise, noise in enumerate(list_of_noises):
            #ax_row = ax_all[i_radius]
            current_sigma = df_all_sigmas[np.all(df_all_sigmas[['radius','noise']] == (current_radius, noise), axis=1)].copy()
            current_sigma_grouped = df_all_sigma_grouped[np.all(df_all_sigma_grouped[['radius','noise']] == (current_radius, noise), axis=1)].copy()
            for i_coord, coord in enumerate( ['x', 'y', 'z', 'xyz']):
                current_sigma_grouped_dec = current_sigma_grouped[current_sigma_grouped['datatype'] == 'dec']
                current_sigma_grouped_gt = current_sigma_grouped[current_sigma_grouped['datatype'] == 'gt']
                ax_dec[i_coord].scatter(current_sigma_grouped_dec['turbulence'],
                                      current_sigma_grouped_dec[f'{coord}_median'],
                                      label=f"noise={current_sigma_grouped_dec['noise'].unique()[0]}, R={current_radius}",
                                      c=color_list[i_noise],
                                      marker=marker_list[i_radius])
                ax_gt[i_coord].scatter(current_sigma_grouped_gt['turbulence'],
                                      current_sigma_grouped_gt[f'{coord}_median'],
                                      label=f"noise={current_sigma_grouped_gt['noise'].unique()[0]}, R={current_radius}",
                                      c=color_list[i_noise],
                                      marker=marker_list[i_radius])
                ax_dec[i_coord].set_ylabel(f'$ \\sigma_ {{{coord}}}$')
                ax_dec[0].set_title(f'Dec')
                ax_gt[0].set_title(f'GT')
            # for i_coord, coord in enumerate( ['x', 'y', 'z', 'xyz']):
            #     #ax_row[i_coord].scatter(current_sigma['turbulence'], current_sigma[coord], label=f"noise={current_sigma['noise'].unique()[0]}")
            #     ax_all[coord].scatter(current_sigma_grouped['turbulence'],
            #                           current_sigma_grouped[f'{coord}_GT_median'],
            #                           label=f"noise={current_sigma_grouped['noise'].unique()[0]}, R={current_radius}, GT",
            #                           c=color_list[i_noise],
            #                           marker=marker_list[i_radius])
                # m = ax_row[i_coord].tricontourf(current_sigma_grouped['turbulence'],
                #                             current_sigma_grouped[f'noise'],
                #                             current_sigma_grouped[f'{coord}_mean'], # - current_sigma_grouped['turbulence'],
                #                             norm=my_norm,
                #                                 cmap=my_cmap,
                #                                 levels=np.linspace(my_norm.vmin, my_norm.vmax, my_levels, endpoint=True)
                #                                 )
                # ax[i_coord].scatter([float(t.split('=')[-1]) * 0.6314318 for t in sigmas_real_dict[coord].keys()], sigmas_dict[coord].values())
                # ax[i_coord].scatter([float(t.split('=')[-1]) * 0.6314318 for t in sigmas_real_dict[coord].keys()], sigmas_real_dict[coord].values())

        #ax_row[i_coord].set_ylabel('$noise\\ \\ (m)$')
        #ax_row[i_coord].set_title(coord.upper())
    # [ax_all[0, i_coord].set_title(coord.upper() )for i_coord, coord in enumerate( ['x', 'y', 'z', 'xyz'])]
    #
    # [a.set_ylabel('$\\sigma \\ \\ (m/s)$') for a in ax_all[:, 0]]
    # [a.set_xlabel('$turbulence\\ \\ (m s^{{-1}})$') for a in ax_all[-1,:]]
    # [ax_all[i_r,0].set_ylabel(f'radius={r}' + '\n' + ax_all[i_r,0].get_ylabel()) for i_r, r in enumerate(list_of_radii)]
    #     #list_of_mappable.append(m)
    #     #[a.set_ylim((0,None)) for a in ax]
    #
    #
    ax_gt[1].legend()


plt.show()

current_noise = list_of_noises[0]
for i_radius, current_radius in enumerate(list_of_radii[:]):
    fig, ax_all = plt.subplots(4,len(list_of_turbulences),figsize=(14,6),
                                     # sharey='row',
                                     # sharex='row',
                                     layout='constrained')
    fig.suptitle(f'radius={current_radius}')


    #ax_all = np.array(list(ax_all.values())).reshape(2,2)
    for i_turbulence, turbulence in enumerate(list_of_turbulences):
            #ax_row = ax_all[i_radius]
            current_fluctuations = df_all_fluctuations[np.all(df_all_fluctuations[['radius','turbulence', 'noise']] == (current_radius, turbulence, current_noise), axis=1)].copy()

            for i_coord, coord in enumerate( ['x', 'y', 'z', 'xyz']):
                current_fluctuations_dec = current_fluctuations.loc[current_fluctuations['datatype'] == 'dec', coord]
                current_fluctuations_gt = current_fluctuations.loc[current_fluctuations['datatype'] == 'gt', coord]

                current_fluctuations_dec = current_fluctuations_dec[
                    np.abs(current_fluctuations_dec) < 1]
                current_fluctuations_gt = current_fluctuations_gt[
                    np.abs(current_fluctuations_gt) < 1]
                ax_all[i_coord, i_turbulence].hist(current_fluctuations_dec,
                                           bins=int(np.sqrt(len(current_fluctuations_dec))), density=True,
                                           alpha=0.5, label=f'DEC proc')
                ax_all[i_coord, i_turbulence].hist(current_fluctuations_gt,
                                           bins=int(np.sqrt(len(current_fluctuations_gt))), density=True,
                                           alpha=0.5, label=f'GT proc')



plt.show()