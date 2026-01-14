import os.path

import numpy as np
import pandas as pd


import matplotlib



from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from misc.config import stork_dataset_renaming_dict, science_matplotlib_config

save = False
figsize_multiplier = 2
science_matplotlib_config(figsize_multiplier=figsize_multiplier, save=save)
from matplotlib import pyplot as plt

marker_list = 'o*+xd^v'

from data.get_data import load_decomposition_data, load_synthetic_and_decomposed, load_synthetic_data
pd.options.mode.chained_assignment = None
root_path = '/home/pedro/PycharmProjects/ThermalModelling'
base_path = os.path.join(root_path, f'results/turbulence/turbulence_noise/small_scale/air/min-occ_5_radius_10')
save_folder = base_path
if save:
    os.makedirs(os.path.join(save_folder, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'svg'), exist_ok=True)
list_of_mappable = []
df_all_sigmas = pd.read_csv(os.path.join(base_path, 'turbulence_sigmas.csv'), index_col=False)
df_all_fluctuations = pd.read_csv(os.path.join(base_path, 'fluctuations.csv'), index_col=False)
df_all_sigma_grouped = pd.read_csv(os.path.join(base_path, f'turbulence_sigmas_grouped.csv'), index_col=False)

df_dec_gt_fit = pd.read_csv(os.path.join(base_path, f'dec_vs_gt_linear_fits.csv'), index_col=False)
df_dec_turb_fit = pd.read_csv(os.path.join(base_path, f'dec_vs_turbulence_linear_fits.csv'), index_col=False)

df_all_sigma_grouped_storks = pd.read_csv(os.path.join('results/turbulence/storks/small_scale/air/min-occ_5_radius_10',
                                                f'turbulence_sigmas_grouped.csv'), index_col=False)

# ==================================================================================================================== #
# ==========================================          PLOTTING         =============================================== #
# ==================================================================================================================== #




# df_all_sigmas = df_all_sigmas[df_all_sigmas['noise'] < 0.6]

# df_all_sigma_grouped = df_all_sigma_grouped[df_all_sigma_grouped['noise'] <=1.0]


list_of_noises = df_all_sigma_grouped['noise'].unique()
list_of_turbulences = df_all_sigma_grouped['turbulence'].unique()
list_of_radii = df_all_sigma_grouped['radius'].unique()
list_of_realizations = df_all_sigmas['realization'].unique()

color_list = [f'C{i + 1}' for i in range(len(list_of_noises))]
#fig, ax_all = plt.subplots(len(list_of_radii), 3, figsize=(14,12), sharey='all', sharex='all', layout='constrained')

noise_norm = Normalize(list_of_noises.min(), list_of_noises.max() + 0.1 * np.ptp(list_of_noises))
my_cmap='viridis'
for i_radius, current_radius in enumerate(list_of_radii[:]):
    fig, ax_all = plt.subplots(4,2,figsize=(figsize_multiplier*3,figsize_multiplier*12),
                                     sharey=True,
                                     sharex=True,
                                     layout='constrained',
                                     width_ratios=[1,1])
    ax_dec = ax_all[:,0]
    ax_gt = ax_all[:,1]

    for i_noise, noise in enumerate(list_of_noises):
        # current_sigma = df_all_sigmas[np.all(df_all_sigmas[['radius','noise']] == (current_radius, noise), axis=1)].copy()
        current_sigma_grouped = df_all_sigma_grouped[np.all(df_all_sigma_grouped[['radius','noise']] == (current_radius, noise), axis=1)].copy()
        current_dec_turb_fit = df_dec_turb_fit[np.all(df_dec_turb_fit[['radius','noise']] == (current_radius, noise), axis=1)].copy()
        for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz']):
            current_sigma_grouped_dec = current_sigma_grouped[current_sigma_grouped['datatype'] == 'dec']
            current_sigma_grouped_gt = current_sigma_grouped[current_sigma_grouped['datatype'] == 'gt']
            current_sigma_grouped_dec = current_sigma_grouped_dec.sort_values('turbulence')
            current_sigma_grouped_gt = current_sigma_grouped_gt.sort_values('turbulence')
            a,b = current_dec_turb_fit.loc[current_dec_turb_fit['comp'] == comp, ['a', 'b']].values[0]
            ax_dec[i_comp].errorbar(current_sigma_grouped_dec['turbulence'],
                                    current_sigma_grouped_dec[f'{comp}_median'],
                                    yerr=current_sigma_grouped_dec[f'{comp}_sem'],
                                    c=matplotlib.colormaps[my_cmap](noise_norm(noise)),
                                    fmt=marker_list[i_radius])
            ax_dec[i_comp].plot(current_sigma_grouped_dec['turbulence'], a * current_sigma_grouped_dec['turbulence'] + b,
                                c=matplotlib.colormaps[my_cmap](noise_norm(noise)))
            ax_gt[i_comp].errorbar(current_sigma_grouped_gt['turbulence'],
                                   current_sigma_grouped_gt[f'{comp}_median'],
                                   yerr=current_sigma_grouped_gt[f'{comp}_sem'],
                                   c=matplotlib.colormaps[my_cmap](noise_norm(noise)),
                                   marker=marker_list[i_radius])
            ax_dec[0].set_title(f'Dec')
            ax_gt[0].set_title(f'GT')
    for t in df_all_sigma_grouped_storks['thermal'].unique():
        current_thermal_sigma = df_all_sigma_grouped_storks[df_all_sigma_grouped_storks['thermal'] == t]
        for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz']):
            ax_dec[i_comp].scatter([-0.1896], current_thermal_sigma[f'{comp}_std'], marker='*', s=50,alpha=0.8,
                                   label=stork_dataset_renaming_dict[t] if i_comp == 0 else None)
            ax_gt[i_comp].scatter([-0.1896], current_thermal_sigma[f'{comp}_std'], marker='*', s=50,alpha=0.8,)

    fig.legend(ncol=4, loc='outside upper center',handletextpad=0.6, )
    plt.colorbar(ScalarMappable(cmap=matplotlib.colormaps[my_cmap], norm=noise_norm), ax=ax_gt[:], label='$\\sigma_{noise}$ (m)')

    [ax_dec[i_comp].set_ylabel(f'$ \\sigma_ {{{comp}}}$ (m s $^{{{-1}}}$)') for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz'])]
    ax_dec[-1].set_xlabel('$\\sigma_{Turb}$ (m s $^{-1}$)')
    ax_gt[-1].set_xlabel('$\\sigma_{Turb}$ (m s $^{-1}$)')
    if save:
        fig.savefig(os.path.join(save_folder, 'png', f'turbulence_sweep.png'),transparent=True)
        fig.savefig(os.path.join(save_folder, 'svg', f'turbulence_sweep.svg'))
    else:
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

            for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz']):
                current_fluctuations_dec = current_fluctuations.loc[current_fluctuations['datatype'] == 'dec', comp]
                current_fluctuations_gt = current_fluctuations.loc[current_fluctuations['datatype'] == 'gt', comp]

                current_fluctuations_dec = current_fluctuations_dec[
                    np.abs(current_fluctuations_dec) < 1]
                current_fluctuations_gt = current_fluctuations_gt[
                    np.abs(current_fluctuations_gt) < 1]
                ax_all[i_comp, i_turbulence].hist(current_fluctuations_dec,
                                                  bins=int(np.sqrt(len(current_fluctuations_dec))), density=True,
                                                  alpha=0.5, label=f'DEC proc')
                ax_all[i_comp, i_turbulence].hist(current_fluctuations_gt,
                                                  bins=int(np.sqrt(len(current_fluctuations_gt))), density=True,
                                                  alpha=0.5, label=f'GT proc')



plt.show()


fig_fit,ax_fit = plt.subplots(2,4, sharex='all', sharey='all')
ax_fit = ax_fit.flatten()
list_of_noise = sorted(list_of_noises)
for i_noise, n in enumerate(list_of_noise):
    for i_comp, comp in enumerate(['x', 'z']):
        a,b = df_dec_gt_fit.loc[(df_dec_gt_fit['comp'] == comp)
                              & (df_dec_gt_fit['noise'] == n), ['a','b']].values[0]

        current_merge = pd.merge(df_all_sigma_grouped[(df_all_sigma_grouped['noise'] == n) & (df_all_sigma_grouped['datatype'] == 'dec')],
                                     df_all_sigma_grouped[(df_all_sigma_grouped['noise'] == n) & (df_all_sigma_grouped['datatype'] == 'gt')],
                                     on=['turbulence'], suffixes=('_dec', '_gt'))
        ax_fit[i_noise].errorbar(current_merge[f'{comp}_median_gt'], current_merge[f'{comp}_median_dec'],
                                 xerr=current_merge[f'{comp}_sem_gt'], yerr=current_merge[f'{comp}_sem_dec'],
                                 fmt='o-',
                                 label=f'{comp}',
                                 #c='g' if i_coord else 'b',
                                 )
        ax_fit[i_noise].plot(list_of_turbulences, a * np.array(list_of_turbulences) + b, label=comp)
        ax_fit[i_noise].set_title(f'{n:.2f}')

        ax_fit[i_noise].set_xlabel('$ \\sigma^{GT}_{v_i} \\ (ms^{-1})$')
        ax_fit[i_noise].set_ylabel('$ \\sigma^{DEC}_{v_i} \\ (ms^{-1})$')
        # ax_fit[i_noise].scatter((df_all_sigma_grouped_storks[f'{coord}_std'] - b) / a, df_all_sigma_grouped_storks[f'{coord}_std'], marker='*')
ax_fit[i_noise].legend()

