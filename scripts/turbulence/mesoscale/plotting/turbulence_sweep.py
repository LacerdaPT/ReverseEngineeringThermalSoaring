import os.path
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import scienceplots
import matplotlib



import yaml
from scipy import signal
from scipy.optimize import curve_fit

from calc.analysis.turbulence import get_single_sweep_autocorrelation
from misc.config import science_matplotlib_config
from misc.constants import root_path
from object.air import ReconstructedAirVelocityField


save = False

figsize_multiplier = 1
science_matplotlib_config(figsize_multiplier=figsize_multiplier, save=save)

from matplotlib import pyplot as plt

#
color_list = [f'C{i + 1}' for i in range(10)]

p = Path(os.path.join(root_path, 'results/turbulence/turbulence_noise/mesoscale_test/air'))
path_to_csv = str(p)
save_folder = os.path.join(path_to_csv, 'figures')

if save:
    os.makedirs(os.path.join(save_folder, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'svg'), exist_ok=True)

df_all_correlations = pd.read_csv(os.path.join(path_to_csv, 'turbulence_correlation_grid_size=2.csv'), index_col=False)
df_all_correlations_avg_per_realization = pd.read_csv(os.path.join(path_to_csv, 'turbulence_correlation_grid_size=2_avg_per_realization.csv'), index_col=False)

df_all_correlations['turbulence'] *= 0.6314318
df_all_correlations_avg_per_realization['turbulence'] *= 0.6314318
df_all_correlations = df_all_correlations[df_all_correlations['noise'] == 0.0]
df_all_correlations_avg_per_realization = df_all_correlations_avg_per_realization[df_all_correlations_avg_per_realization['noise'] == 0.0]

df_all_correlations_avg = df_all_correlations_avg_per_realization.groupby(['noise' ,'turbulence'] + ['datatype', 'size', 'delta_R']
                                                  ).agg(**{f'{col}_{stat}': (f'{col}_{stat}', stat)
                                                           for col in ['inner' ,'x' ,'y' ,'z']
                                                           for stat in ['mean', 'median', 'std', 'count']
                                                           }).reset_index()

for i_comp, comp in enumerate(['inner', 'x', 'y', 'z']):
    df_all_correlations_avg[f'{comp}_sem'] = df_all_correlations_avg[f'{comp}_std'].values / np.sqrt(df_all_correlations_avg[f'{comp}_count'].values)

list_of_turbulence = df_all_correlations['turbulence'].sort_values().unique()
# list_of_turbulence = list_of_turbulence[::2]
list_of_sizes = df_all_correlations['size'].unique()[:1]


dict_of_popt = []
for i_thermal, thermal in enumerate(list_of_turbulence):
    for i_size, size in enumerate(list_of_sizes):
        for i_comp, comp in enumerate(['inner',
                                       'x',
                                       'y',
                                       'z']):
            for i_datatype, datatype in enumerate(['dec','gt']):
                current_corr = df_all_correlations_avg[np.all(df_all_correlations_avg[['turbulence','size', 'datatype']] == (thermal, size, datatype), axis=1)]
                current_corr = current_corr[(current_corr['delta_R'] >= 10)
                #                            & (current_corr[f'{comp}_mean'] > 0)
                ]

                fit_result = curve_fit(lambda x, l, b: b * x ** l,
                                       current_corr['delta_R'].values,
                                       current_corr[f'{comp}_mean'].values,
                                       p0=(0.01,1),
                                       sigma=current_corr[f'{comp}_sem'])
                popt, pcov = fit_result
                dict_of_popt.append({'turbulence': thermal,'size': size, 'comp': comp, 'datatype':datatype,
                                     'l': popt[0], 'b': popt[1],
                                     'sigma_l': np.sqrt(np.diag(pcov))[0],
                                     'sigma_b': np.sqrt(np.diag(pcov))[1]})

df_popt = pd.DataFrame.from_records(dict_of_popt)

fig, ax = plt.subplots(2, len(list_of_turbulence) + 1, figsize=(figsize_multiplier*7.25, figsize_multiplier*7.25),
                       layout='constrained',
                       sharex='all',
                       sharey='row'
                       )

# fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
for i_thermal, thermal in enumerate(list_of_turbulence):
    for i_size, size in enumerate(list_of_sizes):
        for i_comp, comp in enumerate(['inner',
                                       'x',
                                       #'y',
                                       #'z'
                                       ]):
            for i_datatype, datatype in enumerate(['dec','gt']):
                current_corr = df_all_correlations_avg[np.all(df_all_correlations_avg[['turbulence','size', 'datatype']] == (thermal, size, datatype), axis=1)]
                current_corr = current_corr[current_corr['delta_R'] > 2]
                # for realization in range(10):
                #     current_corr_realization = df_all_correlations_avg_per_realization[
                #         np.all(df_all_correlations_avg_per_realization[['turbulence','size', 'datatype', 'realization']] == (thermal, size, datatype, realization), axis=1)]
                #
                #     ax[i_comp, i_thermal].plot(current_corr_realization['delta_R'].values, current_corr_realization[f'{comp}_mean'].values,
                #                                c='g' if i_datatype else 'b', alpha=0.2)
                ax[i_comp, i_thermal].errorbar(current_corr['delta_R'].values,
                                               current_corr[f'{comp}_mean'].values,
                                               current_corr[f'{comp}_sem'].values,
                                               color='g' if i_datatype else 'b',
                                               label=f'{datatype}')
                if i_thermal == 0:
                    ax[i_comp, i_thermal].set_ylabel(comp)
            old_lims_top = ax[0, i_thermal].get_ylim()
            old_lims_bottom = ax[1, i_thermal].get_ylim()

            # hide the spines between ax and ax2
            ax[0, i_thermal].spines.bottom.set_visible(False)
            ax[1, i_thermal].spines.top.set_visible(False)
            ax[0, i_thermal].xaxis.tick_top()
            ax[0, i_thermal].tick_params(labeltop=False)  # don't put tick labels at the top
            ax[1, i_thermal].xaxis.tick_bottom()

            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax[0, i_thermal].plot([0, 1], [0, 0], transform=ax[0, i_thermal].transAxes, **kwargs)
            ax[1, i_thermal].plot([0, 1], [1, 1], transform=ax[1, i_thermal].transAxes, **kwargs)

            #ax[i_comp, i_thermal].axhline(y=0, c='k', ls=':', alpha=0.3 )

    # ax[0, i_thermal].set_title(f'$\\sigma_{{Turb}} = {turbulence_lookup[thermal]}$')

    ax[0, 0].legend()
    ax[0, i_thermal].set_title(f'$\\sigma _{{Turb}} = {thermal:.2f}\\ ms^{{-1}}$')
    ax[-1, i_thermal].set_xlabel('$\\Delta R\\ (m)$')

ax[0, -1].set_ylim(0.961, 1.01)
ax[1, -1].set_ylim(0.09, 0.21)
plt.tight_layout()
fig.subplots_adjust(hspace=0.15)  # adjust space between Axes

if save:
    fig.canvas.draw()
    fig.savefig(os.path.join(save_folder, 'png', f'turbulence_correlation.png'), dpi=300)
    fig.savefig(os.path.join(save_folder, 'svg', f'turbulence_correlation.svg'))
    plt.close(fig)
else:
    plt.show(block=True)
