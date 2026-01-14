import os.path
from functools import reduce
from itertools import combinations

import matplotlib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, PermutationMethod

from misc.config import science_matplotlib_config
from plotting.auxiliar import simple_axis

save = True
figsize_multiplier = 1

science_matplotlib_config(figsize_multiplier=figsize_multiplier, save=save)
from matplotlib import pyplot as plt
root_path = '/home/pedro/ThermalModelling'
path_to_modes = os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/storks/config/individual_bins/bin_z_size=10/n_resamples=1000')
n_resamples = 9999
df_modes = pd.read_csv(os.path.join(path_to_modes, 'modes.csv'),
                       index_col=False)
df_modes['thermal'] = df_modes['thermal'].replace('b010', '0')
df_modes['thermal'] = df_modes['thermal'].replace('b0230', '1')
df_modes['thermal'] = df_modes['thermal'].replace('b0230', '1.1')
df_modes['thermal'] = df_modes['thermal'].replace('b072', '2')
df_modes['thermal'] = df_modes['thermal'].replace('b077', '3')
df_modes['thermal'] = df_modes['thermal'].replace('b112', '4')
df_modes['thermal'] = df_modes['thermal'].replace('b121', '5')
df_modes['thermal'] = df_modes['thermal'].replace('b010_b023_b072_10', '012')
df_modes['thermal'] = df_modes['thermal'].replace('b077_b112_b121_10', '345')
path_to_csv =os.path.join(root_path, 'results/wing_loadings/pooling', 'storks')
path_to_save = os.path.join(path_to_csv, 'figures')
if save:
    os.makedirs(os.path.join(path_to_save, 'png'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, 'svg'), exist_ok=True)
df_pearson = pd.read_csv(os.path.join(path_to_csv, 'all_3-combinations.csv'), index_col=False,
                         dtype={'loss_percentile': int, 'c1': str, 'c2': str,
                                'pearson_r': float, 'p_value': float, 'n': int, })



df_pearson = df_pearson[df_pearson['loss_percentile'] == 2]
df_modes_2 = df_modes[df_modes['loss_percentile'] == 2]

fig_correlations, ax_corr = plt.subplots(2,5, figsize=(figsize_multiplier * 7.25, 2/5*figsize_multiplier * 7.25),
                       layout='constrained', sharey='all', sharex='all'
                                         )
ax_corr = ax_corr.flatten()

for i_c, (_, row) in enumerate(df_pearson.iterrows()):
    c1_str = row['c1']
    c2_str = row['c2']
    pearson = row['pearson_r']
    pvalue = row['p_value']
    c1 = [a for a in c1_str]
    c2 = [a for a in c2_str]
    df_1 = df_modes_2[df_modes_2['thermal'].astype(str).isin(c1)].groupby(['bird_name', 'loss_percentile'
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    df_1.rename(columns={0: 'WL_mode_avg'}, inplace=True)
    df_2 = df_modes_2[df_modes_2['thermal'].astype(str).isin(c2)].groupby(['bird_name', 'loss_percentile'
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    df_2.rename(columns={0: 'WL_mode_avg'}, inplace=True)
    df_merge = pd.merge(df_1, df_2, on=['bird_name', 'loss_percentile'],
                          suffixes=('_1', '_2'))

    current_ax_corr = ax_corr[i_c]
    wl_min, wl_max = df_merge[['WL_mode_avg_1','WL_mode_avg_2']].values.min(),  df_merge[['WL_mode_avg_1','WL_mode_avg_2']].values.max()
    current_ax_corr.scatter(df_merge['WL_mode_avg_1'].values, df_merge['WL_mode_avg_2'].values)
    current_ax_corr.plot([wl_min, wl_max], [wl_min, wl_max], c='r', ls='--')
    current_ax_corr.set_xlabel(f'$W^L_\\mathrm{{{c1_str}}}$ (kg m$^{{-2}}$)')
    current_ax_corr.set_ylabel(f'$W^L_\\mathrm{{{c2_str}}}$ (kg m$^{{-2}}$)')
    # current_ax_corr.set_title(f'{pearson:.3f}, p={pvalue:.3f}')
    current_ax_corr.set_xlim((5.2,8.1))
    current_ax_corr.set_ylim((5.2,8.1))
    current_ax_corr.set_yticks(np.arange(6,8+1), minor=False)
    current_ax_corr.set_aspect('equal')
    simple_axis(current_ax_corr)




if save:
    plt.draw()
    plt.draw_all()
    fig_correlations.savefig(os.path.join(path_to_save, 'png', f'3-combinations-correlations.png'))
    fig_correlations.savefig(os.path.join(path_to_save, 'svg', f'3-combinations-correlations.svg'))
    plt.close(fig_correlations)
else:
    plt.show(block=False)

