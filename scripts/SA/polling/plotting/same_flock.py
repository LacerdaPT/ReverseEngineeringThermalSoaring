import os.path
import numpy as np
import pandas as pd

from misc.config import science_matplotlib_config

save = True
figsize_multiplier = 1

science_matplotlib_config(figsize_multiplier=figsize_multiplier, save=save)

from matplotlib import pyplot as plt
root_path = '/home/pedro/ThermalModelling'
path_to_modes = os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/same_flock_WL=6.0/config/individual_bins/bin_z_size=10/n_resamples=1000')
n_resamples = 9999
df_modes = pd.read_csv(os.path.join(path_to_modes, 'modes_all_percentiles.csv'),
                       index_col=False)
path_to_csv =os.path.join(root_path, 'results/wing_loadings/pooling', 'same_flock_WL=6.0')
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
                       layout='constrained', sharey='all', sharex='all')
ax_corr = ax_corr.flatten()

for i_c, (_, row) in enumerate(df_pearson.iterrows()):
    c1_str = row['c1']
    c2_str = row['c2']
    pearson = row['pearson_r']
    pvalue = row['p_value']
    c1 = [a for a in c1_str]
    c2 = [a for a in c2_str]
    df_1 = df_modes_2[df_modes_2['thermal'].astype(str).isin(c1)].groupby(['bird_name', 'loss_percentile', 'WL_real'
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    df_1.rename(columns={0: 'WL_mode_avg'}, inplace=True)
    df_2 = df_modes_2[df_modes_2['thermal'].astype(str).isin(c2)].groupby(['bird_name', 'loss_percentile', 'WL_real'
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    df_2.rename(columns={0: 'WL_mode_avg'}, inplace=True)
    df_merge = pd.merge(df_1, df_2, on=['bird_name', 'loss_percentile', 'WL_real'],
                          suffixes=('_1', '_2'))

    current_ax_corr = ax_corr[i_c]
    wl_min, wl_max = df_merge[['WL_mode_avg_1','WL_mode_avg_2']].values.min(),  df_merge[['WL_mode_avg_1','WL_mode_avg_2']].values.max()
    current_ax_corr.scatter(df_merge['WL_mode_avg_1'].values, df_merge['WL_mode_avg_2'].values)
    current_ax_corr.plot([wl_min, wl_max], [wl_min, wl_max], c='r', ls='--')
    current_ax_corr.set_xlabel(f'$W^L_\\mathrm{{{c1_str}}}$ (kg m$^{{-2}}$)')
    current_ax_corr.set_ylabel(f'$W^L_\\mathrm{{{c2_str}}}$ (kg m$^{{-2}}$)')
    # current_ax_corr.set_title(f'{pearson:.3f}, p={pvalue:.3f}')
    current_ax_corr.set_xlim((4.8,8.2))
    current_ax_corr.set_ylim((4.8,8.2))
    current_ax_corr.set_yticks(np.arange(5,8+1), minor=False)
    current_ax_corr.set_aspect('equal')
    current_ax_corr.spines[['right', 'top']].set_visible(False)
    visible_ticks = {
        "top": False,
        "right": False
    }
    current_ax_corr.tick_params(axis="x", which="both", **visible_ticks)
    current_ax_corr.tick_params(axis="y", which="both", **visible_ticks)


if save:
    plt.draw()
    plt.draw_all()
    fig_correlations.savefig(os.path.join(path_to_save, 'png', f'3-combinations-correlations.png'))
    fig_correlations.savefig(os.path.join(path_to_save, 'svg', f'3-combinations-correlations.svg'))
    plt.close(fig_correlations)
else:
    plt.show(block=False)

df_metrics = pd.read_csv(os.path.join(path_to_csv, 'pooling_metrics.csv'), index_col=False)


my_dfs = {'singles': df_modes[df_modes['thermal'].astype(str).isin(['0', '1', '2', '3', '4', '5'])],
          'triples': df_modes[df_modes['thermal'].astype(str).isin(['012', '345'])],
          'all':     df_modes[df_modes['thermal'].astype(str).isin(['all'])],
          'all_avg': df_modes.copy(), }




my_dfs_avg ={}
for k, df in my_dfs.items():
    my_dfs_avg[k] = df.groupby(['bird_name', 'loss_percentile', 'WL_real'
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    my_dfs_avg[k].rename(columns={0: 'WL_mode_avg'}, inplace=True)
    my_dfs_avg[k]['delta'] = my_dfs_avg[k]['WL_mode_avg'] - my_dfs_avg[k]['WL_real']


fig_violin, ax_arr_violin = plt.subplots(1,4, sharey='all', figsize=(7.25 * figsize_multiplier,
                                                                     1 / 4 * 7.25 * figsize_multiplier),
                                         layout='tight')
ax_dict = {k: a for k,a in zip(my_dfs.keys() , ax_arr_violin.flatten())}

for k, current_ax in ax_dict.items():
    current_avgs = my_dfs_avg[k]
    for i in range(1,5):
        current_avg_percentile = current_avgs[current_avgs['loss_percentile'] == i]

        parts = current_ax.violinplot(current_avg_percentile['delta'].values, [i] , showmeans=True, showmedians=True)
        parts['cmedians'].set_linestyle('--')
        current_ax.axhline(y=0, ls=':', c='k')
        current_ax.set_title(k)
        current_ax.set_xlabel('loss percentile')
ax_dict['singles'].set_ylabel('$\\Delta W^L$ (kg m$^{{-2}}$)')

if save:
    plt.draw()
    plt.draw_all()
    fig_violin.savefig(os.path.join(path_to_save, 'png',f'delta_WL_pooling_violin.png'))
    fig_violin.savefig(os.path.join(path_to_save, 'svg',f'delta_WL_pooling_violin.svg'))
    plt.close(fig_violin)
else:
    plt.show(block=False)