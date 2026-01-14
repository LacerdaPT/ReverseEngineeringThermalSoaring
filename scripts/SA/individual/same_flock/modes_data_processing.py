import os
from functools import reduce
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yaml
import matplotlib

from simulated_annealing.analysis import get_pearson_correlations

save = True
if save:
    matplotlib.use('Cairo')
    matplotlib.interactive(False)

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, gaussian_kde, PermutationMethod
from sklearn.neighbors import KernelDensity

from calc.stats import get_mode_with_KDE, get_weighted_average
from data.get_data import load_synthetic_and_decomposed, load_synthetic_data
from plotting.sim_annealing import plot_kde_modes_histograms
from simulated_annealing.post_process import post_process_annealing_results

# list_of_thermals = ['b010_0.1', 'b023_0.1', 'b023_1.1',
#                     'b072_0.1', 'b077_0.1',
#     'b112_0.2', 'b121_0.1'
# ]

append=True
list_of_birds = None
loss_percentile_list = [4] #
n_resamples = 1000
candidate = True
do_correlations = True
do_deviation = True
kernel_bandwidth = None
config_root = 'synthetic_data/from_atlasz/last/same_flock_WL=6.0/config/individual_bins/bin_z_size=10'
save_folder = os.path.join(config_root, f'permutation_{n_resamples=}')
p = Path(config_root)
# config_file_dict = {(a.parent.parent.name, a.parent.name): str(a)
#                     for a in p.glob('sa_individ*.yaml')}
config_file_dict = {('012', 0): os.path.join(config_root, '012', 'sa.yaml'),
                    ('345', 0): os.path.join(config_root, '345', 'sa.yaml'),
                    ('all', 0): os.path.join(config_root, 'all', 'sa.yaml'),
                    ('0', 0): os.path.join(config_root, '0', 'sa.yaml'),
                    ('1', 0): os.path.join(config_root, '1', 'sa.yaml'),
                    ('2', 0): os.path.join(config_root, '2', 'sa.yaml'),
                    ('3', 0): os.path.join(config_root, '3', 'sa.yaml'),
                    ('4', 0): os.path.join(config_root, '4', 'sa.yaml'),
                    ('5', 0): os.path.join(config_root, '5', 'sa.yaml')
                    }

if save:
    os.makedirs(save_folder, exist_ok=True)
synthetic_data_dict = load_synthetic_data('synthetic_data/from_atlasz/last/same_flock_WL=6.0/0')
bird_parameters = synthetic_data_dict['bird_parameters']
del synthetic_data_dict
bird_parameters = [[d['bird_name'], d['physical_parameters']['mass'] / d['physical_parameters']['wing_area']] for d in
                   bird_parameters]
if list_of_birds is None:
    list_of_birds = list(map(lambda x: x[0], bird_parameters))
df_bird_parameters = pd.DataFrame(bird_parameters, columns=['bird_name', 'WL_real'])
if append:
    df_corr_all_percentiles = pd.read_csv(os.path.join(save_folder,'pearson_correlations_all_percentiles.csv'), index_col=False)
    df_corr_real_all_percentiles = pd.read_csv(os.path.join(save_folder,'pearson_correlations_real_all_percentiles.csv'), index_col=False)
    df_modes_all_percentiles = pd.read_csv(os.path.join(save_folder,'modes_all_percentiles.csv'), index_col=False)
    df_modes_avg_all_percentiles = pd.read_csv(os.path.join(save_folder,'modes_average_all_percentiles.csv'), index_col=False)
else:
    df_corr_all_percentiles = pd.DataFrame()
    df_corr_real_all_percentiles = pd.DataFrame()
    df_modes_all_percentiles = pd.DataFrame()
    df_modes_avg_all_percentiles = pd.DataFrame()
for loss_percentile in loss_percentile_list:
    print(loss_percentile)

    df_modes = pd.DataFrame()
    df_history = pd.DataFrame()
    for i_thermal, ((current_thermal, realization), config_file) in enumerate(config_file_dict.items()):

        if not os.path.exists(config_file):
            continue
        print(config_file)
        try:
            df_history, (current_df_modes, kde_dict) = post_process_annealing_results(config_file,
                                                                                      loss_percentile,
                                                                                      candidate=True,
                                                                                      do_deviation=do_deviation,
                                                                                      n_resamples=n_resamples,
                                                                                      kernel_bandwidth=kernel_bandwidth)
        except FileNotFoundError as e:
            print(e)
            continue
        parameter_cols = df_history.columns[:-1]
        loss_col = df_history.columns[-1]

        current_df_modes = pd.merge(current_df_modes, df_bird_parameters, on='bird_name', how='left')

        current_df_modes['delta_WL'] = current_df_modes['WL_mode'] - current_df_modes['WL_real']
        current_df_modes['config_file'] = config_file
        current_df_modes['thermal'] = current_thermal

        current_df_modes['realization'] = realization

        df_modes = pd.concat([df_modes, current_df_modes])

    if do_deviation:
        df_modes_avg = pd.DataFrame(columns=['WL_mode_mean', 'WL_mode_std', 'WL_mode_sem', 'WL_mode_count'],
                                    index=list_of_birds)
        aa = df_modes.groupby('bird_name').apply(
            lambda row: get_weighted_average(row['WL_mode'].values,
                                             1 / row['WL_mode_std'].values) if row['WL_mode_std'].values is not None else np.nan,
            include_groups=False)

        for i_col, col in enumerate(['WL_mode_mean', 'WL_mode_std', 'WL_mode_sem', 'WL_mode_count']):
            df_modes_avg[col] = aa.apply(lambda x: x[i_col])
    else:
        df_modes_avg = pd.DataFrame(columns=['WL_mode_mean', 'WL_mode_std', 'WL_mode_sem', 'WL_mode_count'],
                                    index=list_of_birds)
        aa = df_modes.groupby('bird_name').apply(lambda row: np.nanmean(row['WL_mode']), include_groups=False)
        for i_col, col in enumerate(['WL_mode_mean', 'WL_mode_std', 'WL_mode_sem', 'WL_mode_count']):
            df_modes_avg[col] = aa.apply(lambda x: x[i_col])
    df_modes_avg.reset_index(names='bird_name', inplace=True)
    if do_correlations:
        # ================================================================================================================ #
        # ======================================         CORRELATIONS         ============================================ #
        # ================================================================================================================ #
        list_of_thermal = np.sort(df_modes['thermal'].unique())

        df_corr = get_pearson_correlations(df_modes,
                                           'bird_name',
                                           'WL_mode',
                                           'thermal',
                                           method=PermutationMethod(n_resamples=9999, batch=9999 ))

        df_corr_real = []
        for i_tt, t1 in enumerate(list_of_thermal):
            df_t1 = df_modes[df_modes['thermal'] == t1].copy()

            current_corr = stats.pearsonr(df_t1['WL_mode'],
                                          df_t1['WL_real'],
                              method=PermutationMethod(n_resamples=9999, batch=9999))
            df_corr_real.append({f'thermal': t1,
                                 'pearson_r':  current_corr.statistic,
                                 'pvalue':     current_corr.pvalue,
                                 'CI_low':     current_corr.confidence_interval(0.95).low,
                                 'CI_high':    current_corr.confidence_interval(0.95).high
                                 })

        df_corr_real = pd.DataFrame.from_dict(df_corr_real, orient='columns')
        df_corr['loss_percentile'] = loss_percentile
        df_corr_real['loss_percentile'] = loss_percentile
        df_corr_all_percentiles = pd.concat([df_corr_all_percentiles, df_corr])
        df_corr_real_all_percentiles = pd.concat([df_corr_real_all_percentiles, df_corr_real])
        if save:
            df_corr.to_csv(os.path.join(save_folder, f'pearson_correlations_{loss_percentile}.csv'))
            df_corr_real.to_csv(os.path.join(save_folder, f'pearson_correlations_real_{loss_percentile}.csv'))
    df_modes['loss_percentile'] = loss_percentile
    df_modes_avg['loss_percentile'] = loss_percentile
    if save:
        df_modes.to_csv(os.path.join(save_folder, f'modes_{loss_percentile}.csv'))
        df_modes_avg.to_csv(os.path.join(save_folder, f'modes_average_{loss_percentile}.csv'))

    df_modes_all_percentiles = pd.concat([df_modes_all_percentiles, df_modes])
    df_modes_avg_all_percentiles = pd.concat([df_modes_avg_all_percentiles, df_modes_avg])

    if save:
        if do_correlations:
            df_corr_all_percentiles.to_csv(os.path.join(save_folder, f'pearson_correlations_all_percentiles.csv'), index=False)
            df_corr_real_all_percentiles.to_csv(os.path.join(save_folder, f'pearson_correlations_real_all_percentiles.csv'), index=False)
        df_modes_all_percentiles.to_csv(os.path.join(save_folder, f'modes_all_percentiles.csv'), index=False)
        df_modes_avg_all_percentiles.to_csv(os.path.join(save_folder, f'modes_average_all_percentiles.csv'), index=False)
