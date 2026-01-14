import os

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import curve_fit
from scipy.stats import PermutationMethod
from sklearn.neighbors import KernelDensity

from calc.stats import get_mode_with_KDE, get_mode_with_deviation_with_KDE, get_weighted_average
from simulated_annealing.analysis import get_pearson_correlations



def get_modes_from_multiple_sim_annealing_history(config_file_dict, list_of_percentiles, aggregated_save_folder, n_resamples,
                                                  list_of_birds, save=True, do_deviation=True, kernel_bandwidth=None, append=False):
    if append:
        df_modes_all_percentiles = pd.read_csv(os.path.join(aggregated_save_folder, f'modes_all_percentiles.csv'),
                                               index_col=False)
        df_modes_avg_all_percentiles = pd.read_csv(
            os.path.join(aggregated_save_folder, f'modes_average_all_percentiles.csv'), index_col=False)
    else:
        df_modes_all_percentiles = pd.DataFrame()
        df_modes_avg_all_percentiles = pd.DataFrame()
    for loss_percentile in list_of_percentiles:  #
        print(loss_percentile)

        df_modes = pd.DataFrame()
        for i_thermal, ((current_thermal, subindex), config_file) in enumerate(config_file_dict.items()):

            if not os.path.exists(config_file):
                continue
            print(i_thermal, current_thermal)
            try:
                df_history, (current_df_modes,
                             kde_dict) = post_process_annealing_results(os.path.join(config_file, 'sa.yaml'),
                                                                        loss_percentile,
                                                                        candidate=True,
                                                                        do_deviation=do_deviation,
                                                                        n_resamples=n_resamples,
                                                                        kernel_bandwidth=kernel_bandwidth)
            except FileNotFoundError as e:
                print(e)
                continue
            current_df_modes['config_file'] = os.path.join(config_file, 'sa.yaml')
            current_df_modes['thermal'] = current_thermal
            current_df_modes['thermal_subindex'] = subindex
            current_df_modes['loss_percentile'] = loss_percentile
            current_df_modes['multiple'] = 'multiple' in config_file

            current_df_modes.to_csv(os.path.join(config_file, f'modes_{loss_percentile}.csv'))
            df_modes = pd.concat([df_modes, current_df_modes])

        df_modes_avg = pd.DataFrame(columns=['WL_mode_mean', 'WL_mode_std', 'WL_mode_sem', 'WL_mode_count'],
                                    index=list_of_birds)
        if do_deviation:
            aa = df_modes.groupby('bird_name').apply(lambda row: get_weighted_average(row['WL_mode'].values,
                                                                                      1 / row['WL_mode_std'].values) if
            row['WL_mode_std'].values is not None else np.nan,
                                                     include_groups=False)

        else:
            aa = df_modes.groupby('bird_name').apply(lambda row:( np.nanmean(row['WL_mode']), np.nan, np.nan,row['WL_mode'].size ),
                                                     include_groups=False)
        for i_col, col in enumerate(['WL_mode_mean', 'WL_mode_std', 'WL_mode_sem', 'WL_mode_count']):
            df_modes_avg[col] = aa.apply(lambda x: x[i_col])
        df_modes_avg.reset_index(names='bird_name', inplace=True)

        df_modes_avg['loss_percentile'] = loss_percentile
        df_modes_all_percentiles = pd.concat([df_modes_all_percentiles, df_modes])
        df_modes_avg_all_percentiles = pd.concat([df_modes_avg_all_percentiles, df_modes_avg])
        if save:
            df_modes.to_csv(os.path.join(aggregated_save_folder, f'modes_{loss_percentile}.csv'))
            df_modes_avg.to_csv(os.path.join(aggregated_save_folder, f'modes_average_{loss_percentile}.csv'))

    if save:
        df_modes_all_percentiles.to_csv(os.path.join(aggregated_save_folder, f'modes.csv'))
        df_modes_avg_all_percentiles.to_csv(os.path.join(aggregated_save_folder, f'modes_average.csv'))

    return df_modes_all_percentiles, df_modes_avg_all_percentiles




def post_process_annealing_results(path_to_config_file, loss_percentile, candidate=True, do_deviation=True,
                                   **kwargs):
    with open(path_to_config_file, 'r') as f:
        sa_config = yaml.safe_load(f)

    path_to_annealing = sa_config['run_parameters']['output_folder']
    is_individual = sa_config['search_parameters']['individual_search']

    df_history = pd.read_csv(os.path.join(path_to_annealing, 'annealing_history.csv'))
    list_of_col = df_history.columns[:-4]
    df_history = df_history[list_of_col]
    if candidate:
        candidate_cols = list_of_col[len(list_of_col) // 2:]
        df_history = df_history[candidate_cols]
    else:
        accepted_cols = list_of_col[:len(list_of_col) // 2]
        df_history = df_history[accepted_cols]

    with open(os.path.join(path_to_annealing, 'sim_annealing_results.yaml'), 'r') as f:
        sa_results = yaml.safe_load(f)

    if is_individual:
        return df_history, post_process_annealing_results_individual(df_history, loss_percentile,
                                                                     do_deviation=do_deviation, **kwargs)
    else:
        return df_history, post_process_annealing_results_average(sa_results, df_history, loss_percentile)


def get_correlations_between_sim_annealing_runs(df_modes_all_percentiles, n_resamples, save_folder, save=True):


    list_of_percentiles = np.sort(df_modes_all_percentiles['loss_percentile'].unique())
    df_corr_all_percentiles = pd.DataFrame()
    for loss_percentile in list_of_percentiles:
        df_modes = df_modes_all_percentiles[df_modes_all_percentiles['loss_percentile'] == loss_percentile]
        df_corr = get_pearson_correlations(df_modes,
                                           'bird_name',
                                           'WL_mode',
                                           'thermal',
                                           method=PermutationMethod(n_resamples=n_resamples, ))
        df_corr['loss_percentile'] = loss_percentile
        if save:
            df_corr.to_csv(os.path.join(save_folder, f'pearson_correlations_{loss_percentile}.csv'))
        df_corr_all_percentiles = pd.concat([df_corr_all_percentiles,df_corr])
    if save:
        df_corr_all_percentiles.to_csv(os.path.join(save_folder, f'pearson_correlations_all_percentiles.csv'), index=False)

    return df_corr_all_percentiles

def post_process_annealing_results_individual(df, loss_percentile=None, do_deviation=True, n_resamples=1000,
                                              kernel_bandwidth=None, loss_col='loss_candidate'):

    current_df_modes = pd.DataFrame()

    parameter_to_use = [c for c in df.columns if c != loss_col]
    parameter_to_use = sorted(parameter_to_use)
    if loss_percentile is not None:
        loss_threshold = np.percentile(df[loss_col], loss_percentile)
        df_threshold = df[df[loss_col] < loss_threshold]
    else:
        df_threshold = df

    wl_array = {}
    kde_dict = {}
    for i, current_col in enumerate(parameter_to_use):
        list_of_modes = {}
        bird_name = current_col.replace('wing_loading_', '').replace('_candidate', '')
        print(bird_name)
        if do_deviation:
            (mode_avg, confidence_interval, mode_std,
             kde) = get_mode_with_deviation_with_KDE(df_threshold.loc[~df_threshold[current_col].isna(), current_col].values,
                                                     n_resamples=n_resamples,
                                                     kernel_bandwidth=kernel_bandwidth,
                                                     #weights=1/ df_threshold['loss_candidate'].values
                                                    )
        else:
            mode_avg, kde = get_mode_with_KDE(df_threshold.loc[~df_threshold[current_col].isna(), current_col].values,
                                              kernel_bandwidth=kernel_bandwidth)
            mode_std = None
            confidence_interval=None
        kde_dict[bird_name] = kde
        list_of_modes['WL_mode'] = [mode_avg]
        list_of_modes['WL_mode_std'] = [mode_std]
        list_of_modes['WL_mode_CI'] = [confidence_interval]
        list_of_modes['WL_count'] = [df_threshold[current_col].count()]
        list_of_modes['WL_full_count'] = [df[current_col].count()]
        list_of_modes['bird_name'] = [bird_name]
        if isinstance(kde, KernelDensity):
            list_of_modes['kde_params'] = [kde.get_params()]

        current_df_modes = pd.concat([current_df_modes, pd.DataFrame(list_of_modes)])

    return current_df_modes, kde_dict


def post_process_annealing_results_average(sa_results, df, loss_percentile):

    loss = df.columns[-1]
    best_array = sa_results['x']

    loss_threshold = np.percentile(df[loss], loss_percentile)
    df_threshold = df[df[loss] < loss_threshold]

    f = lambda x, a, b, c: a * x ** 2 + b * x + c

    popt, pcov = curve_fit(f, df_threshold['wing_loading_candidate'].values,
                           df_threshold[loss].values, p0=(1, 1, 1), )

    x_min = - popt[1] / (2 * popt[0])
    loss_min = f(x_min, *popt)
    return x_min, loss_min, best_array, popt, pcov