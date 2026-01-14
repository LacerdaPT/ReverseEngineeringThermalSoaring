import os.path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import ConstantInputWarning, NearConstantInputWarning

from calc.stats import get_rms_and_pearson
from data.get_data import load_synthetic_and_decomposed, load_decomposition_data, load_synthetic_data
from decomposition.auxiliar import get_decomposition_args, calculate_decomposition_loss
from decomposition.core import decomposition_iteration


def get_rms_and_correlations(df_dec, df_gt_dec, list_of_cols=None, merge_cols=None, **kwargs):

    if merge_cols is None:
        merge_cols = ['bird_name', 'time']
    if list_of_cols is None:
        list_of_cols = ['X_bird_TC',
                        'Y_bird_TC',
                        'dXdT_bird_ground',
                        'dYdT_bird_ground',
                        'dZdT_bird_ground',
                        'dXdT_bird_air',
                        'dYdT_bird_air',
                        'dZdT_bird_air',
                        'curvature_bird_air',
                        # 'radius_bird_air',
                        'bank_angle_bird_air',
                        'X_TC_ground',
                        'Y_TC_ground',
                        'wind_X',
                        'wind_Y',
                        'dXdT_thermal_ground',
                        'dYdT_thermal_ground',
                        'dZdT_thermal_ground',
                        # 'dXdT_air_ground_avg',
                        # 'dYdT_air_ground_avg',
                        # 'dZdT_air_ground_avg',
                        'dXdT_air_ground',
                        'dYdT_air_ground',
                        'dZdT_air_ground',
                        #'epsilon_X',
                        #'epsilon_Y',
                        #'epsilon_Z',
                        ]

    df_join = pd.merge(df_dec, df_gt_dec, on=merge_cols, suffixes=('_dec', '_GT'))
    df_correlations = pd.DataFrame()
    for col in list_of_cols:
        GT_col, dec_col = f'{col}_GT', f'{col}_dec'
        current_rms, current_correlation, current_p_value, current_n = get_rms_and_pearson(df_join[GT_col], df_join[dec_col], **kwargs)
        if df_correlations.empty:
            df_correlations = pd.Series({'GT_col':            GT_col, 'dec_col': dec_col,
                                                'RMS': current_rms,
                                                'pearson_r':       current_correlation,
                                                'pearson_p_value': current_p_value}).to_frame().T
        else:
            df_correlations = pd.concat([df_correlations,
                                         pd.Series({'GT_col':            GT_col,
                                                    'dec_col': dec_col,
                                                    'RMS': current_rms,
                                                    'pearson_r':       current_correlation,
                                                    'pearson_p_value': current_p_value}).to_frame().T], ignore_index=True,
                                        axis=0)

    return df_correlations

def get_rms_and_correlations_per_bird(df_decomposed, df_real, cols_mapping=None, merge_cols=None, bird_col='bird_name',
                                      **kwargs):

    list_of_birds = df_decomposed[bird_col].unique()
    df_correlations = pd.DataFrame()
    for bird in list_of_birds:
        print(bird)
        current_dec = df_decomposed[df_decomposed[bird_col] == bird]
        current_real = df_real[df_real[bird_col] == bird]
        current_correlations = get_rms_and_correlations(current_dec, current_real,
                                                        list_of_cols=cols_mapping, merge_cols=merge_cols,**kwargs)
        current_correlations[bird_col] = bird
        if df_correlations.empty:
            df_correlations = current_correlations.copy()
        else:
            df_correlations = pd.concat([df_correlations, current_correlations])

    return df_correlations

def get_metrics_from_path(decomposition_path, iteration, synthetic_path=None, do_per_bird=True, **kwargs):
    df_air_gt = pd.read_csv(os.path.join(decomposition_path, 'ground_truth_reconstructed', 'thermal.csv'), index_col=False)
    df_air_gt.drop(columns=['curvature'], inplace=True, errors='ignore')
    syn = load_synthetic_data(synthetic_path, list_of_object=['bird.csv'])
    df_bird_gt = syn['bird']
    df_bird_gt.drop(columns = [col for col in df_air_gt.columns if col not in ['bird_name', 'time']],
                                 errors='ignore',
                    inplace=True)
    df_gt = pd.merge(df_bird_gt, df_air_gt, on=['bird_name', 'time'], how='inner' )
    dec = load_decomposition_data(decomposition_path, list_of_files=['thermal.csv'], iteration='best')

    df_dec = dec['thermal']
    df_dec.rename(columns={'curvature':  'curvature_bird_air',
                           'radius':     'radius_bird_air',
                           'bank_angle': 'bank_angle_bird_air', },
                  inplace=True)

    df_correlations = get_rms_and_correlations(df_dec, df_gt,**kwargs)
    if do_per_bird:
        df_correlations_per_bird = get_rms_and_correlations_per_bird(df_dec, df_gt, bird_col='bird_name',**kwargs)
    else:
        df_correlations_per_bird = None
    return df_correlations, df_correlations_per_bird


def get_deltas(path_to_decomposition, path_to_synthetic, iteration):
    synthetic_data_dict, dec = load_synthetic_and_decomposed(path_to_decomposition,
                                                             input_folder=path_to_synthetic,
                                                             iteration=iteration)
    if len(dec['iterations_reconstructed'].index) == 0:
        raise IndexError

    df_real = synthetic_data_dict['data_real']

    df_real.drop(columns=['dXdT_air_thermal_real', 'dYdT_air_thermal_real', 'radius', 'bank_angle', 'curvature'],
                 inplace=True)
    df_real.rename(columns={'dXdT_air_rotation_real': 'dXdT_air_thermal_real',
                            'dYdT_air_rotation_real': 'dYdT_air_thermal_real'}, inplace=True)

    df_dec = dec['iterations_reconstructed']
    df_dec = df_dec[np.abs(df_dec['curvature']) < 0.1]
    df_dec = df_dec[df_dec['in_hull'] & (~df_dec['in_hull'].isna())]
    merge_cols = ['bird_name', 'time']

    cols_mapping = {'X_thermal_real':           'X_bird_TC',
                    'Y_thermal_real':           'Y_bird_TC',
                    'dXdT':                     'dXdT_bird_ground',
                    'dYdT':                     'dYdT_bird_ground',
                    'dZdT':                     'dZdT_bird_ground',
                    'dXdT_bird_real':           'dXdT_bird_air',
                    'dYdT_bird_real':           'dYdT_bird_air',
                    'dZdT_bird_real':           'dZdT_bird_air',
                    'radius_bird_real':         'radius',
                    'bank_angle_bird_real':     'bank_angle',
                    'curvature_bird_real':      'curvature',
                    'thermal_core_X_real':      'X_TC_ground',
                    'thermal_core_Y_real':      'Y_TC_ground',
                    'dXdT_air_wind_real':       'wind_X',
                    'dYdT_air_wind_real':       'wind_Y',
                    'dXdT_air_thermal_real':    'dXdT_thermal_ground_avg',
                    'dYdT_air_thermal_real':    'dYdT_thermal_ground_avg',
                    'dZdT_air_thermal_real':    'dZdT_thermal_ground_avg',
                    'dXdT_air_turbulence_real': 'dXdT_thermal_TC_res',
                    'dYdT_air_turbulence_real': 'dYdT_thermal_TC_res',
                    'dZdT_air_turbulence_real': 'dZdT_thermal_TC_res',
                    'dXdT_air_real':            'dXdT_air_ground',
                    'dYdT_air_real':            'dYdT_air_ground',
                    'dZdT_air_real':            'dZdT_air_ground', }

    columns_to_keep_on_data_real = merge_cols + list(cols_mapping.keys())

    columns_to_keep_on_decomposed = merge_cols + list(cols_mapping.values())
    df_real = df_real[columns_to_keep_on_data_real]
    df_decomposed = df_dec[columns_to_keep_on_decomposed]

    df_join = pd.merge(df_decomposed, df_real,
                       on=merge_cols)

    df_deltas = pd.DataFrame()
    for real_col, dec_col in cols_mapping.items():
        df_deltas[dec_col] = df_join[real_col] - df_join[dec_col]

    df_deltas[['V_air_X', 'V_air_Y', 'V_air_Z']] = (df_join[['dXdT_air_real', 'dYdT_air_real', 'dZdT_air_real']].values
                                                    - df_join[['dXdT_air_ground', 'dYdT_air_ground',
                                                               'dZdT_air_ground']].values)
    df_deltas[['V_bird_X', 'V_bird_Y', 'V_bird_Z']] = (
                df_join[['dXdT_bird_real', 'dYdT_bird_real', 'dZdT_bird_real']].values
                - df_join[['dXdT_bird_air', 'dYdT_bird_air', 'dZdT_bird_air']].values)
    df_deltas[['V_gps_X', 'V_gps_Y', 'V_gps_Z']] = (df_join[['dXdT', 'dYdT', 'dZdT']].values
                                                    - df_join[
                                                        ['dXdT_bird_air', 'dYdT_bird_air', 'dZdT_bird_air']].values
                                                    - df_join[['dXdT_air_ground', 'dYdT_air_ground',
                                                               'dZdT_air_ground']].values)

    df_deltas['V_air'] = np.linalg.norm(df_deltas[['V_air_X', 'V_air_Y', 'V_air_Z']], axis=1)
    df_deltas['V_bird'] = np.linalg.norm(df_deltas[['V_bird_X', 'V_bird_Y', 'V_bird_Z']], axis=1)
    df_deltas['V_gps'] = np.linalg.norm(df_deltas[['V_gps_X', 'V_gps_Y', 'V_gps_Z']], axis=1)

    df_deltas_mean = df_deltas.mean()
    df_deltas_median = df_deltas.median()
    df_deltas_std = df_deltas.std()
    df_deltas_mean.name = 'mean'
    df_deltas_median.name = 'median'
    df_deltas_std.name = 'std'
    df_deltas_agg = pd.concat([df_deltas_mean,
                               df_deltas_median,
                               df_deltas_std], axis=1)

    return df_deltas, df_deltas_agg


def decomposition_last_iteration(current_iteration, current_physical_parameters, path_to_yaml, binning_parameters):
    cols_to_keep = ['bird_name', 'time',
                    'X', 'Y', 'Z', 'rho_bird_TC', 'phi_bird_TC', 'curvature',
                    'dXdT_bird_ground', 'dXdT_air_ground', 'dXdT_air_TC', 'dXdT_bird_air', 'd2XdT2_bird_air',
                    'dYdT_bird_ground', 'dYdT_air_ground', 'dYdT_air_TC', 'dYdT_bird_air', 'd2YdT2_bird_air',
                    'dZdT_bird_ground', 'dZdT_air_ground', 'dZdT_air_TC', 'dZdT_bird_air', 'd2ZdT2_bird_air',
                    ]
    current_iteration = current_iteration.copy()[cols_to_keep]

    parameter_dict = get_decomposition_args({'yaml': path_to_yaml})
    run_parameters = parameter_dict['run_parameters']

    thermal_core_ma_args = parameter_dict['thermal_core_ma_args']

    max_radius = 30  #thermal_core_ma_args.pop('max_radius')

    smoothing_ma_args = parameter_dict['smoothing_ma_args']
    spline_parameters = parameter_dict['spline_parameters']


    min_occupation = 0  # binning_parameters.pop('min_occupation')
    dt = run_parameters['dt']

    debug_dict = {'thermal_core_wind_debug': False,
                  'binning_debug':           False,
                  'flight_debug':            False}
    try:

        (current_iteration,
         current_thermal_core,
         current_physical_parameters,
         current_bins,
         current_spline_stats) = decomposition_iteration(current_iteration, current_physical_parameters, thermal_core_ma_args,
                                                         smoothing_ma_args,
                                                         binning_parameters, spline_parameters, dt, min_occupation,
                                                         max_radius=max_radius, previous_iteration=None,
                                                         iteration_damping=1, step_damping=1, debug_dict=debug_dict)
    except Exception as e:
        raise Exception(str(e))
    current_loss, N_NA, N_total = calculate_decomposition_loss(current_iteration=current_iteration,
                                                               z_limits=[
                                                                   np.min(current_spline_stats['wind']['wind_X']['tck'][0]),
                                                                   np.max(
                                                                       current_spline_stats['wind']['wind_X']['tck'][0])],
                                                               max_allowed_curvature=0.1,
                                                               loss_function='closure')

    list_of_losses = [current_loss, N_NA, N_total]
    return current_iteration, current_thermal_core, current_physical_parameters, current_bins, current_spline_stats, list_of_losses