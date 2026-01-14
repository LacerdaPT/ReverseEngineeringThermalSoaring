import logging
import os

import numpy as np
import pandas as pd
import yaml

from calc.auxiliar import get_smoothed_diff_per_partition, get_geometric_characteristics, get_flight_characteristics, \
    UnivariateSplineWrapper
from calc.stats import get_bins, assign_bins
from calc.thermal import get_thermal_core_by_weighted_average, merge_dfs_with_spline,get_horizontal_thermal_velocity_by_moving_average
from decomposition.auxiliar import get_initial_conditions, get_relative_change, \
    early_stopping, calculate_decomposition_loss
from misc.auxiliar import sanitize_dict_for_yaml

pd.set_option("mode.chained_assignment", None)
logger = logging.getLogger(__name__)


def decomposition_iteration(current_iteration, current_physical_parameters, thermal_core_ma_args, smoothing_ma_args,
                            binning_parameters, spline_args, dt, min_occupation, max_radius=30, debug_dict=False):
    current_spline_stats = {}

    logger.info(f'starting iteration')
    # ================================================================================================================ #
    # ==============================         THERMAL CORE CALCULATION         ======================================== #
    # ================================================================================================================ #

    dict_for_thermal_core = {'X': lambda x: x['dZdT_air_ground'] if ((x['dZdT_air_ground'] > 0)
                                                                and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'Y': lambda x: x['dZdT_air_ground'] if ((x['dZdT_air_ground'] > 0)
                                                                and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'Z': lambda x: x['dZdT_air_ground'] if ((x['dZdT_air_ground'] > 0)
                                                                and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'dXdT_air_ground': lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                           and ((x['rho_bird_TC'] < max_radius) or ('rho_bird_TC' not in x))
                                                           and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'dYdT_air_ground': lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                           and ((x['rho_bird_TC'] < max_radius) or ('rho_bird_TC' not in x))
                                                           and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'dZdT_air_ground': lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                           and ((x['rho_bird_TC'] < max_radius) or ('rho_bird_TC' not in x))
                                                           and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'dXdT_bird_ground': lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                           and ((x['rho_bird_TC'] < max_radius) or ('rho_bird_TC' not in x))
                                                           and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'dYdT_bird_ground': lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                           and ((x['rho_bird_TC'] < max_radius) or ('rho_bird_TC' not in x))
                                                           and (not np.isnan(x['dZdT_air_ground']))) else 0,
                             'dZdT_bird_ground': lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                           and ((x['rho_bird_TC'] < max_radius) or ('rho_bird_TC' not in x))
                                                           and (not np.isnan(x['dZdT_air_ground']))) else 0
                             }
    if 'rho_bird_TC' in current_iteration.columns:
        current_iteration['considered_for_thermal_core'] = ((np.abs(current_iteration['curvature'].values) < 0.1)
                                                            & (~np.isnan(current_iteration['dZdT_air_ground'].values)))
    else:
        current_iteration['considered_for_thermal_core'] = True

        for coord in ['X', 'Y', 'Z']:
            dict_for_thermal_core[f'd{coord}dT_bird_ground']= lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                                        and (not np.isnan(x['dZdT_air_ground']))) else 0
            dict_for_thermal_core[f'd{coord}dT_air_ground']= lambda x: 1 if ((x['dZdT_air_ground'] > 0)
                                                                        and (not np.isnan(x['dZdT_air_ground']))) else 0

    current_thermal_core = get_thermal_core_by_weighted_average(
        current_iteration,
        dict_for_thermal_core,
        ma_args=thermal_core_ma_args,
        filter_col='considered_for_thermal_core',
        sort_by='Z')

    (current_iteration,
     current_spline_stats['thermal_core_positions']) = merge_dfs_with_spline(current_iteration, current_thermal_core,
                                                                         other_cols_to_merge={
                                                                             'X_avg': 'X_sem',
                                                                             'Y_avg': 'Y_sem',
                                                                             'Z_avg': 'Z_sem',
                                                                             'dXdT_bird_ground_avg': 'dXdT_bird_ground_sem',
                                                                             'dYdT_bird_ground_avg': 'dYdT_bird_ground_sem',
                                                                             'dZdT_bird_ground_avg': 'dZdT_bird_ground_sem',
                                                                             'dXdT_air_ground_avg': 'dXdT_air_ground_sem',
                                                                             'dYdT_air_ground_avg': 'dYdT_air_ground_sem',
                                                                             'dZdT_air_ground_avg': 'dZdT_air_ground_sem',
                                                                         },
                                                                     other_merge_on='Z_avg',
                                                                     merge_on='Z',
                                                                     spline_degree=spline_args['degree'],
                                                                     smoothing_factor=spline_args['smoothing_factor'],
                                                                     extrapolate=spline_args['extrapolate'])

    current_iteration.rename(columns={'X_avg': 'X_TC_ground',
                                      'Y_avg': 'Y_TC_ground',
                                      }, inplace=True)


    for coord in ['X', 'Y']:
        merging_spline = UnivariateSplineWrapper.from_tck(current_spline_stats['thermal_core_positions'][f'{coord}_avg']['tck'])

        current_thermal_core[f'd{coord}dZ'] = merging_spline(current_thermal_core['Z_avg'].values, nu=1,
                                                      extrapolate=spline_args['extrapolate'])

    current_thermal_core['dXdT_TC_ground'] = current_thermal_core['dXdZ'] * current_thermal_core['dZdT_bird_ground_avg']
    current_thermal_core['dYdT_TC_ground'] = current_thermal_core['dYdZ'] * current_thermal_core['dZdT_bird_ground_avg']
    current_thermal_core['dZdT_TC_ground'] = current_thermal_core['dZdT_bird_ground_avg']
    (current_iteration,
     current_spline_stats['thermal_core_velocities']) = merge_dfs_with_spline(current_iteration, current_thermal_core,
                                                                              other_cols_to_merge=['dXdZ',
                                                                                                   'dYdZ',
                                                                                                   'dXdT_TC_ground',
                                                                                                   'dYdT_TC_ground',
                                                                                                   'dZdT_TC_ground'],
                                                                              other_merge_on='Z_avg',
                                                                              merge_on='Z',
                                                                              spline_degree=spline_args['degree'],
                                                                              smoothing_factor=spline_args['smoothing_factor'],
                                                                              extrapolate=spline_args['extrapolate'])


    current_iteration['X_bird_TC'] = current_iteration['X'] - current_iteration['X_TC_ground']
    current_iteration['Y_bird_TC'] = current_iteration['Y'] - current_iteration['Y_TC_ground']
    current_iteration['Z_bird_TC'] = current_iteration['Z']  # - thermal_core['Z_avg']
    current_iteration['dXdT_bird_TC'] = current_iteration['dXdT_bird_ground'] - current_iteration['dXdT_TC_ground']
    current_iteration['dYdT_bird_TC'] = current_iteration['dYdT_bird_ground'] - current_iteration['dYdT_TC_ground']
    current_iteration['dZdT_bird_TC'] = current_iteration['dZdT_bird_ground'] - current_iteration['dZdT_TC_ground']

    current_iteration['dXdT_bird_air'] = current_iteration['dXdT_bird_TC'] - current_iteration['dXdT_air_TC']
    current_iteration['dYdT_bird_air'] = current_iteration['dYdT_bird_TC'] - current_iteration['dYdT_air_TC']
    current_iteration['dZdT_bird_air'] = current_iteration['dZdT_bird_TC'] - current_iteration['dZdT_air_TC']

    diff_params = {'dXdT_bird_air': smoothing_ma_args,
                   'dYdT_bird_air': smoothing_ma_args,
                   'dZdT_bird_air': smoothing_ma_args,
                   'dXdT_air_TC': smoothing_ma_args,
                   'dYdT_air_TC': smoothing_ma_args,
                   'dZdT_air_TC': smoothing_ma_args}
    diff_results = get_smoothed_diff_per_partition(current_iteration, diff_params, n=1, dt=dt, partition_key='bird_name')

    current_iteration['d2XdT2_bird_air'] = diff_results['dXdT_bird_air']
    current_iteration['d2YdT2_bird_air'] = diff_results['dYdT_bird_air']
    current_iteration['d2ZdT2_bird_air'] = diff_results['dZdT_bird_air']

    current_iteration['d2XdT2_air_TC'] = diff_results['dXdT_air_TC']
    current_iteration['d2YdT2_air_TC'] = diff_results['dYdT_air_TC']
    current_iteration['d2ZdT2_air_TC'] = diff_results['dZdT_air_TC']

    current_iteration['rho_bird_TC'] = np.linalg.norm(current_iteration[['X_bird_TC', 'Y_bird_TC']], axis=-1)
    current_iteration['phi_bird_TC'] = np.arctan2(current_iteration['Y_bird_TC'], current_iteration['X_bird_TC'])


    # ============================================================================================================ #
    # ========================================  FLIGHT DYNAMICS  ================================================= #
    # ============================================================================================================ #
    current_iteration.drop(columns='curvature', inplace=True, errors='ignore')
    df_calc = get_geometric_characteristics(current_iteration,
                                            state_vector_columns={'Vx': f'dXdT_bird_air',
                                                                  'Vy': f'dYdT_bird_air',
                                                                  'Ax': f'd2XdT2_bird_air',
                                                                  'Ay': f'd2YdT2_bird_air'})

    current_iteration = pd.merge(current_iteration, df_calc, on=['time', 'bird_name'])
    df_calc = get_flight_characteristics(current_iteration, df_bird_parameters=current_physical_parameters,
                                         state_vector_columns={'Vx': f'dXdT_bird_air',
                                                               'Vy': f'dYdT_bird_air'
                                                               },
                                         radius_col='radius')
    current_iteration = pd.merge(current_iteration, df_calc, on=['time', 'bird_name'])

    current_iteration['dXdT_bird_air'] = current_iteration['Vx']
    current_iteration['dYdT_bird_air'] = current_iteration['Vy']
    current_iteration['dZdT_bird_air'] = current_iteration['Vz']
    current_iteration['dXdT_air_TC'] = current_iteration['dXdT_bird_TC'] - current_iteration['dXdT_bird_air']
    current_iteration['dYdT_air_TC'] = current_iteration['dYdT_bird_TC'] - current_iteration['dYdT_bird_air']
    current_iteration['dZdT_air_TC'] = current_iteration['dZdT_bird_TC'] - current_iteration['dZdT_bird_air']
    # ================================================================================================================ #
    # =====================================         BINNING         ================================================== #
    # ================================================================================================================ #
    # Preprocessing
    # np.digitize returns one-indexed binning, e.g., if bin_edges = [0,1,2,3], 0.2 will be assigned bin 1 not zero

    cols_to_bin = ['Z_bird_TC', 'phi_bird_TC', 'rho_bird_TC']
    logger.info('starting Binning process')
    current_bins, bin_index_cols = get_bins(current_iteration, cols_to_bin=cols_to_bin, **binning_parameters)

    current_iteration = current_iteration.drop(columns=bin_index_cols, errors='ignore')
    current_iteration = current_iteration.join(assign_bins(current_iteration, current_bins, is_adaptive=False,
                                                           grid_cols_to_bin=cols_to_bin[:-1],
                                                           grid_index_cols=bin_index_cols[:-1],
                                                           adaptive_col_to_bin=cols_to_bin[-1],
                                                           adaptive_index_col=bin_index_cols[-1]
                                                           )
                                               )

    # ============================================================================================================ #
    # =========================================   BIN AVERAGES   ================================================= #
    # ============================================================================================================ #

    df_averages = current_iteration.groupby(bin_index_cols).agg(dXdT_air_TC_bin_avg=('dXdT_air_TC', 'median'),
                                                                dYdT_air_TC_bin_avg=('dYdT_air_TC', 'median'),
                                                                dZdT_air_TC_bin_avg=('dZdT_air_TC', 'median'),
                                                                dXdT_air_TC_bin_std=('dXdT_air_TC', 'std'),
                                                                dYdT_air_TC_bin_std=('dYdT_air_TC', 'std'),
                                                                dZdT_air_TC_bin_std=('dZdT_air_TC', 'std'),
                                                                dXdT_air_TC_bin_count=('dXdT_air_TC', 'count'),
                                                                dYdT_air_TC_bin_count=('dYdT_air_TC', 'count'),
                                                                dZdT_air_TC_bin_count=('dZdT_air_TC', 'count'),
                                                                )

    current_bins = current_bins.merge(df_averages, left_on=bin_index_cols, right_index=True).sort_index()
    current_iteration = current_iteration.merge(df_averages, left_on=bin_index_cols, right_index=True).sort_index()

    # TODO
    # Change to >= min_occupation
    for coord in ['X', 'Y', 'Z']:
        current_mask = ((current_iteration[f'd{coord}dT_air_TC_bin_count'] > min_occupation)
                              & np.all(current_iteration[bin_index_cols] != -1, axis=1))
        current_iteration.loc[current_mask, f'd{coord}dT_air_TC'] = current_iteration.loc[current_mask, f'd{coord}dT_air_TC_bin_avg']
        current_iteration.loc[~current_mask, f'd{coord}dT_air_TC'] = current_iteration.loc[~current_mask, f'd{coord}dT_air_TC']


    current_iteration[f'dXdT_air_ground'] = current_iteration[f'dXdT_air_TC'] + current_iteration['dXdT_TC_ground']
    current_iteration[f'dYdT_air_ground'] = current_iteration[f'dYdT_air_TC'] + current_iteration['dYdT_TC_ground']
    current_iteration[f'dZdT_air_ground'] = current_iteration[f'dZdT_air_TC'] + current_iteration['dZdT_TC_ground']


    # ================================================================================================================ #
    # =====================================     REJECTION OF HORIZONTAL THERMAL VELOCITY    ========================== #
    # ================================================================================================================ #

    current_df_thermal_V_H = get_horizontal_thermal_velocity_by_moving_average(current_iteration,
                                                                               window=thermal_core_ma_args['window'],
                                                                               min_periods=thermal_core_ma_args['min_periods'],
                                                                               Z_col='Z', V_H_cols=['dXdT_air_ground',
                                                                                                    'dYdT_air_ground']
                                                                               )

    current_iteration, current_spline_stats['wind'] = merge_dfs_with_spline(current_iteration, current_df_thermal_V_H,
                                                                                   merge_on='Z',
                                                                                   other_cols_to_merge={'wind_X': 'wind_X_sem',
                                                                                                        'wind_Y': 'wind_Y_sem',},
                                                                                   other_merge_on='Z_avg',
                                                                                   spline_degree=spline_args['degree'],
                                                                                   smoothing_factor=spline_args['smoothing_factor'],
                                                                                   extrapolate=spline_args['extrapolate'],
                                                                                       debug=False)

    current_iteration['dXdT_thermal_ground'] = current_iteration['dXdT_air_ground'] - current_iteration['wind_X']
    current_iteration['dYdT_thermal_ground'] = current_iteration['dYdT_air_ground'] - current_iteration['wind_Y']
    current_iteration['dZdT_thermal_ground'] = current_iteration['dZdT_air_ground']

    return current_iteration, current_thermal_core, current_physical_parameters, current_bins, current_spline_stats


def start_decomposition(df, run_parameters, initial_physical_parameters, thermal_core_ma_args=None,
                        smoothing_ma_args=None, binning_parameters=None, spline_parameters=None, debug_dict=None):

    dt = run_parameters['dt']
    alpha = run_parameters['alpha']
    n_iterations = run_parameters['n_iterations']
    save_only_last = run_parameters['save_only_last']
    relative_change_threshold = run_parameters['relative_change_threshold']
    n_relative_change = run_parameters['n_relative_change']
    min_iter = run_parameters['min_iter']
    logger.setLevel(run_parameters['verbosity'])
    max_radius = thermal_core_ma_args.pop('max_radius')
    min_occupation = binning_parameters.pop('min_occupation')

    df_calc = df.copy()
    df_initial = get_initial_conditions(df_calc, dt=dt, alpha=alpha, smoothing_ma_args=smoothing_ma_args)
    if debug_dict is None:
        debug_dict = {'thermal_core_wind_debug': False,
                      'binning_debug': False,
                      'flight_debug': False}

    current_physical_parameters = initial_physical_parameters.copy()

    df_iterations = df_initial.copy()

    df_physical_parameters = initial_physical_parameters.copy()
    df_physical_parameters['iteration'] = 0

    df_bins = pd.DataFrame()
    df_thermal_core = pd.DataFrame()
    df_spline_stats = pd.DataFrame()
    current_iteration = df_initial.copy()
    #current_iteration['wind_X'] = 0
    #current_iteration['wind_Y'] = 0
    previous_iteration = None
    list_of_relative_change = []
    list_of_losses = []
    # ======================================================================================================================
    #                                                       ITERATIONS
    # ======================================================================================================================

    for i in range(1, n_iterations):
        logger.info(f'iteration {i}')
        (current_iteration,
         current_thermal_core,
         current_physical_parameters,
         current_bins,
         current_spline_stats) = decomposition_iteration(current_iteration=current_iteration,
                                                         dt=dt,
                                                         min_occupation=min_occupation,
                                                         max_radius=max_radius,
                                                         current_physical_parameters=current_physical_parameters,
                                                         thermal_core_ma_args=thermal_core_ma_args,
                                                         smoothing_ma_args=smoothing_ma_args,
                                                         binning_parameters=binning_parameters,
                                                         spline_args=spline_parameters,
                                                         debug_dict=debug_dict)
        # ================================================================================================
        #                                        UPDATE
        # ================================================================================================

        current_thermal_core['iteration'] = i
        current_iteration['iteration'] = i
        current_bins['iteration'] = i
        current_physical_parameters['iteration'] = i
        current_spline_stats['iteration'] = i
        if not save_only_last:
            if i == 1:
                df_thermal_core = current_thermal_core.copy()
                df_iterations = current_iteration.copy()
                df_bins = current_bins.copy()
                df_physical_parameters = current_physical_parameters.copy()
                df_spline_stats = pd.DataFrame.from_records([current_spline_stats], ).copy()
            else:
                df_thermal_core = pd.concat([df_thermal_core, current_thermal_core])
                df_iterations = pd.concat([df_iterations, current_iteration])  # index
                df_bins = pd.concat([df_bins, current_bins])
                df_physical_parameters = pd.concat(
                    [df_physical_parameters, current_physical_parameters])
                df_spline_stats = pd.concat(
                    [df_spline_stats, pd.DataFrame.from_records([current_spline_stats], )])
        else:
            df_thermal_core = current_thermal_core.copy()
            df_iterations = current_iteration.copy()
            df_bins = current_bins.copy()
            df_physical_parameters = current_physical_parameters.copy()
            df_spline_stats = current_spline_stats.copy()

        if previous_iteration is not None:
            list_of_cols = ['dXdT_air_TC', 'dYdT_air_TC','dZdT_air_TC',
                            'dXdT_bird_air','dYdT_bird_air','dZdT_bird_air']
            relative_change = np.mean(np.linalg.norm(get_relative_change(current_iteration, previous_iteration,
                                                       list_of_cols=list_of_cols,
                                                       merge_on=['bird_name', 'time']),
                                                     axis=1)
                                      )
            logger.info(f'{relative_change=:.3g}')
            list_of_relative_change.append(relative_change)
        current_loss, N_NA, N_total = calculate_decomposition_loss(current_iteration=current_iteration,
                                                          z_limits=[np.min(current_spline_stats['wind']['wind_X']['tck'][0]),
                                                                    np.max(current_spline_stats['wind']['wind_X']['tck'][0])],
                                                          max_allowed_curvature=0.1,
                                                          loss_function='closure')
        logger.info(f'{current_loss=:.10g}')
        list_of_losses.append([current_loss, N_NA, N_total])
        if early_stopping(i,np.array(list_of_losses)[:,0], relative_change_threshold, n_relative_change, min_iter, n_iterations):
            break

        cols_to_keep = ['bird_name', 'time',
                        'X', 'Y', 'Z', 'rho_bird_TC', 'phi_bird_TC', 'curvature',
                        'dXdT_bird_ground', 'dXdT_air_ground',  'dXdT_air_TC', 'dXdT_bird_air',  'd2XdT2_bird_air',  # 'd2XdT2_air_0', 'd2XdT2_bird_0'
                        'dYdT_bird_ground', 'dYdT_air_ground',  'dYdT_air_TC', 'dYdT_bird_air',  'd2YdT2_bird_air',  # 'd2YdT2_air_0', 'd2YdT2_bird_0'
                        'dZdT_bird_ground', 'dZdT_air_ground',  'dZdT_air_TC', 'dZdT_bird_air',  'd2ZdT2_bird_air',
                        #'X_TC_air', 'Y_TC_air', 'Z_TC_air',
                        #'X_bird_air', 'Y_bird_air', 'Z_bird_air'
                        ]  # 'd2ZdT2_air_0', 'd2ZdT2_bird_0'
        previous_iteration = current_iteration.copy()
        current_iteration = current_iteration.copy()[cols_to_keep]
        if i != 1:
            current_iteration = pd.merge(current_iteration,
                                         previous_iteration[['bird_name', 'time'] + ['dXdT_air_TC', 'dYdT_air_TC', 'dZdT_air_TC',]],
                                         on=['bird_name', 'time'], how='left', suffixes=(None, '_prev'))
            current_iteration.loc[current_iteration['dXdT_air_TC'].isna(), 'dXdT_air_TC'] = current_iteration.loc[current_iteration['dXdT_air_TC'].isna(), 'dXdT_air_TC_prev']
            current_iteration.loc[current_iteration['dYdT_air_TC'].isna(), 'dYdT_air_TC'] = current_iteration.loc[current_iteration['dYdT_air_TC'].isna(), 'dYdT_air_TC_prev']
            current_iteration.loc[current_iteration['dZdT_air_TC'].isna(), 'dZdT_air_TC'] = current_iteration.loc[current_iteration['dZdT_air_TC'].isna(), 'dZdT_air_TC_prev']
            current_iteration.drop(columns=['dXdT_air_TC_prev',
                                            'dYdT_air_TC_prev',
                                            'dZdT_air_TC_prev',], inplace=True)

        current_iteration['dXdT_air_TC'] = current_iteration['dXdT_air_TC'].fillna(0.0)
        current_iteration['dYdT_air_TC'] = current_iteration['dYdT_air_TC'].fillna(0.0)
        current_iteration['dZdT_air_TC'] = current_iteration['dZdT_air_TC'].fillna(0.0)

        del current_physical_parameters['iteration']

    if run_parameters['save']:
        if run_parameters['output_folder'] is None:
            destination_folder = os.path.join(run_parameters['input_folder'], 'decomposition',
                                              run_parameters['run_time'])
        else:
            destination_folder = run_parameters['output_folder']

        try:
            os.makedirs(destination_folder)
        except FileExistsError as e:
            pass

        df_thermal_core.to_csv(f'{destination_folder}/thermal_core.csv',)
        df_physical_parameters.to_csv(f'{destination_folder}/aerodynamic_parameters.csv')
        df_iterations.to_csv(f'{destination_folder}/iterations.csv')
        df_bins.to_csv(f'{destination_folder}/bins.csv')
        df_loss = pd.DataFrame(list_of_losses, columns=['loss', 'N_NA', 'N_total'])
        df_loss['iteration'] = np.arange(1, df_loss['loss'].size + 1)
        df_loss.to_csv(f'{destination_folder}/losses.csv')

        list_of_splines = [row.to_dict() for _, row in df_spline_stats.iterrows()]

        with open(os.path.join(destination_folder, 'splines.yaml'), 'w') as f:
            yaml.dump(list(map(sanitize_dict_for_yaml, list_of_splines)), f, default_flow_style=False)

        logger.info(f'saved to {destination_folder}')
    logger.info('Done')
    return df_iterations, df_thermal_core, df_physical_parameters, df_bins, df_spline_stats, list_of_losses
