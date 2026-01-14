import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd
import yaml

from calc.auxiliar import get_smoothed_diff_per_partition, get_moving_average_per_bird, get_na_mask
from data.auxiliar import get_args_from_yaml_with_default

logger = logging.getLogger(__name__)

default_decomposition_parameters = {'run_parameters': {'alpha': 0.9,
                                                       'dt': 1,
                                                       'n_iterations': 30,
                                                       'verbosity': 2,
                                                       'save': True,
                                                       'save_only_last': False,
                                                       'input_folder': None,
                                                       'output_folder': None,
                                                       'iteration_damping': 1,
                                                       'step_damping': 1,
                                                       'relative_change_threshold': 0,
                                                       'n_relative_change': 3,
                                                       'min_iter': 12,
                                                       'debug': False},
                                    'data_parameters': {'columns': {'time_col': 'time',
                                                                    'bird_name_col': 'bird_name',
                                                                    'X_col': 'X',
                                                                    'Y_col': 'Y',
                                                                    'Z_col': 'Z'},
                                                        'true_values_cols': None,
                                                        'filters': None},
                                    'thermal_core_ma_args': {'window': None,
                                                             'win_type': 'gaussian',
                                                             'min_periods': None,
                                                             'center': True,
                                                             'max_radius': 30,
                                                             'window_args': None,
                                                             'debug': False},
                                    'smoothing_ma_args': {'window': 3,
                                                          'win_type': 'gaussian',
                                                          'min_periods': 3,
                                                          'center': True,
                                                          'window_args': {'std': 3}
                                                          },
                                    'physical_parameters': {'CL': 1.2,
                                                            'mass': None,
                                                            'wing_area': None,
                                                            'wing_loading': None,
                                                            'CD': 0.08684,
                                                            'debug': False},
                                    'binning_parameters': {'n_bins': (20, 20),
                                                           'method': 'adaptive',
                                                           'adaptive_kwargs':
                                                               {'adaptive_bin_min_size': 1,
                                                                'adaptive_bin_max_size': 5,
                                                                'max_bin_count': 10},
                                                           'min_occupation': 4,
                                                           'debug': False
                                                           },
                                    'spline_parameters': {'degree': 3,
                                                          'smoothing_factor': 0.1,
                                                          'extrapolate': 3,
                                                          'weighted': None,
                                                          'weight_normalization': False}
                                    }


def get_decomposition_args_from_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        yaml_dict = yaml.load(f, yaml.FullLoader)
    parameter_dict = default_decomposition_parameters.copy()

    for parameter_set in default_decomposition_parameters.keys():
        if parameter_set in yaml_dict:
            if yaml_dict[parameter_set] is not None:
                parameter_dict[parameter_set].update(yaml_dict[parameter_set])
            else:
                parameter_dict[parameter_set] = yaml_dict[parameter_set]
    return parameter_dict


def get_decomposition_args(args):
    # Read from yaml
    if args['yaml']:
        parameter_dict = get_args_from_yaml_with_default(args['yaml'], default_decomposition_parameters)
    else:
        parameter_dict = default_decomposition_parameters.copy()

    # Read from Command line - OVERWRITES YAML!!
    for parameter_set in default_decomposition_parameters.keys():
        if parameter_set not in args:
            continue
        for parameter in default_decomposition_parameters[parameter_set].keys():
            if (parameter in args[parameter_set]) and (args[parameter_set][parameter] is not None):
                if parameter == 'window_args':
                    parameter_dict[parameter_set]['window_args'] = {elem.split(':')[0]: cast(elem.split(':')[1],
                                                                                             elem.split(':')[2])
                                                                    for elem in args[parameter_set][parameter]}
                else:
                    parameter_dict[parameter_set][parameter] = args[parameter_set][parameter]

    if isinstance(parameter_dict['run_parameters']['verbosity'], int):
        parameter_dict['run_parameters']['verbosity'] = parameter_dict['run_parameters']['verbosity'] * 10
    else:
        parameter_dict['run_parameters']['verbosity'] = logging.getLevelName(parameter_dict['run_parameters']['verbosity'].upper())
    parameter_dict['run_parameters']['run_time'] = datetime.datetime.isoformat(datetime.datetime.now()).split(".")[0]

    return parameter_dict


def data_preparation(df, dt, smooth_coordinates=False, smoothing_ma_args=None, columns=None, filters=None):

    if columns is not None:
        columns = {'bird_name': 'bird_name', 'time': 'time', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}
        columns.update(columns)

    df_calc = df.copy()

    df_calc = df_calc[[columns['bird_name'], columns['time'], columns['X'], columns['Y'], columns['Z']]]
    if smooth_coordinates:
        coord_ma_args = {'center': True,
                         'min_periods': 1,
                         'window': 3,
                         'win_type': 'gaussian',
                         'window_args': {'std': 1}}
        ma_params = {columns['X']: coord_ma_args,
                     columns['Y']: coord_ma_args,
                     columns['Z']: coord_ma_args
                     }

        ma_results = get_moving_average_per_bird(df_calc, ma_params)
        df_calc[columns['X']] = ma_results[columns['X']]
        df_calc[columns['Y']] = ma_results[columns['Y']]
        df_calc[columns['Z']] = ma_results[columns['Z']]
    # Calculate Velocities
    diff_params = {columns['X']: smoothing_ma_args,
                   columns['Y']: smoothing_ma_args,
                   columns['Z']: smoothing_ma_args
                   }

    diff_results = get_smoothed_diff_per_partition(df_calc, diff_params, dt=dt, partition_key='bird_name')

    df_calc[f'dXdT'] = diff_results[columns['X']]
    df_calc[f'dYdT'] = diff_results[columns['Y']]
    df_calc[f'dZdT'] = diff_results[columns['Z']]

    # Filter
    # Save the extreme values before filtering. Otherwise, the max and min on the following filters will be affected
    # by the previous filters.
    if filters is not None:
        extrema_dict = {col: {'min': df_calc[col].min(),
                              'max': df_calc[col].max()} for col in filters}
        for col, filter_parameters in filters.items():
            min_value = None
            max_value = None
            if 'min' in filter_parameters:
                min_value = filter_parameters['min']
            if 'max' in filter_parameters:
                max_value = filter_parameters['max']

            if filter_parameters['filter_type'] == 'relative':
                min_value = extrema_dict[col]['min'] + min_value if min_value is not None else None
                max_value = extrema_dict[col]['max'] + max_value if max_value is not None else None

            if min_value is not None:
                df_calc = df_calc[df_calc[col] >= min_value]

            if max_value is not None:
                df_calc = df_calc[df_calc[col] <= max_value]

    return df_calc


def get_initial_conditions(df, dt, alpha, smoothing_ma_args, partition_name='bird_name'):
    if np.isscalar(alpha):
        alpha = [alpha] * 3
    df_iteration = pd.DataFrame()

    df_iteration['bird_name'] = df[partition_name]
    df_iteration['time'] = df['time']

    df_iteration['X'] = df['X']
    df_iteration['Y'] = df['Y']
    df_iteration['Z'] = df['Z']

    diff_params = {'X': smoothing_ma_args,
                   'Y': smoothing_ma_args,
                   'Z': smoothing_ma_args
                   }

    diff_results = get_smoothed_diff_per_partition(df_iteration, diff_params, dt=dt, partition_key='bird_name')


    df_iteration['dXdT_bird_ground'] = diff_results['X']
    df_iteration['dYdT_bird_ground'] = diff_results['Y']
    df_iteration['dZdT_bird_ground'] = diff_results['Z']

    df_iteration['dXdT_air_ground'] = alpha[0] * df_iteration['dXdT_bird_ground']
    df_iteration['dYdT_air_ground'] = alpha[1] * df_iteration['dYdT_bird_ground']
    df_iteration['dZdT_air_ground'] = alpha[2] * df_iteration['dZdT_bird_ground']

    df_iteration['dXdT_bird_air'] = df_iteration['dXdT_bird_ground'] - df_iteration['dXdT_air_ground']
    df_iteration['dYdT_bird_air'] = df_iteration['dYdT_bird_ground'] - df_iteration['dYdT_air_ground']
    df_iteration['dZdT_bird_air'] = df_iteration['dZdT_bird_ground'] - df_iteration['dZdT_air_ground']

    df_iteration['dXdT_air_TC'] = 0
    df_iteration['dYdT_air_TC'] = 0
    df_iteration['dZdT_air_TC'] = 0
    df_iteration['X_TC_air'] = 0
    df_iteration['Y_TC_air'] = 0
    df_iteration['Z_TC_air'] = 0
    df_iteration['dXdT_air_TC'] = df_iteration['dXdT_air_TC'].astype(np.float64)
    df_iteration['dYdT_air_TC'] = df_iteration['dYdT_air_TC'].astype(np.float64)
    df_iteration['dZdT_air_TC'] = df_iteration['dZdT_air_TC'].astype(np.float64)
    df_iteration['X_TC_air'] = df_iteration['X_TC_air'].astype(np.float64)
    df_iteration['Y_TC_air'] = df_iteration['Y_TC_air'].astype(np.float64)
    df_iteration['Z_TC_air'] = df_iteration['Z_TC_air'].astype(np.float64)
    # diff_results = get_smoothed_diff_per_partition(df_iteration, diff_params, n=2, dt=dt, partition_key=partition_name)
    #
    # df_iteration['d2XdT2_bird_ground'] = diff_results['X']
    # df_iteration['d2YdT2_bird_ground'] = diff_results['Y']
    # df_iteration['d2ZdT2_bird_ground'] = diff_results['Z']
    df_iteration['iteration'] = 0
    return df_iteration


def cast(value, type_name):
    try:
        if type_name == 'int':
            value = int(value)
        elif type_name == 'float':
            value = float(value)
        else:
            value = value
    except TypeError as e:
        raise e
    else:
        return value


def parse_decomposition_arguments(parse_args):

    args = {'yaml': parse_args['yaml'],
            'run_parameters': {'alpha': parse_args['alpha'],
                               'n_iterations': parse_args['n_iterations'],
                               'verbosity': parse_args['verbosity'],
                               'save': parse_args['save'],
                               'input_folder': parse_args['input_folder'],
                               'output_folder': parse_args['output_folder'],
                               'dt': parse_args['dt'],
                               'debug': parse_args['debug']
                               },
            'data_parameters': {'time_col': parse_args['time_col'],
                                'bird_name_col': parse_args['bird_name_col'],
                                'X_col': parse_args['X_col'],
                                'Y_col': parse_args['Y_col'],
                                'Z_col': parse_args['Z_col']},
            'thermal_core_ma_args': {'window': parse_args['thermal_window_size'],
                                     'min_periods': parse_args['thermal_min_periods'],
                                     'center': parse_args['thermal_center'],
                                     'win_type': parse_args['thermal_window_type'],
                                     'window_args': parse_args['thermal_window_params']},
            'smoothing_ma_args': {'window': parse_args['smoothing_window_size'],
                                  'min_periods': parse_args['smoothing_min_periods'],
                                  'center': parse_args['smoothing_center'],
                                  'win_type': parse_args['smoothing_window_type'],
                                  'window_args': parse_args['smoothing_window_params']},
            'physical_parameters': {'CL': parse_args['CL'],
                                    'CD': parse_args['CD'],
                                    'mass': parse_args['mass'],
                                    'wing_area': parse_args['wing_area']
                                    },
            }

    return args




def decomposition_preparation(decomposition_args):
    parameter_dict = get_decomposition_args(decomposition_args)

    run_parameters = parameter_dict['run_parameters']
    data_parameters = parameter_dict['data_parameters']
    thermal_core_ma_args = parameter_dict['thermal_core_ma_args']
    smoothing_ma_args = parameter_dict['smoothing_ma_args']
    physical_parameters = parameter_dict['physical_parameters']
    binning_parameters = parameter_dict['binning_parameters']
    spline_parameters = parameter_dict['spline_parameters']

    debug_dict = {'thermal_core_wind_debug': thermal_core_ma_args.pop('debug', False),
                  'binning_debug': binning_parameters.pop('debug', False),
                  'flight_debug': physical_parameters.pop('debug', False)}
    if run_parameters['debug']:
        debug_dict = {'thermal_core_wind_debug': True,
                      'binning_debug': True,
                      'flight_debug': True}

    if run_parameters['save']:
        if run_parameters['output_folder'] is None:
            run_parameters['output_folder'] = os.path.join(run_parameters['input_folder'], 'decomposition',
                                                           run_parameters['run_time'])

        destination_folder = run_parameters['output_folder']

        try:
            os.makedirs(destination_folder)
        except FileExistsError as e:
            pass
    else:
        destination_folder = ''

    df = pd.read_csv(os.path.join(run_parameters["input_folder"], 'data.csv'))

    list_of_birds = df['bird_name'].unique()
    n_birds = len(list_of_birds)
    df_physical_parameters = pd.DataFrame()
    df_physical_parameters['bird_name'] = sorted(list_of_birds)

    for param, value in physical_parameters.items():
        df_physical_parameters[param] = value

    if data_parameters['true_values_cols'] is not None:
        true_values_cols = ['bird_name', 'time'] + data_parameters['true_values_cols']
        df_true_values = df[true_values_cols]
    else:
        df_true_values = None
    df = data_preparation(df, dt=run_parameters['dt'], smoothing_ma_args=smoothing_ma_args,
                          columns=data_parameters['columns'],
                          filters=data_parameters['filters'])
    return df, df_true_values, run_parameters, data_parameters, thermal_core_ma_args, smoothing_ma_args, df_physical_parameters, binning_parameters, spline_parameters, debug_dict


def get_relative_change(a, b, list_of_cols=None, merge_on=None):

    if merge_on is None:
        merge_on = ['bird_name', 'time']
    if list_of_cols is None:
        list_of_cols = ['dXdT_air_4', 'dYdT_air_4', 'dZdT_air_4',
                        'dXdT_bird_4', 'dYdT_bird_4', 'dZdT_bird_4']

    df_change = pd.merge(a[merge_on + list_of_cols],
                         b[merge_on + list_of_cols],
                         on=merge_on, how='inner', suffixes=(None, '_prev'))

    if np.any(np.all(np.isclose(df_change[[f'{col}_prev' for col in list_of_cols]].values, 0), axis=0)):
        return np.full_like(df_change[list_of_cols].values, fill_value=np.nan)
    else:
        relative_change = ((df_change[list_of_cols].values
                            - df_change[[f'{col}_prev' for col in list_of_cols]].values)
                           / df_change[[f'{col}_prev' for col in list_of_cols]].values)

        na_mask = np.isnan(relative_change)
        na_mask = np.any(na_mask, axis=1)
        relative_change = relative_change[~na_mask]

        return relative_change


def calculate_decomposition_loss(current_iteration: pd.DataFrame, z_limits: list, max_allowed_curvature: float, loss_function,
                                 true_values: pd.DataFrame = None):
    z_min, z_max = z_limits

    df_calc = current_iteration.copy()
    N_total = df_calc['time'].size
    #df_calc = df_calc[df_calc['Z'].between(z_min, z_max)]
    df_calc = df_calc[np.abs(df_calc['curvature']) < max_allowed_curvature]

    na_mask = get_na_mask(df_calc['dXdT_air_ground'].values,
                          df_calc['dYdT_air_ground'].values,
                          df_calc['dZdT_air_ground'].values)
    N_NA = (~na_mask).sum()
    logger.debug(f'number of NA = {N_NA}, {N_total}, {N_NA/N_total:.4f}')
    try:
        if true_values is None:
            if 'closure' in loss_function:
                current_closure = (df_calc[['dXdT_bird_ground', 'dYdT_bird_ground', 'dZdT_bird_ground']].values
                                   - df_calc[['dXdT_air_ground', 'dYdT_air_ground', 'dZdT_air_ground']].values
                                   - df_calc[['dXdT_bird_air', 'dYdT_bird_air', 'dZdT_bird_air']].values
                                         #- (df_calc[['wind_X', 'wind_Y', 'wind_Y']] @ np.diag([1, 1, 0])).values
                                   )

                current_closure = np.linalg.norm(current_closure, axis=1)
                loss = np.mean(current_closure[~np.isnan(current_closure)], axis=0)
            else:
                if 'air_velocity' in loss_function:
                    velocity_cols = ['dXdT_air_4', 'dYdT_air_4', 'dZdT_air_4']
                else:
                    velocity_cols = ['dXdT_bird_4', 'dYdT_bird_4', 'dZdT_bird_4']

                if 'horizontal' in loss_function:
                    velocity_cols = velocity_cols[:2]

                predictions = df_calc[velocity_cols].values
                predictions = predictions[na_mask]
                #predictions = np.linalg.norm(predictions, axis=-1)
                if 'horizontal_norm_median' == loss_function:
                    loss = np.linalg.norm(np.median(predictions, axis=0), axis=-1)
                elif 'horizontal_median_norm' == loss_function:
                    loss = np.median(np.linalg.norm(predictions, axis=-1), axis=0)
                elif 'mode' in loss_function:
                    counts, bins = np.histogram(predictions, bins=round(np.sqrt(len(predictions))))
                    n_max = np.argmax(counts)
                    loss = np.mean([bins[n_max], bins[n_max + 1]])
                elif 'std' in loss_function:
                    loss = np.std(predictions)
                else:
                    loss = np.mean(predictions)
        else:
            df_loss = pd.merge(df_calc, true_values, on=['bird_name', 'time'])
            df_loss['diff_X'] = df_loss['dXdT_air_4'] - df_loss['dXdT_air_rotation_real']
            df_loss['diff_Y'] = df_loss['dYdT_air_4'] - df_loss['dYdT_air_rotation_real']
            df_loss['diff_Z'] = 0 # df_loss['dZdT_air_3'] - df_loss['dZdT_air_thermal_real']
            df_loss.dropna(inplace=True)
            loss = np.median(
                np.linalg.norm(
                    df_loss[['diff_X', 'diff_Y', 'diff_Z']].values,
                    #predictions[na_mask, :2] - true_values[na_mask],
                    axis=-1))
        if np.isnan(loss):
            print('asd')
    except RuntimeWarning as e:
        print(e)

    return loss, N_NA, N_total

def early_stopping(i: int, list_of_metrics: np.ndarray, metric_threshold: float,
                   n_last_iterations: int, min_iter: int, max_iterations: int):
    if (len(list_of_metrics) < 2) or (i < min_iter):
        return False
    relative_changes = np.diff(list_of_metrics[-n_last_iterations - 1:]) / list_of_metrics[-n_last_iterations:]
    logger.debug(f'{relative_changes}')
    logger.debug(f'{relative_changes > -metric_threshold}')
    should_stop = (np.all( relative_changes > -metric_threshold)
                   and (i <= max_iterations))
    return should_stop