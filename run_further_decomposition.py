import os
from argparse import ArgumentParser
from copy import deepcopy

import dill as pickle

import numpy as np
import pandas as pd

import yaml

from calc.post_processing.air_velocity_field import get_thermal_and_wind_from_air_velocity_points, \
    get_stats_on_wind_and_thermal, get_thermal_rotation
from misc.auxiliar import sanitize_dict_for_yaml

save = True

from scipy.stats import PermutationMethod

from calc.auxiliar import UnivariateSplineWrapper
from data.get_data import load_decomposition_data, load_synthetic_data

parser = ArgumentParser()

parser.add_argument('-i', '--input', dest='path_to_decomposition', type=str,
                    help='path to the decomposition folder')
parser.add_argument('-ns', '--not-synthetic', dest='is_synthetic',
                    action='store_false', default=True,
                    help='whether or not to run the same for ground-truth dataset and comparative statistics')

pearson_kwargs = { 'method': PermutationMethod(n_resamples=9999,
                                               batch=9999,
                                               random_state=np.random.default_rng())
                   }
def main():
    args = parser.parse_args()
    path_to_decomposition = args.path_to_decomposition
    is_synthetic = args.is_synthetic
    path_to_save_stats = path_to_decomposition

    df_stats = pd.DataFrame()
    if save and is_synthetic:
        os.makedirs(path_to_save_stats, exist_ok=True)

    dec = load_decomposition_data(path_to_decomposition, iteration='best',
                                  list_of_files=['aggregated.csv',
                                                 'iterations.csv',
                                                 'splines.yaml',
                                                 'decomposition_args.yaml',
                                                 'inter_args.yaml'
                                                 ])
    path_to_save = {'decomposition': path_to_decomposition}


    df_dec = dec['iterations']

    df_dec['time'] = round(df_dec['time'], 3)


    thermal_core_ma_args = dec['decomposition_args']['thermal_core_ma_args']
    spline_args = dec['decomposition_args']['spline_parameters']
    spline_args['weight_normalization'] = True
    #
    thermal_inter_kwargs = deepcopy(dec['inter_args'].to_dict(orient='records')[0])
    thermal_inter_kwargs.pop('interpolator_class_name')
    dict_of_df = {'decomposition': df_dec}

    if is_synthetic:
        path_to_save['ground_truth'] =  os.path.join(path_to_decomposition, 'ground_truth_reconstructed')
        path_to_synthetic = dec['decomposition_args']['run_parameters']['input_folder']
        syn = load_synthetic_data(path_to_synthetic, list_of_object=['air.csv'])
        df_air = syn['air']

        thermal_core_positions = {coord: UnivariateSplineWrapper.from_tck(spline_parameters['tck'])
                                  for coord, spline_parameters in dec['splines'][0]['thermal_core_positions'].items()}
        thermal_core_velocities = {coord: UnivariateSplineWrapper.from_tck(spline_parameters['tck'])
                                   for coord, spline_parameters in dec['splines'][0]['thermal_core_velocities'].items()}
        df_gt_processed = df_air.copy()
        for coord in ['X', 'Y', 'Z']:
            df_gt_processed[f'{coord}_TC_ground'] = thermal_core_positions[f'{coord}_avg'](df_gt_processed['Z']) * int(
                coord != 'Z')
            df_gt_processed[f'{coord}_bird_TC'] = df_gt_processed[f'{coord}'] - df_gt_processed[
                f'{coord}_TC_ground'] * int(coord != 'Z')
        df_gt_processed['rho_bird_TC'] = np.linalg.norm(df_gt_processed[['X_bird_TC', 'Y_bird_TC']], axis=1)
        df_gt_processed['phi_bird_TC'] = np.arctan2(df_gt_processed['Y_bird_TC'], df_gt_processed['X_bird_TC'])

            # df_gt_processed[f'd{coord}dT_air_ground_avg'] = df_gt_processed[f'd{coord}dT_air_ground'] - df_gt_processed[f'd{coord}dT_turbulence_ground']

        df_gt_processed = df_gt_processed[['bird_name', 'time', 'X', 'Y', 'Z',
                                           'X_bird_TC', 'Y_bird_TC', 'Z_bird_TC', 'rho_bird_TC', 'phi_bird_TC',
                                           'dXdT_air_ground',
                                           'dYdT_air_ground',
                                           'dZdT_air_ground',
                                           'curvature_bird_air']].copy()
        dict_of_df['ground_truth'] =  df_gt_processed

    if save:
        [os.makedirs(p, exist_ok=True) for p in path_to_save.values()]


    return_dict = {}
    df_air = pd.read_csv(os.path.join(path_to_synthetic, 'air.csv'), index_col=False)
    return_dict['decomposition'] = []
    return_dict['ground_truth'] = []
    for data_type, current_df in dict_of_df.items():
        return_dict[data_type] = get_thermal_and_wind_from_air_velocity_points(current_df,
                                                                                   thermal_core_ma_args=dec['decomposition_args']['thermal_core_ma_args'],
                                                                                   spline_args=spline_args,
                                                                                   x_cols=[ 'X_bird_TC', 'Y_bird_TC', 'Z_bird_TC'],
                                                                                   v_columns = ['dXdT_air_ground',
                                                                                                'dYdT_air_ground',
                                                                                                'dZdT_air_ground'],
                                                                               curvature_col='curvature' if data_type == 'decomposition' else 'curvature_bird_air',
                                                                                   Z_col='Z',
                                                                               **thermal_inter_kwargs)
        return_dict[data_type] = list(return_dict[data_type])
        return_dict[data_type].append(get_thermal_rotation(return_dict[data_type][0], thermal_core_ma_args))

    if is_synthetic:
        df_thermal_merge = pd.merge(return_dict['ground_truth'][0], return_dict['decomposition'][0], on=['bird_name','time'], suffixes=('_dec_GT', '_dec'))
        df_thermal_merge = pd.merge(df_thermal_merge,
                                    df_air[['bird_name','time']
                                           + ['dXdT_thermal_ground',
                                              'dYdT_thermal_ground',
                                              'dZdT_thermal_ground',
                                              'wind_X',
                                              'wind_Y',
                                              'wind_Z',]].rename(columns={'dXdT_thermal_ground': 'dXdT_thermal_ground_GT',
                                                                          'dYdT_thermal_ground': 'dYdT_thermal_ground_GT',
                                                                          'dZdT_thermal_ground': 'dZdT_thermal_ground_GT',
                                                                          'wind_X': 'wind_X_GT',
                                                                          'wind_Y': 'wind_Y_GT',
                                                                          'wind_Z': 'wind_Z_GT'}),
                                    on=['bird_name','time'], how='left')
        df_compare = df_thermal_merge[['bird_name','time',
                                       'dXdT_thermal_ground_GT',
                                       'dXdT_thermal_ground_dec_GT',
                                       'dXdT_thermal_ground_dec',
                                       'dYdT_thermal_ground_GT',
                                       'dYdT_thermal_ground_dec_GT',
                                       'dYdT_thermal_ground_dec',
                                       'dZdT_thermal_ground_GT',
                                       'dZdT_thermal_ground_dec_GT',
                                       'dZdT_thermal_ground_dec',
                                       'wind_X_GT', 'wind_X_dec_GT', 'wind_X_dec',
                                       'wind_Y_GT', 'wind_Y_dec_GT','wind_Y_dec']]
        list_of_stats = get_stats_on_wind_and_thermal(df_compare, **pearson_kwargs)
        df_stats = pd.DataFrame.from_records(list_of_stats)


    if save:
        for data_type, (current_thermal, current_wind_spline, current_dict_of_interpolators, current_rotation) in return_dict.items():
            current_path_to_save = path_to_save[data_type]
            current_all_splines = [dec['splines'][0] | {'wind': current_wind_spline}]
            current_thermal['time'] = round(current_thermal['time'], 3)
            current_thermal.to_csv(os.path.join(current_path_to_save, f'thermal.csv', ))
            current_rotation.to_csv(os.path.join(current_path_to_save, f'rotation_binned.csv', ))
            with open(os.path.join(current_path_to_save, f'inter_kwargs.yaml'), 'w') as f:
                yaml.dump(sanitize_dict_for_yaml(thermal_inter_kwargs), f, default_flow_style=False)
            with open(os.path.join(current_path_to_save, f'decomposition_args.yaml'), 'w') as f:
                yaml.dump(sanitize_dict_for_yaml(dec['decomposition_args']), f, default_flow_style=False)

            with open(os.path.join(current_path_to_save, f'splines.yaml'), 'w') as f:
                yaml.dump(list(map(sanitize_dict_for_yaml, current_all_splines)), f, default_flow_style=False)

            if os.path.exists(os.path.join(path_to_save['decomposition'], f'interpolators.pkl')):
                with open(os.path.join(path_to_save['decomposition'], f'interpolators.pkl'), 'rb') as ef:
                    existing_interpolators = pickle.load(ef)
                    new_interpolators = existing_interpolators | current_dict_of_interpolators
                with open(os.path.join(current_path_to_save, f'interpolators.pkl'), 'wb') as f:
                    pickle.dump(new_interpolators, f)

            else:
                with open(os.path.join(current_path_to_save, f'interpolators.pkl'), 'wb') as f:
                    pickle.dump(current_dict_of_interpolators, f)


    if save and is_synthetic:
        df_stats.to_csv(os.path.join(path_to_save_stats, f'thermal_comparison.csv', ),
                        index=False)

if __name__ == '__main__':
    main()