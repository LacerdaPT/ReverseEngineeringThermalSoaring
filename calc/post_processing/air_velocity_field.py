import os
from copy import deepcopy

import dill as pickle
import yaml
from scipy.stats import cauchy, norm, laplace

from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame
from calc.stats import get_rms_and_pearson
from decomposition.auxiliar import calculate_decomposition_loss
from decomposition.post import decomposition_last_iteration
from misc.auxiliar import sanitize_dict_for_yaml

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, CloughTocher2DInterpolator
from scipy.spatial import ConvexHull
from calc.thermal import get_horizontal_thermal_velocity_by_moving_average, merge_dfs_with_spline
from data.get_data import load_decomposition_data


def process_jacobian_matrix(p, v_at_p, jacobian_matrix, max_extrapolated_distance):

    min_slope = np.abs((v_at_p - 0) / max_extrapolated_distance)
    thermal_x_at_p = v_at_p[0]
    thermal_y_at_p = v_at_p[1]
    thermal_z_at_p = v_at_p[2]
    jacobian_matrix_processed = jacobian_matrix.copy()
    for i_coord in range(3):
        dvxdi, dvydi, dvzdi = jacobian_matrix[:, i_coord]
        if np.all(np.abs(dvxdi) != 0):
            #dvdy = dvdy / np.abs(dvdy) * np.max([np.abs(dvdy), np.abs(min_slope)], axis=0)
            dvxdi = -np.sign(p[i_coord]) * np.sign(thermal_x_at_p) * np.max([np.abs(dvxdi), min_slope[0]])
            dvydi = -np.sign(p[i_coord]) * np.sign(thermal_y_at_p) * np.max([np.abs(dvydi), min_slope[1]])
            #dvzdi = -np.sign(p[i_coord]) * np.sign(thermal_z_at_p) * np.max([np.abs(dvzdi), min_slope[2]])
        else:
            dvxdi = -np.sign(p[i_coord]) * np.sign(thermal_x_at_p) * min_slope[0]
            dvydi = -np.sign(p[i_coord]) * np.sign(thermal_y_at_p) * min_slope[1]
            #dvzdi = -np.sign(p[i_coord]) * np.sign(thermal_z_at_p) * min_slope[2]
        jacobian_matrix_processed[:,i_coord] = [dvxdi, dvydi, dvzdi]

    return jacobian_matrix_processed
#
def is_in_hull(X, hull, tolerance):
    in_hull = np.apply_along_axis(lambda x: np.all(np.add(np.dot(x,
                         hull.equations[:, :-1].T),
                  hull.equations[:, -1]) <= tolerance,
           axis=-1), axis=-1, arr=X)
    return in_hull

def RBF_interpolate_extrapolate(df, X_cols, y_col, tolerance, hull=None, interpolator=None, **interpolator_kwargs):

    df_return = df[X_cols + y_col].copy()
    if hull is None:
        hull = ConvexHull(df_return.loc[~df_return[y_col].isna(), X_cols])

    in_hull = np.all(np.add(np.dot(df_return.loc[:, X_cols],
                                   hull.equations[:, :-1].T),
                            hull.equations[:, -1]) <= tolerance,
                     axis=1)

    df_return['in_hull'] = in_hull

    df_return['interpolated'] = df_return[y_col].isna() & df_return['in_hull']
    if interpolator is None:
        inter_rbf = RBFInterpolator(df_return.loc[~df_return[y_col].isna(), ['X_bird_TC_avg',
                                                                               'Y_bird_TC_avg']],
                                df_return.loc[~df_return[y_col].isna(), y_col], **interpolator_kwargs)


    interpolated_values = inter_rbf(df_return.loc[df_return['interpolated'], ['X_bird_TC', 'Y_bird_TC']])
    df_return.loc[df_return['interpolated'], y_col] = interpolated_values


def complete_decomposition_data(df_all_iterations, df_bins_all_iterations, df_splines_all_iterations,
                                min_occupation_number):
    df_all_iterations_post = pd.DataFrame()
    df_bins_all_iterations_post = pd.DataFrame()
    df_agg_all_iterations_post = pd.DataFrame()
    list_of_losses = []
    list_of_iterations = list(map(lambda x:x['iteration'], df_splines_all_iterations))
    for current_iteration in list_of_iterations:
        df = df_all_iterations[df_all_iterations['iteration'] == current_iteration].copy()
        df_bins = df_bins_all_iterations[
            df_bins_all_iterations['iteration'] == current_iteration].copy()
        current_spline = next(filter(lambda x: x['iteration'] == current_iteration,df_splines_all_iterations))

        # READ SPLINES

        # current_thermal_core_spline = UnivariateSplineWrapper.from_tck(current_spline['thermal_core_positions'][f'X_avg']['tck'])

        # FILTER DATA TO VALID REGION
        # df = df[df['Z'].between(current_thermal_core_spline.x_min, current_thermal_core_spline.x_max)]
        # df_bins = df_bins[(df_bins['Z_bird_TC_min'] >= current_thermal_core_spline.x_min)
        #                   & (df_bins['Z_bird_TC_max'] <= current_thermal_core_spline.x_max)]

        #df = df.sort_values(['Z']).dropna()

        #
        df['dXdT_thermal_TC'] = df['dXdT_thermal_ground'] - df['dXdT_TC_ground']
        df['dYdT_thermal_TC'] = df['dYdT_thermal_ground'] - df['dYdT_TC_ground']
        df['dZdT_thermal_TC'] = df['dZdT_thermal_ground'] - df['dZdT_TC_ground']
        df['curvature_abs'] = np.abs(df['curvature'])
        df_bins.rename(columns={'phi_bird_TC_avg': 'phi_bird_TC_bin_avg',
                                'phi_bird_TC_min': 'phi_bird_TC_bin_min',
                                'phi_bird_TC_max': 'phi_bird_TC_bin_max',
                                'rho_bird_TC_avg': 'rho_bird_TC_bin_avg',
                                'rho_bird_TC_min': 'rho_bird_TC_bin_min',
                                'rho_bird_TC_max': 'rho_bird_TC_bin_max',
                                'Z_bird_TC_avg': 'Z_bird_TC_bin_avg',
                                'Z_bird_TC_min': 'Z_bird_TC_bin_min',
                                'Z_bird_TC_max': 'Z_bird_TC_bin_max',
                                },
                       inplace=True)
        # CALCULATE LOCAL AVERAGES
        for metric in ['avg', 'min', 'max']:
            df_bins[f'X_bird_TC_bin_{metric}'] = df_bins[f'rho_bird_TC_bin_{metric}'] * np.cos(df_bins['phi_bird_TC_bin_avg'])
            df_bins[f'Y_bird_TC_bin_{metric}'] = df_bins[f'rho_bird_TC_bin_{metric}'] * np.sin(df_bins['phi_bird_TC_bin_avg'])


        bin_cols = ['bin_index_rho_bird_TC',
                    'bin_index_phi_bird_TC',
                    'bin_index_Z_bird_TC']
        df['in_bin'] = (df[bin_cols[0]] != -1) & (df[bin_cols[1]] != -1) & (df[bin_cols[2]] != -1)
        agg_dict = {}
        for coord in ['X', 'Y', 'Z']:
            for stat in ['mean', 'std', 'count']:
                stat_name = stat.replace("mean", "avg")
                for reference_frame in['ground', 'TC']:
                    agg_dict[f'd{coord}dT_thermal_{reference_frame}_bin_{stat_name}'] = (f'd{coord}dT_thermal_{reference_frame}', stat)
                agg_dict[f'd{coord}dT_TC_ground_bin_{stat_name}'] = (f'd{coord}dT_TC_ground', stat)
                agg_dict[f'{coord}_bird_TC_bin_{stat_name}'] = (f'{coord}_bird_TC', stat)
        agg_dict['curvature_bin_max'] = ('curvature_abs', 'max')
        agg_dict['curvature_bin_avg'] = ('curvature_abs', 'mean')
        agg_dict['curvature_bin_std'] = ('curvature_abs', 'std')
        agg_dict['curvature_bin_count'] = ('curvature_abs', 'count')

        df_thermal_bin_avg = df.loc[df['in_bin'] & (df['curvature_abs'] < 0.1), bin_cols +  [v[0] for v in agg_dict.values() if v[1] == 'mean']].groupby(bin_cols
                                                                                      ).agg(**agg_dict).reset_index()
        # df_thermal_bin_avg = df_thermal_bin_avg[df_thermal_bin_avg['in_bin']]

        df_agg = df_bins.copy()

        common_cols = list(set(df_thermal_bin_avg.columns) & set(df_agg.columns))
        [common_cols.remove(c) for c in bin_cols if c in common_cols]
        #df_agg = df_agg[df_agg['dXdT_air_TC_bin_count'] >= min_occupation_number]
        df_agg = pd.merge(df_agg.drop(columns=common_cols), df_thermal_bin_avg, on=bin_cols, how='left')
        # CALCULATE RESIDUALS
        common_cols = list(set(df.columns) & set(df_agg.columns))
        [common_cols.remove(c) for c in bin_cols + ['iteration']]
        df = pd.merge(df.drop(columns=common_cols), df_agg, on=bin_cols + ['iteration'], how='left')

        for reference_frame in ['ground', 'TC']:

            df[f'V_horizontal_{reference_frame}'] = np.linalg.norm(df[[f'dXdT_thermal_{reference_frame}',
                                                                       f'dYdT_thermal_{reference_frame}']], axis=1)
            df_agg[f'V_horizontal_{reference_frame}_agg'] = np.linalg.norm(df_agg[[f'dXdT_thermal_{reference_frame}_bin_avg',
                                                                                   f'dYdT_thermal_{reference_frame}_bin_avg']], axis=1)

        df['epsilon'] = np.linalg.norm(df[['dXdT_bird_ground', 'dYdT_bird_ground', 'dZdT_bird_ground']].values
                                       - df[['dXdT_air_ground', 'dYdT_air_ground', 'dZdT_air_ground']].values
                                       - df[['dXdT_bird_air', 'dYdT_bird_air', 'dZdT_bird_air']].values,
                                       axis=1
                                       )

        current_loss, N_NA, N_total = calculate_decomposition_loss(current_iteration=df,
                                                          z_limits=[np.min(current_spline['wind']['wind_X']['tck'][0]),
                                                                    np.max(current_spline['wind']['wind_X']['tck'][0])],
                                                          max_allowed_curvature=0.1,
                                                          loss_function='closure')

        list_of_losses.append([current_iteration, current_loss, N_NA, N_total])
        df_agg['iteration'] = current_iteration
        df_all_iterations_post = pd.concat([df_all_iterations_post, df])
        df_bins_all_iterations_post = pd.concat([df_bins_all_iterations_post, df_bins])
        df_agg_all_iterations_post = pd.concat([df_agg_all_iterations_post, df_agg])


    df_all_iterations_post['epsilon'] = np.linalg.norm(df_all_iterations_post[['dXdT_bird_ground',
                                                                               'dYdT_bird_ground',
                                                                               'dZdT_bird_ground']].values
                                                       - df_all_iterations_post[['dXdT_air_ground',
                                                                                 'dYdT_air_ground',
                                                                                 'dZdT_air_ground']].values
                                                       - df_all_iterations_post[['dXdT_bird_air',
                                                                                 'dYdT_bird_air',
                                                                                 'dZdT_bird_air']].values,
                                                       axis=1
                                                       )
    df_losses = pd.DataFrame(list_of_losses, columns=['iteration', 'loss', 'N_NA', 'N_total'])
    return df_all_iterations_post, df_bins_all_iterations_post, df_agg_all_iterations_post, df_losses

def reconstruct(df_all_iterations: pd.DataFrame, df_agg_all_iterations: pd.DataFrame,
                list_of_cols: list[str], min_occupation_number:int, rho_quantile: float,
                **inter_kwargs)-> (pd.DataFrame,pd.DataFrame,pd.DataFrame, list[ConvexHull], list[dict], list[dict]):
    df_all_iterations_reconstructed = pd.DataFrame()

    df_agg_all_iterations_reconstructed = pd.DataFrame()
    list_of_iterations = df_agg_all_iterations['iteration'].unique()
    list_of_interpolators = []
    list_of_wind_splines = []
    list_of_losses = []
    for i in list_of_iterations:
        current_df_agg = df_agg_all_iterations[df_agg_all_iterations['iteration'] == i].copy()
        current_df = df_all_iterations[df_all_iterations['iteration'] == i].copy()
        (current_df_ret,
         current_df_agg_ret,
         current_interpolator_dict) = reconstruct_single_iteration(current_df, current_df_agg,
                                                                   rho_quantile=rho_quantile,
                                                                   list_of_cols=list_of_cols,
                                                                   min_occupation_number=min_occupation_number,
                                                                   **inter_kwargs)

        current_loss, N_NA, N_total = calculate_decomposition_loss(current_iteration=current_df_ret,
                                                          z_limits=[np.min(current_df_agg['Z_bird_TC_bin_avg']),
                                                                    np.max(current_df_agg['Z_bird_TC_bin_avg'])],
                                                          max_allowed_curvature=0.1,
                                                          loss_function='closure')


        thermal_inter_kwargs = deepcopy(inter_kwargs)
        # thermal_inter_kwargs['neighbors'] = 0
        current_df_ret.drop(columns=['dXdT_thermal_ground',
                                     'dYdT_thermal_ground',
                                     'dZdT_thermal_ground',
                                     'dXdT_thermal_ground_avg',
                                     'dYdT_thermal_ground_avg',
                                     'dZdT_thermal_ground_avg']
                                    + ['wind_X', 'wind_Y'],
                            errors='ignore', inplace=True)
        thermal_inter_kwargs['smoothing'] = 0.0
        list_of_losses.append([i, current_loss, N_NA, N_total])
        list_of_interpolators.append(current_interpolator_dict)
        df_all_iterations_reconstructed = pd.concat([df_all_iterations_reconstructed, current_df_ret])
        df_agg_all_iterations_reconstructed = pd.concat([df_agg_all_iterations_reconstructed, current_df_agg_ret])
    df_loss = pd.DataFrame(list_of_losses, columns=['iteration', 'loss', 'N_NA', 'N_total'])

    return (df_all_iterations_reconstructed, df_agg_all_iterations_reconstructed, df_loss,
            list_of_interpolators, list_of_wind_splines)



def reconstruct_single_iteration(df, df_agg, list_of_cols, min_occupation_number, rho_quantile: float=90,
                                 **inter_kwargs):


    interpolator_dict = {}

    interpolator_kwargs = deepcopy(inter_kwargs)
    interpolator_class = interpolator_kwargs.pop('interpolator_class', RBFInterpolator)
    bin_cols = ['bin_index_rho_bird_TC',
                'bin_index_phi_bird_TC',
                'bin_index_Z_bird_TC']
    xyz_na_mask = np.any(np.isnan(df[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC']]), axis=1)
    xyz_na_mask_agg = np.any(np.isnan(df[['X_bird_TC_bin_avg', 'Y_bird_TC_bin_avg', 'Z_bird_TC_bin_avg']]), axis=1)
    v_na_mask = np.any(np.isnan(df[list_of_cols]), axis=1)
    v_na_mask_agg = np.any(np.isnan(df_agg[[f'{col}_bin_avg' for col in list_of_cols]]), axis=1)

    # Get Convex Hull

    delta_z = 20
    z_rho_array = []
    delta_phi = np.pi / 8
    for z_min in np.arange(df['Z'].min(), df['Z'].max() + delta_z, delta_z):
        for phi_min in np.arange(-np.pi, np.pi, delta_phi):
            current_slice = df[df['Z_bird_TC'].between(z_min, z_min + delta_z)
                                   & df['phi_bird_TC'].between(phi_min, phi_min + delta_phi)
                                   & df['in_bin']]
            if current_slice.size < 10:
                continue
            z_rho_array.append([z_min, phi_min, np.percentile( current_slice['rho_bird_TC'], rho_quantile)])

    df_rho_max = pd.DataFrame(z_rho_array, columns=['Z', 'phi', 'rho_max'])

    df_extra = df_rho_max[df_rho_max['phi'] == df_rho_max['phi'].min()].copy()
    df_extra['phi'] = np.pi
    df_rho_max = pd.concat([df_rho_max, df_extra])
    df_rho_max.drop_duplicates(subset=['Z', 'phi'], keep='first', inplace=True, ignore_index=True)
    df_rho_max.sort_values(['Z', 'phi'], inplace=True)
    rho_spline = CloughTocher2DInterpolator(df_rho_max[['Z', 'phi']].values, df_rho_max['rho_max'])
    interpolator_dict['valid_region'] = rho_spline
    # ===========    Check in hull      =========== #

    # Iterations
    df['in_hull'] = df['rho_bird_TC'] <= rho_spline(df[['Z_bird_TC', 'phi_bird_TC']])

    df_agg['in_hull'] = df_agg['rho_bird_TC_bin_avg'] <= rho_spline(df_agg[['Z_bird_TC_bin_avg', 'phi_bird_TC_bin_avg']])

    df['interpolated'] = (xyz_na_mask
                          | v_na_mask
                          | (~df['in_hull'])
                          | (np.abs(df['curvature']) > 0.1))
    df_agg['interpolated'] = (xyz_na_mask_agg
                              | v_na_mask_agg
                              | (~df_agg['in_hull'])
                              | (np.abs(df_agg['curvature_bin_avg']) > 0.1))
    for current_col in list_of_cols:
        current_col_bin_avg = f'{current_col}_bin_avg'
        current_col_avg = f'{current_col}_avg'

        # ======================    Get interpolator    ======================#
        df_agg_to_fit = df_agg[df_agg[f'{current_col}_bin_count'] >= min_occupation_number]

        df_agg_to_fit = df_agg_to_fit[~df_agg_to_fit['interpolated']]
        x_to_fit = df_agg_to_fit[['X_bird_TC_bin_avg', 'Y_bird_TC_bin_avg', 'Z_bird_TC_bin_avg']]
        y_to_fit = df_agg_to_fit[current_col_bin_avg]

        interpolator_instance = interpolator_class(x_to_fit,y_to_fit, **interpolator_kwargs)
        # ====================== Interpolate aggregated data in the hull and extrapolate outside of the hull


        x_to_interpolate = df[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC']]
        df[current_col_avg] = interpolator_instance(x_to_interpolate)

        x_agg_to_interpolate = df_agg[['X_bird_TC_bin_avg', 'Y_bird_TC_bin_avg', 'Z_bird_TC_bin_avg']]
        df_agg[current_col_avg] = interpolator_instance(x_agg_to_interpolate)
        interpolator_dict[current_col] = interpolator_instance

    for coord in ['X', 'Y', 'Z']:
        df[f'd{coord}dT_air_ground_avg'] = df[f'd{coord}dT_air_TC_avg'] + df[f'd{coord}dT_TC_ground']
        df[f'epsilon_{coord}'] = (df[f'd{coord}dT_bird_ground'].values
                                  - df[f'd{coord}dT_air_ground_avg'].values
                                  - df[f'd{coord}dT_bird_air'].values)
        df[f'd{coord}dT_air_ground'] = df[f'd{coord}dT_air_ground_avg'] + df[f'epsilon_{coord}']

    df['epsilon'] = np.linalg.norm(df[['epsilon_X', 'epsilon_Y', 'epsilon_Z']], axis=1)
    return df, df_agg, interpolator_dict


def postprocess_decomposition(path_to_decomposition, path_to_save, min_occupation_number, rho_quantile, inter_kwargs,
                              subfolders=None):
    inter_kwargs = deepcopy(inter_kwargs)
    save = bool(path_to_save)
    if subfolders is None:
        subfolders = ['post', 'reconstructed']
    if save:
        os.makedirs(os.path.join(path_to_save, subfolders[0]), exist_ok=True)
        os.makedirs(os.path.join(path_to_save, subfolders[1]), exist_ok=True)

    try:

        dec = load_decomposition_data(path_to_decomposition, list_of_files=['iterations.csv',
                                                                            'bins.csv',
                                                                            'losses.csv',
                                                                            'splines.yaml',
                                                                            'decomposition_args.yml',
                                                                            'decomposition_args.yaml',
                                                                            'aerodynamic_parameters.csv'],
                                      iteration='best'
                                      )

        dec_post = deepcopy(dec)

        (dec_post['iterations'],
         dec_post['bins'],
         dec_post['aggregated'],
         dec_post['losses']) = complete_decomposition_data(dec['iterations'], dec['bins'], dec['splines'],
                                        min_occupation_number=min_occupation_number)
        del dec

        if save:
            for filename, current_df in dec_post.items():
                if isinstance(current_df, pd.DataFrame):
                    current_df.to_csv(os.path.join(path_to_save,subfolders[0],
                                                          f'{filename}.csv', ))
                elif isinstance(current_df, dict):
                    with open(os.path.join(path_to_save, subfolders[0], f'{filename}.yaml'), 'w') as f:
                        yaml.dump(sanitize_dict_for_yaml(current_df), f, default_flow_style=False)
                else:
                    with open(os.path.join(path_to_save, subfolders[0], f'{filename}.yaml'), 'w') as f:
                        yaml.dump(list(map(sanitize_dict_for_yaml, current_df)), f, default_flow_style=False)
    except Exception as e:
        raise Exception(str(e))

    try:
        dec_reconstructed = deepcopy(dec_post)
        (dec_reconstructed['iterations'],
         dec_reconstructed['aggregated'],
         dec_reconstructed['losses'],
         list_of_interpolators,
         list_of_wind_splines) = reconstruct(dec_post['iterations'],
                                             dec_post['aggregated'],
                                             min_occupation_number=min_occupation_number,
                                             rho_quantile=rho_quantile,
                                             list_of_cols=[f'd{coord}dT_air_TC'
                                                           for coord in ['X', 'Y', 'Z']],
                                             **inter_kwargs)
    except Exception as e:
        raise Exception("Failed to reconstruct - " + path_to_save)

    print(path_to_save)
    if save:
        if 'interpolator_class' in inter_kwargs:
            inter_kwargs['interpolator_class_name'] = str(inter_kwargs['interpolator_class'])
        else:
            inter_kwargs['interpolator_class_name'] = 'RBFInterpolator'
        with open(os.path.join(path_to_save, subfolders[1], f'inter_args.yaml'), 'w') as f:
            yaml.dump(sanitize_dict_for_yaml(inter_kwargs), f, default_flow_style=False)


        for filename, current_df in dec_reconstructed.items():
            if isinstance(current_df, pd.DataFrame):
                current_df.to_csv(os.path.join(path_to_save,subfolders[1],
                                                      f'{filename}.csv', ))
            elif isinstance(current_df, dict):
                with open(os.path.join(path_to_save, subfolders[1], f'{filename}.yaml'), 'w') as f:
                    yaml.dump(sanitize_dict_for_yaml(current_df), f, default_flow_style=False)
            else:
                with open(os.path.join(path_to_save, subfolders[1], f'{filename}.yaml'), 'w') as f:
                    yaml.dump(list(map(sanitize_dict_for_yaml, current_df)), f, default_flow_style=False)

        with open(os.path.join(path_to_save, subfolders[1], f'interpolators.pkl'), 'wb') as f:
            pickle.dump(list_of_interpolators[0], f)
    return dec_post, dec_reconstructed | {'interpolators': list_of_interpolators}



def get_thermal_and_wind_from_air_velocity_points(df: pd.DataFrame, spline_args: dict,
                                                  thermal_core_ma_args: dict, Z_col: str ='Z',
                                                  x_cols:list[str] =None,
                                                  v_columns: list[str] = None,
                                                  curvature_col: str = None,
                                                  wind_spline: dict = None, **inter_kwargs
                                                  ) -> (pd.DataFrame, dict, dict):
    """

    :rtype: tuple(pd.DataFrame, dict, list)
    """


    interpolator_kwargs = deepcopy(inter_kwargs)
    interpolator_class = interpolator_kwargs.pop('interpolator_class', RBFInterpolator)
    if v_columns is None:
        v_columns = ['dXdT_air_ground',
                     'dYdT_air_ground',
                     'dZdT_air_ground']
    if curvature_col is None:
        curvature_col = 'curvature'
    if x_cols is None:
        x_cols = ['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC']
    if wind_spline is None:

        df_mean_wind = get_horizontal_thermal_velocity_by_moving_average(df[np.abs(df[curvature_col]) < 0.1] ,
                                                                         window=thermal_core_ma_args['window'],
                                                                         min_periods=thermal_core_ma_args['min_periods'],
                                                                         Z_col=Z_col, V_H_cols=v_columns[:2]
                                                                         )

        df, wind_spline = merge_dfs_with_spline(df.drop(columns=['wind_X', 'wind_Y'], errors='ignore'),
                                                df_mean_wind,
                                                merge_on=Z_col,
                                                other_cols_to_merge={'wind_X': 'wind_X_sem',
                                                                     'wind_Y': 'wind_Y_sem', },
                                                # if min_occupation_number > 1 else {'wind_X': 'wind_X_sem', 'wind_Y': 'wind_Y_sem', }
                                                other_merge_on=f'{Z_col}_avg',
                                                spline_degree=spline_args['degree'],
                                                smoothing_factor=spline_args['smoothing_factor'],
                                                extrapolate=spline_args['extrapolate'],
                                                weight_normalization=False, #spline_args['weight_normalization'],
                                                debug=False)
    else:
        for col in wind_spline.keys():
            df[col] = wind_spline[col](df[Z_col].values, extrapolate=spline_args['extrapolate'])
    df['wind_Z'] = 0.0
    df['dXdT_thermal_ground'] = df[v_columns[0]] - df['wind_X']         # 'dXdT_air_ground' - 'wind_X'
    df['dYdT_thermal_ground'] = df[v_columns[1]] - df['wind_Y']         # 'dYdT_air_ground' - 'wind_Y'
    df['dZdT_thermal_ground'] = df[v_columns[2]] - df['wind_Z']         # 'dZdT_air_ground' - 'wind_Z'
    dict_of_interpolators = {}
    xyz_na_mask = np.any(np.isnan(df[x_cols]), axis=1)
    bad_curvature_mask = np.abs(df[curvature_col]) > 0.1
    for i_coord, coord in enumerate(['X', 'Y', 'Z']):
        current_col = f'd{coord}dT_thermal_ground'

        df[f'interpolated_thermal_{coord}'] = df[current_col].isna() | xyz_na_mask | bad_curvature_mask

        df_to_fit = df[~df[f'interpolated_thermal_{coord}']]
        x_to_fit = df_to_fit[x_cols]
        y_to_fit = df_to_fit[current_col]

        interpolator_instance = interpolator_class(x_to_fit, y_to_fit, **interpolator_kwargs)
        df.loc[df[f'interpolated_thermal_{coord}'], current_col] = interpolator_instance(df.loc[df[f'interpolated_thermal_{coord}'], x_cols])
        dict_of_interpolators[current_col] = interpolator_instance

    return df, wind_spline, dict_of_interpolators

def get_thermal_rotation(df, thermal_core_ma_args, bin_size=1):
    df[[f'V_rho_rotating_thermal_ground', f'V_phi_rotating_thermal_ground']] = df[
        ['X_bird_TC', 'Y_bird_TC',
         f'dXdT_thermal_ground',
         f'dYdT_thermal_ground'
         ]].apply(lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
                  axis=1, result_type='expand')

    df_rotation = pd.DataFrame()

    df['interpolated_thermal'] = np.any(
        df[['interpolated_thermal_X', 'interpolated_thermal_Y', 'interpolated_thermal_Z']].values, axis=1)
    df = df[~df['interpolated_thermal']]
    if 'in_hull' in df.columns:
        df = df[df['in_hull']]

    bins = np.arange(0, df['rho_bird_TC'].max() + bin_size, bin_size)
    df['bin_index_rotation'] = np.digitize(df['rho_bird_TC'].values, bins=bins,)
    df_rotation = df.groupby('bin_index_rotation').agg(**{f'{col}_{current_stat}': (col, current_stat)
                                          for col in ['rho_bird_TC','V_phi_rotating_thermal_ground']
                                          for current_stat in ['mean', 'std', 'count', 'sem']})
    # for current_stat in ['mean', 'std', 'count', 'sem']:
    #     df_rotation[[f'rho_bird_TC_{current_stat}',
    #                  f'V_phi_rotating_thermal_ground_{current_stat}']] = df.sort_values('rho_bird_TC'
    #                                                                                     )[['rho_bird_TC',
    #                                                                                        'V_phi_rotating_thermal_ground']].rolling(
    #
    #         window=thermal_core_ma_args['window'],
    #         min_periods=thermal_core_ma_args['min_periods'],
    #         center=thermal_core_ma_args['center'],
    #     ).agg(current_stat)
    # df_rotation.dropna(axis=0, how='any', ignore_index=True, inplace=True)
    return df_rotation

def full_postprocessing_pipeline(path_to_decomposition, min_occupation_number, rho_quantile, inter_kwargs,
                                 final_inter_kwargs,
                                 binning_parameters, save):

    ####################################################################################################################
    ########################################            POST PROCESSING             ####################################
    ####################################################################################################################
    _, _ = postprocess_decomposition(path_to_decomposition,
                                     min_occupation_number=min_occupation_number,
                                     rho_quantile=rho_quantile,
                                     inter_kwargs=inter_kwargs,
                                     path_to_save=path_to_decomposition if save else '',
                                     subfolders=['post', 'reconstructed']
                                     )

    ####################################################################################################################
    ########################################          FINAL DECOMPOSITION           ####################################
    ####################################################################################################################

    dec = load_decomposition_data(os.path.join(path_to_decomposition, 'reconstructed'), iteration='best',
                                  list_of_files=['iterations.csv',
                                                 'aerodynamic_parameters.csv',
                                                 'decomposition_args.yaml'
                                                 ]
                                  )
    path_to_decomposition_yaml = os.path.join(path_to_decomposition, 'reconstructed', 'decomposition_args.yaml')
    current_iteration = dec['iterations'].copy()
    current_physical_parameters = deepcopy( dec['aerodynamic_parameters'])
    current_decomposition_args =  deepcopy( dec['decomposition_args'])
    current_decomposition_args['spline_parameters']['weight_normalization'] = True
    current_decomposition_args['binning_parameters'] = binning_parameters
    del dec
    path_to_final_decomposition = os.path.join(path_to_decomposition, 'final')
    os.makedirs(path_to_final_decomposition, exist_ok=True)
    current_physical_parameters.drop(columns=['debug', 'iteration'], errors='ignore', inplace=True)
    if np.all(current_physical_parameters['wing_loading'].isna().values):
        current_physical_parameters['wing_loading'] = current_physical_parameters['mass'] / current_physical_parameters['wing_area']

    (current_iteration, current_thermal_core, current_physical_parameters, current_bins, current_spline_stats,
     list_of_losses) = decomposition_last_iteration(current_iteration, current_physical_parameters,
                                                    path_to_decomposition_yaml,
                                                    binning_parameters=deepcopy(binning_parameters))
    current_thermal_core['iteration'] = 1
    current_physical_parameters['iteration'] = 1
    current_iteration['iteration'] = 1
    current_bins['iteration'] = 1
    df_loss = pd.DataFrame([list_of_losses], columns=['loss', 'N_NA', 'N_total'])
    df_loss['iteration'] = 1
    df_loss.to_csv(f'{path_to_final_decomposition}/losses.csv')
    df_spline_stats = pd.DataFrame.from_records([current_spline_stats], ).copy()
    df_spline_stats['iteration'] = 1
    list_of_splines = [row.to_dict() for _, row in df_spline_stats.iterrows()]
    if save:
        current_thermal_core.to_csv(f'{path_to_final_decomposition}/thermal_core.csv') #, index=False)
        current_physical_parameters.to_csv(f'{path_to_final_decomposition}/aerodynamic_parameters.csv') #, index=False)
        current_iteration.to_csv(f'{path_to_final_decomposition}/iterations.csv') #, index=False)
        current_bins.to_csv(f'{path_to_final_decomposition}/bins.csv') #, index=False)
        with open(os.path.join(path_to_final_decomposition, 'splines.yaml'), 'w') as f:
            yaml.dump(list(map(sanitize_dict_for_yaml, list_of_splines)), f, default_flow_style=False)
        with open(os.path.join(path_to_final_decomposition, 'decomposition_args.yaml'), 'w') as f:
            yaml.dump(sanitize_dict_for_yaml(current_decomposition_args), f, default_flow_style=False)

    ####################################################################################################################
    ########################################            POST PROCESSING             ####################################
    ####################################################################################################################
    dec_final_post, dec_final_reconstruted = postprocess_decomposition(path_to_decomposition=path_to_final_decomposition,
                                                                       min_occupation_number=1,
                                                                       rho_quantile=rho_quantile,
                                                                       inter_kwargs=final_inter_kwargs,
                                                                       path_to_save=path_to_final_decomposition,
                                                                       subfolders=['post',
                                                                                   'reconstructed']
                                                                       )



def get_stats_on_wind_and_thermal(df_compare, **kwargs):

    list_of_stats = []
    for i_coord, coord in enumerate(['X', 'Y', 'Z']):
        for i_col1, col1 in enumerate([f'd{coord}dT_thermal_ground_GT',
                                       f'd{coord}dT_thermal_ground_dec_GT',
                                       f'd{coord}dT_thermal_ground_dec']):
            for i_col2, col2 in enumerate([f'd{coord}dT_thermal_ground_GT',
                                           f'd{coord}dT_thermal_ground_dec_GT',
                                           f'd{coord}dT_thermal_ground_dec'][i_col1 + 1:]):
                r = get_rms_and_pearson(df_compare[col1].values, df_compare[col2].values,**kwargs
                                        )
                current_rms, current_correlation, current_p_value, current_n = r
                diff = df_compare[col1].values - df_compare[col2].values
                diff = diff[(~np.isnan(diff)) & (diff != np.inf)]
                u_cauchy, variance_cauchy = cauchy.fit(diff)
                u_norm, variance_norm = norm.fit(diff)
                u_laplace, variance_laplace = laplace.fit(diff)
                # u, variance = current_normal.stats
                sigma_cauchy = np.sqrt(variance_cauchy)
                sigma_norm = np.sqrt(variance_norm)
                sigma_laplace = np.sqrt(variance_laplace)
                list_of_stats.append({'col1':         col1, 'col2': col2,
                                      'rms':          current_rms,
                                      'correlation':  current_correlation,
                                      'p_value':      current_p_value,
                                      'n':            current_n,
                                      'mean':         np.mean(diff),
                                      'sigma':        np.std(diff, ddof=1),
                                      'mean_cauchy':  u_cauchy, 'sigma_cauchy': sigma_cauchy,
                                      'mean_norm':    u_norm, 'sigma_norm': sigma_norm,
                                      'mean_laplace': u_laplace, 'sigma_laplace': sigma_laplace
                                      })

    for i_coord, coord in enumerate(['X', 'Y']):
        for i_col1, col1 in enumerate([f'wind_{coord}_GT', f'wind_{coord}_dec', f'wind_{coord}_dec_GT']):
            for i_col2, col2 in enumerate([f'wind_{coord}_GT', f'wind_{coord}_dec', f'wind_{coord}_dec_GT'][i_col1 + 1:]):
                r = get_rms_and_pearson(df_compare[col1].values, df_compare[col2].values, **kwargs
                                        )
                current_rms, current_correlation, current_p_value, current_n = r
                diff = df_compare[col1].values - df_compare[col2].values
                diff = diff[(~np.isnan(diff)) & (diff != np.inf)]
                u_cauchy, variance_cauchy = cauchy.fit(diff)
                u_norm, variance_norm = norm.fit(diff)
                u_laplace, variance_laplace = laplace.fit(diff)
                # u, variance = current_normal.stats
                sigma_cauchy = np.sqrt(variance_cauchy)
                sigma_norm = np.sqrt(variance_norm)
                sigma_laplace = np.sqrt(variance_laplace)
                list_of_stats.append({'col1':         col1, 'col2': col2,
                                      'rms':          current_rms,
                                      'correlation':  current_correlation,
                                      'p_value':      current_p_value,
                                      'n':            current_n,
                                      'mean':         np.mean(diff),
                                      'sigma':        np.std(diff, ddof=1),
                                      'mean_cauchy':  u_cauchy, 'sigma_cauchy': sigma_cauchy,
                                      'mean_norm':    u_norm, 'sigma_norm': sigma_norm,
                                      'mean_laplace': u_laplace, 'sigma_laplace': sigma_laplace
                                      })
    return list_of_stats