import warnings
from copy import deepcopy
from types import LambdaType

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    import pandas as pd

from calc.auxiliar import UnivariateSplineWrapper, rolling_apply_multiple_columns
from calc.stats import weights_preprocessing, get_weighted_average, \
    get_maximum_vertical_velocity2d, get_maximum_by_quadratic_fit, get_rolling_basic_statistics, get_mode_with_KDE


def get_maximum_vertical_velocity_by_moving_average(df_iteration, independent_var_col, vz_col, Z_col, window,
                                                    min_periods, n_sigma=0, debug=False):
    n_dim = len(independent_var_col)
    df_calc = df_iteration.sort_values(Z_col)
    if n_dim == 2:
        x_col, y_col = independent_var_col
        array_vz_max = rolling_apply_multiple_columns(get_maximum_vertical_velocity2d,
                                                      window, min_periods,
                                                      df_calc[x_col].values,
                                                      df_calc[y_col].values,
                                                      df_calc[vz_col].values,
                                                      df_calc[Z_col].values, n_sigma=n_sigma, debug=debug,
                                                      derivative_epsilon=1e-3
                                                      )

        df_vz_max = pd.DataFrame(array_vz_max, columns=['X_max',
                                                        'Y_max',
                                                        'dZdT_air_max',
                                                        'e_X_max',
                                                        'e_Y_max',
                                                        'e_dZdT_air_max',
                                                        'Z_avg',
                                                        'fit_coefficients',
                                                        'fit_pcov',
                                                        'N']
                                 )
        df_vz_max['X_max'] = df_vz_max['X_max'].astype(float)
        df_vz_max['Y_max'] = df_vz_max['Y_max'].astype(float)
        df_vz_max['dZdT_air_max'] = df_vz_max['dZdT_air_max'].astype(float)
        df_vz_max['e_X_max'] = df_vz_max['e_X_max'].astype(float)
        df_vz_max['e_Y_max'] = df_vz_max['e_Y_max'].astype(float)
        df_vz_max['e_dZdT_air_max'] = df_vz_max['e_dZdT_air_max'].astype(float)
        df_vz_max['Z_avg'] = df_vz_max['Z_avg'].astype(float)
        df_vz_max['N'] = df_vz_max['N'].astype(int)
    else:
        rho_col = independent_var_col
        array_vz_max = rolling_apply_multiple_columns(get_maximum_vertical_velocity,
                                                      window, min_periods,
                                                      df_calc[rho_col].values,
                                                      df_calc[vz_col].values,
                                                      df_calc[Z_col].values, n_sigma=n_sigma, debug=debug)

        df_vz_max = pd.DataFrame(array_vz_max,
                                         columns=['rho_max',
                                                  'dZdT_air_max',
                                                  'Z_avg',
                                                  'fit_coefficients',
                                                  'fit_pcov',
                                                  'N']
                                         )
        df_vz_max['rho_max'] = df_vz_max['x_max'].astype(float)
        df_vz_max['dZdT_air_max'] = df_vz_max['dZdT_air_max'].astype(float)
        df_vz_max['Z_avg'] = df_vz_max['Z_avg'].astype(float)
        df_vz_max['N'] = df_vz_max['N'].astype(int)

    # It's likely that there are several duplicates is the point that gets in and the point that gets out of the window
    # is filtered out and therefore has no consequence.
    df_vz_max.dropna(subset=['dZdT_air_max'], inplace=True)
    df_vz_max.drop_duplicates(subset=['dZdT_air_max', 'Z_avg'], inplace=True, ignore_index=True)

    return df_vz_max


def get_wind_from_thermal_core(df, vx_ground_col='dXdT_ground_avg',
                               vy_ground_col='dYdT_ground_avg',
                               vz_ground_col='dZdT_air_0_avg',
                               vz_air_max_col='dZdT_air_max',
                               x_col='X_avg', y_col='Y_avg', z_col='Z_avg',
                               sort_on='Z_avg',
                               method='velocities',
                               spline_parameters=None,
                               smoothing_ma_args=None,
                               debug=False):
    if smoothing_ma_args is None:
        smoothing_ma_args = {'window': 720,
                             'win_type': 'gaussian',
                             'window_args': {'std': 300},
                             'min_periods': 1,
                             'center': True}
    else:
        smoothing_ma_args = smoothing_ma_args.copy()

    if 'window_args' in smoothing_ma_args.keys():
        window_args = smoothing_ma_args.pop('window_args')
    else:
        window_args = {}

    df_calc = df.copy()
    df_calc = df_calc.drop_duplicates(subset=['Z_avg'], ignore_index=True)
    df_calc.sort_values('Z_avg', inplace=True)

    if method == 'velocities':
        df_calc['wind_X_raw'] = df_calc[vx_ground_col] / df_calc[vz_ground_col] * df_calc[vz_air_max_col]  # dXdt = dXdZ . dZdT
        df_calc['wind_Y_raw'] = df_calc[vy_ground_col] / df_calc[vz_ground_col] * df_calc[vz_air_max_col]
    else:
        N = df_calc['Z_avg'].size
        if spline_parameters is None:
            spline_parameters = {'degree': 1,
                                 'smoothing_factor': 0.1
                                 }
        for i_col, coord_col in enumerate([x_col, y_col]):

            weights = df_calc['X_sem'].values if i_col == 0 else df_calc['Y_sem'].values

            weights = 1 / weights

            merging_spline = UnivariateSplineWrapper(df_calc[z_col].values,
                                                     df_calc[coord_col].values,
                                                     w=weights,
                                                     degree=spline_parameters['degree'],
                                                     s=spline_parameters['smoothing_factor'] * N,
                                                     weight_normalization=spline_parameters['weight_normalization']
                                                     )

            df_calc[['dXdZ', 'dYdZ'][i_col]] = merging_spline(df_calc[z_col].values, nu=1,
                                                              extrapolate=spline_parameters['extrapolate'])
        for coord in ['X', 'Y']:
            df_calc[f'wind_{coord}_raw'] = df_calc[f'd{coord}dZ'] * df_calc[vz_air_max_col] #  1.37 #
            df_calc[f'wind_{coord}_new'] = df_calc[f'd{coord}dZ'] * df_calc['dZdT_ground_avg'] - df_calc[f'd{coord}dT_bird_0_avg']- df_calc[f'd{coord}dT_air_0_avg']

    df_calc['wind_X'] = df_calc['wind_X_raw'].rolling(**smoothing_ma_args).mean(**window_args)
    df_calc['wind_Y'] = df_calc['wind_Y_raw'].rolling(**smoothing_ma_args).mean(**window_args)

    if debug:
        from plotting.decomposition.debug import debug_plot_decomposition_wind
        debug_plot_decomposition_wind(df_calc)

    df_calc.dropna(inplace=True)
    return df_calc


def is_in_thermal(df, rho_col, n_modes, bin_size):

    mode, _, _, _ = get_mode_with_KDE(df[rho_col], bin_size)
    df['is_in_thermal'] =  df[rho_col] < n_modes * mode

    return df

def get_thermal_core_by_weighted_average(df, columns_and_weights, max_col=None, sort_by='Z', filter_col=None,
                                         avg_suffix='_avg', std_suffix='_std', sem_suffix='_sem', count_suffix='_N',
                                         ma_args=None):
    df_calc = df.copy()
    if filter_col is not None:
        df_calc = df_calc[df_calc[filter_col]]
    df_return = pd.DataFrame()

    if ma_args is None:
        ma_args = {'window': 36,
                   'min_periods': 1,
                   'center': True}

    df_calc.sort_values(by=[sort_by], inplace=True)

    # Moving Weighted Average
    for value_col, weights in columns_and_weights.items():
        current_df = df_calc.copy()

        if isinstance(weights, LambdaType):
            current_df['weights'] = current_df.apply(weights, axis=1)
        elif isinstance(weights, str):
            current_df['weights'] = current_df[weights]
        else:
            current_df['weights'] = weights

        # Deal with NA and negative values on the weight values
        current_df['weights'] = weights_preprocessing(current_df['weights'].values)

        array_vz_max = rolling_apply_multiple_columns(get_weighted_average,
                                                      ma_args['window'],
                                                      ma_args['min_periods'],
                                                      current_df[value_col].values,
                                                      current_df['weights'].values,
                                                      drop_na=True)

        df_return[value_col + avg_suffix] = array_vz_max[:, 0]
        df_return[value_col + std_suffix] = array_vz_max[:, 1]
        df_return[value_col + sem_suffix] = array_vz_max[:, 2]
        df_return[value_col + count_suffix] = array_vz_max[:, 3]

    if max_col is not None:
        if 'win_type' in ma_args.keys():
            ma_args.pop('win_type')
        if 'window_args' in ma_args.keys():
            ma_args.pop('window_args')
        df_return[max_col + '_max'] = df_calc[max_col].rolling(**ma_args).max().values

    df_return.dropna(inplace=True)
    df_return.drop_duplicates(inplace=True)
    return df_return


def merge_dfs_with_spline(df, df_other, other_cols_to_merge, other_merge_on, merge_on,
                          spline_degree=1, smoothing_factor=0, extrapolate=0, weight_normalization=False, debug=False):
    if other_cols_to_merge is None:
        other_cols_to_merge = df_other.columns

    df_calc = df.copy()
    df_other_calc = df_other.copy()

    df_other_calc.drop_duplicates(subset=other_merge_on, inplace=True, ignore_index=True)
    df_other_calc.sort_values([other_merge_on], inplace=True)
    # The amount of smoothness is determined by satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s,
    # So if the weights w are normalized, that sum will be of the order of N, so the smoothing_factor should be of the
    # order of 1
    if smoothing_factor is None:
        smoothing = df_other_calc[other_merge_on].size
    else:
        smoothing = smoothing_factor * df_other_calc[other_merge_on].size

    # Weighted spline
    merging_spline = {}
    if isinstance(other_cols_to_merge, dict):
        for col, w in other_cols_to_merge.items():

            if isinstance(w, LambdaType):
                weights = df_other_calc.apply(w, axis=1).values
            elif isinstance(w, str):
                if np.all(df_other_calc[w].values == 0):
                    weights = np.ones(shape=df_other_calc[w].values.shape)
                else:
                    weights = 1 / df_other_calc[w].values
            else:
                raise TypeError
            weights = np.array(weights)

            merging_spline[col] = UnivariateSplineWrapper(df_other_calc[other_merge_on].values.T,
                                                          df_other_calc[col].values,
                                                          degree=spline_degree, s=smoothing, w=weights,
                                                          weight_normalization=weight_normalization
                                                          )
            df_calc[col] = merging_spline[col](df_calc[merge_on].values, extrapolate=extrapolate)

    else:  # Not Weighted Spline
        for col in other_cols_to_merge:
            merging_spline[col] = UnivariateSplineWrapper(df_other_calc[other_merge_on].values.T,
                                                          df_other_calc[col].values,
                                                          degree=spline_degree, s=smoothing,
                                                          )
            df_calc[col] = merging_spline[col](df_calc[merge_on].values,  extrapolate=extrapolate)
    if debug:
        for i_col, col in enumerate(other_cols_to_merge):
            import matplotlib
            matplotlib.use('QtAgg')
            import matplotlib.pyplot as plt
            x_values = df_calc.sort_values(merge_on)[merge_on]
            plt.plot(x_values,
                     merging_spline[col](x_values)
                     )
            plt.plot(df_other_calc.sort_values(other_merge_on)[other_merge_on].values,
                     df_other_calc.sort_values(other_merge_on)[col].values
                     )
            plt.xlabel(merge_on)
            plt.ylabel(col)
            plt.show(block=True)

    spline_stats = {col: {'tck': [spl.get_tck()[0].tolist(), spl.get_tck()[1].tolist(), spl.get_tck()[2] ],
                          'residuals': spl.get_residual()}
                    for col, spl in merging_spline.items()}
    return df_calc, spline_stats

def get_horizontal_thermal_velocity_by_moving_average(df, window, min_periods, centered=True,
                                                      V_H_cols=None, Z_col='Z_thermal_1'):
    if V_H_cols is None:
        V_H_cols = ['dXdT_air_0', 'dYdT_air_0']

    list_of_metrics = ['median', 'std', 'count']
    df_calc = df[ [Z_col] + V_H_cols]
    df_roll = get_rolling_basic_statistics(df_calc, window, min_periods, V_H_cols, Z_col,
                                           list_of_metrics=list_of_metrics, center=centered)

    df_roll.rename(columns={col: col.replace(V_H_cols[0], 'wind_X') for col in df_roll.columns}, inplace=True)
    df_roll.rename(columns={col: col.replace(V_H_cols[1], 'wind_Y') for col in df_roll.columns}, inplace=True)
    df_roll.rename(columns={f'{Z_col}_{list_of_metrics[0]}': f'{Z_col}_avg'}, inplace=True)
    df_roll.rename(columns={f'wind_X_{list_of_metrics[0]}':'wind_X'}, inplace=True)
    df_roll.rename(columns={f'wind_Y_{list_of_metrics[0]}':'wind_Y'}, inplace=True)
    #ma_args = thermal_core_ma_args,
    return df_roll


def get_flock_compensation(df, ma_args):
    df_calc = df.copy()
    df_calc = df_calc[['Z', 'dXdT_bird_air', 'dYdT_bird_air']]
    df_calc = df_calc.sort_values('Z')
    ma_args = deepcopy(ma_args)
    _ = ma_args.pop('win_type')
    _ = ma_args.pop('window_args')
    df_return = get_rolling_basic_statistics(df_calc.loc[(~df_calc['dXdT_bird_air'].isna())
                                                         & (~df_calc['dYdT_bird_air'].isna()), ['Z', 'dXdT_bird_air', 'dYdT_bird_air']],
                                             rolling_col='Z',
                                             stats_cols=['dXdT_bird_air', 'dYdT_bird_air'],
                                             **ma_args)
    # df_return = (df_calc.loc[(~df_calc['dXdT_bird_air'].isna())
    #             & (~df_calc['dYdT_bird_air'].isna()), ['Z', 'dXdT_bird_air', 'dYdT_bird_air']].rolling(**ma_args)
    #              .agg(Z_avg=('Z', np.nanmean ),
    #                   dXdT_bird_air_avg=('dXdT_bird_air', np.nanmean ),
    #                   dYdT_bird_air_avg=('dYdT_bird_air', np.nanmean ),
    #                   Z_std= ('Z', np.nanstd ),
    #                   dXdT_bird_air_std= ('dXdT_bird_air', np.nanstd ),
    #                   dYdT_bird_air_std= ('dYdT_bird_air', np.nanstd )
    #                   ))

    df_return.rename(columns={'Z_mean': 'Z_avg',
                              'dXdT_bird_air_mean': 'dXdT_bird_compensation_avg',
                              'dXdT_bird_air_std': 'dXdT_bird_compensation_std',
                              'dYdT_bird_air_mean': 'dYdT_bird_compensation_avg',
                              'dYdT_bird_air_std': 'dYdT_bird_compensation_std',
                              }, inplace=True)
    return df_return

def get_bird_compensation(df, thermal_core_spline, ma_args):
    df_calc = df.copy()
    df_calc = df_calc[['time', 'bird_name', 'dZdT_bird_0', 'Z']]
    ma_args = deepcopy(ma_args)
    list_of_birds = df_calc['bird_name'].unique()
    n_birds = len(list_of_birds)
    ma_args['window'] = round(ma_args['window'] / n_birds)
    ma_args['min_periods'] = round(ma_args['min_periods'] / n_birds)
    win_type = ma_args.pop('win_type', {})
    window_args = ma_args.pop('window_args', {})
    thermal_core_spline = {'X': UnivariateSplineWrapper.from_tck(thermal_core_spline['thermal_core_X']['tck']),
                           'Y': UnivariateSplineWrapper.from_tck(thermal_core_spline['thermal_core_Y']['tck'])}

    df_bird_compensations = pd.DataFrame()
    for bird in list_of_birds:
        df_bird = df_calc[df_calc['bird_name'] == bird]

        df_bird['dZdT_bird_0_avg'] = df_bird['dZdT_bird_0'].rolling(**ma_args).apply(lambda x: np.nanmean(x))
        df_bird['dZdT_bird_0_avg'] = df_bird['dZdT_bird_0_avg'].ffill()
        df_bird['dZdT_bird_0_avg'] = df_bird['dZdT_bird_0_avg'].bfill()
        for coord in ['X', 'Y']:
            df_bird[f'd{coord}dT'] = thermal_core_spline[coord](df_bird['Z'].values, nu=1)
            df_bird[f'd{coord}dT_bird_0_avg'] = df_bird[f'd{coord}dT'] * df_bird['dZdT_bird_0_avg']
        df_bird_compensations = pd.concat([df_bird_compensations, df_bird])

    return df_bird_compensations

def get_random_walk_wind_space_time(wind_avg, wind_std, z_max, x_max, time_max, n_steps=200):

    x_values = np.linspace(-x_max, x_max, n_steps)
    z_values = np.linspace(0, z_max, n_steps)
    t_values = np.linspace(0, time_max, n_steps)
    mgs = np.meshgrid(x_values, z_values,t_values)

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)

    for i in range(n_steps * n_steps):
        wind_x_values_avg = wind_avg[0]
        wind_x_values = wind_std[0] * np.random.standard_normal(n_steps).cumsum()
        wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

        wind_y_values_avg = wind_avg[1]

        wind_y_values = wind_std[1] * np.random.standard_normal(n_steps).cumsum()
        wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

        current_wind_values = np.vstack([wind_x_values, wind_y_values]).T
        if i == 0:
            wind_values = current_wind_values
        else:
            wind_values = np.vstack([wind_values, current_wind_values])


    return xyzt_array, wind_values


def get_ND_random_walk(mean, std, n_vars, n_steps=200, shape=2):

    if np.isscalar(mean):
        mean = np.full(shape=shape, fill_value=mean)
    if np.isscalar(std):
        std = np.full(shape=shape, fill_value=std)

    RW_values = mean + np.multiply(np.random.standard_normal((n_steps ** n_vars, shape)), std).cumsum(axis=0)
    RW_values = mean + (RW_values - np.mean(RW_values, axis=0))

    return RW_values


def get_4D_random_walk_wind(mean, std, limits, n_steps=200):

    mgs = np.meshgrid(*[np.linspace(*limits, n_steps) for limits in limits.values()])
    used_vars = list(limits.keys())
    used_vars = ''.join(used_vars)

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)

    for i in range(len(used_vars)):
        wind_x_values_avg = mean[0]
        wind_x_values = std[0] * np.random.standard_normal(n_steps).cumsum()
        wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

        wind_y_values_avg = mean[1]

        wind_y_values = std[1] * np.random.standard_normal(n_steps).cumsum()
        wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

        current_wind_values = np.vstack([wind_x_values, wind_y_values]).T
        if i == 0:
            wind_values = current_wind_values
        else:
            wind_values = np.vstack([wind_values, current_wind_values])

    if len(used_vars) == 1:
        xyzt_array = xyzt_array.flatten()
    return xyzt_array, wind_values, used_vars


def get_random_walk_wind_with_time(wind_avg, wind_std, z_max, x_max, time_max, n_steps=200):

    x_values = np.linspace(-x_max, x_max, n_steps)
    z_values = np.linspace(0, z_max, n_steps)
    t_values = np.linspace(0, time_max, n_steps)
    mgs = np.meshgrid(x_values, z_values,t_values)

    xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)

    for i in range(n_steps * n_steps):
        wind_x_values_avg = wind_avg[0]
        wind_x_values = wind_std[0] * np.random.standard_normal(n_steps).cumsum()
        wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

        wind_y_values_avg = wind_avg[1]

        wind_y_values = wind_std[1] * np.random.standard_normal(n_steps).cumsum()
        wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

        current_wind_values = np.vstack([wind_x_values, wind_y_values]).T
        if i == 0:
            wind_values = current_wind_values
        else:
            wind_values = np.vstack([wind_values, current_wind_values])


    return xyzt_array, wind_values


def get_random_walk_wind(wind_avg, wind_std, z_max, n_steps=200):

    wind_x_values_avg = wind_avg[0]
    wind_x_values = wind_std[0] * np.random.standard_normal(n_steps).cumsum()
    wind_x_values = wind_x_values_avg + (wind_x_values - wind_x_values.mean())

    wind_y_values_avg = wind_avg[1]

    wind_y_values = wind_std[1] * np.random.standard_normal(n_steps).cumsum()
    wind_y_values = wind_y_values_avg + (wind_y_values - wind_y_values.mean())

    z_values = np.linspace(0, z_max, n_steps)

    return np.vstack([z_values, wind_x_values, wind_y_values])


def get_hierarchical_turbulence(X, t, scales_dict, list_of_interpolator, interpolator_parameters):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n_dim = X.ndim
    if X.ndim == 1:
        X = X.reshape(1, X.shape[0])
    output_shape = np.hstack([X.shape[:-1], [len(list_of_interpolator), len(scales_dict)]])
    results = np.empty(shape=output_shape, dtype=float)

    for i_dimension in range(len(list_of_interpolator)):  # loop on the three components of velocity
        for i_scale, (spatial_scale, velocity_scale) in enumerate(scales_dict.items()):

            X_alt = np.mod(X, spatial_scale) * interpolator_parameters['spatial_scale'] / spatial_scale
            Xt = np.empty(shape=X.shape[:-1] + (4,))
            Xt[..., 1:] = X_alt
            Xt[..., 0] = t
            results[..., i_dimension, i_scale] = (list_of_interpolator[i_dimension](Xt) * velocity_scale
                                                  / interpolator_parameters['velocity_scale'])
        #print(3)
    if len(list_of_interpolator) == 1:
        results = results[..., 0, :]
    if n_dim == 1:
        results = results[0]
    return results


def get_maximum_vertical_velocity(rho_array, Vz_array, Z_array, n_sigma=None, b_epsilon=None, debug=False):
    rho_array_mask = np.logical_not(np.isnan(rho_array))
    Vz_array_mask = np.logical_not(np.isnan(Vz_array))
    Z_array_mask = np.logical_not(np.isnan(Z_array))

    na_mask = np.logical_and(rho_array_mask, Vz_array_mask, Z_array_mask)
    if b_epsilon is None:
        b_epsilon = np.inf
    if n_sigma is not None:
        rho_median = np.median(rho_array)
        rho_std = np.std(rho_array)
        mask = rho_array < rho_median + n_sigma * rho_std
        mask = np.logical_and(mask, na_mask)
    else:
        mask = na_mask
    Z_avg = np.mean(Z_array)

    N = mask.sum()

    rho_max, dZdT_air_max, popt, pcov, is_maximum = get_maximum_by_quadratic_fit(rho_array[mask],
                                                                     Vz_array[mask],
                                                                     bounds=([-np.inf, -b_epsilon, -np.inf],
                                                                             [np.inf, b_epsilon, np.inf]))

    if debug and Z_avg > 300:
        from plotting.decomposition.debug import debug_plot_decomposition_get_maximum_vertical_velocity
        debug_plot_decomposition_get_maximum_vertical_velocity(rho_array, Vz_array, mask, rho_max, dZdT_air_max, popt)
    if not is_maximum:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return rho_max, dZdT_air_max, Z_avg, popt, pcov, N


def get_rolling_maximum_vertical_velocity(df_iteration, radius_col='rho_thermal', Vz_col='dZdT_air_0',
                                          sort_by_col='Z_thermal',
                                          ma_args=None, debug=False):
    ma_args = ma_args.copy()

    if ma_args is None:
        ma_args = {'window': 12, 'min_periods': 1, 'center': True}

    if 'window_args' in ma_args.keys():
        window_args = ma_args.pop('window_args')
    else:
        window_args = {}

    df_calc = df_iteration.copy()
    df_calc = df_calc.sort_values(sort_by_col)

    # Remove outliers that strongly bias the fit
    df_calc = df_calc.dropna()

    df_return = pd.DataFrame()

    # min_periods to window_size
    for i in range(ma_args['min_periods'], ma_args['window']):
        current_row = {}
        current_df = df_calc.copy().iloc[0: i]

        rho_max_return, dZdT_air_max_return, rho_max, dZdT_air_max, (a, b, c), status = get_maximum_vertical_velocity(
            current_df[radius_col], current_df[Vz_col])
        current_row['Z_avg'] = current_df['Z_thermal'].mean()
        current_row['rho_max'] = rho_max_return
        current_row['dZdT_air_max'] = dZdT_air_max_return

        if debug:
            current_row['real_rho_max'] = rho_max
            current_row['real_dZdT_air_max'] = dZdT_air_max
            current_row['quadratic_fit'] = (a, b, c)
            current_row['status'] = status

        df_return = df_return.append(current_row, ignore_index=True)

    for i in range(len(df_calc.index) - ma_args['window']):
        # print(i)
        current_row = {}
        current_df = df_calc.copy().iloc[i: i + ma_args['window']]

        rho_max_return, dZdT_air_max_return, rho_max, dZdT_air_max, (a, b, c), status = get_maximum_vertical_velocity(
            current_df[radius_col], current_df[Vz_col])
        current_row['Z_avg'] = current_df['Z_thermal'].mean()
        current_row['rho_max'] = rho_max_return
        current_row['dZdT_air_max'] = dZdT_air_max_return

        if debug:
            current_row['real_rho_max'] = rho_max
            current_row['real_dZdT_air_max'] = dZdT_air_max
            current_row['quadratic_fit'] = (a, b, c)
            current_row['status'] = status

        df_return = df_return.append(current_row, ignore_index=True)

    # From window_size to min_periods
    for i in range(ma_args['window']):
        current_row = {}
        current_df = df_calc.copy().iloc[-ma_args['window'] + i:]
        if len(current_df.index) < ma_args['min_periods']:
            break

        rho_max_return, dZdT_air_max_return, rho_max, dZdT_air_max, (a, b, c), status = get_maximum_vertical_velocity(
            current_df[radius_col], current_df[Vz_col])
        current_row['Z_avg'] = current_df['Z_thermal'].mean()
        current_row['rho_max'] = rho_max_return
        current_row['dZdT_air_max'] = dZdT_air_max_return

        if debug:
            current_row['real_rho_max'] = rho_max
            current_row['real_dZdT_air_max'] = dZdT_air_max
            current_row['quadratic_fit'] = (a, b, c)
            current_row['status'] = status

        df_return = df_return.append(current_row, ignore_index=True)

    return df_return
