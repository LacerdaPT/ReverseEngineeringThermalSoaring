import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev, splrep, UnivariateSpline, RegularGridInterpolator

from calc.flight import get_bank_angle_from_radius, \
    get_min_sink_rate_from_bank_angle, get_horizontal_velocity_from_bird_parameters
from calc.geometry import get_curvature_from_trajectory
from numba import njit, float64


def cart_to_cyl_point(X: np.ndarray):
    return np.array([np.linalg.norm(X[...,:2]),
                     np.arctan2(X[...,1], X[...,0]),
                     X[...,2]
                     ])

def cyl_to_cart_point(X_cyl):
    return np.array([X_cyl[0] * np.cos(X_cyl[1]),
                     X_cyl[0] * np.sin(X_cyl[1]),
                     X_cyl[2]])

def cart_to_cyl_matrix(t):
    return np.array([[np.cos(t), np.sin(t), 0],
                     [-np.sin(t), np.cos(t), 0],
                     [0, 0, 1]
                     ])



def get_regular_grid_from_irregular_data(x_array, y_array, *dependent_arrays, resolution):
    epsilon = 0
    from matplotlib import tri
    if isinstance(resolution, int):
        resolution = [resolution] * 2
    xi = np.linspace(np.min(x_array) - epsilon,
                     np.max(x_array) + epsilon, resolution[0], endpoint=True)
    yi = np.linspace(np.min(y_array) - epsilon,
                     np.max(y_array) + epsilon, resolution[1], endpoint=True)
    Xi, Yi = np.meshgrid(xi, yi)

    triang = tri.Triangulation(x_array.flatten(),
                               y_array.flatten())
    interpolators = [tri.LinearTriInterpolator(triang, current_array.flatten())
                     for current_array in dependent_arrays if current_array is not None]

    v_i = [interpolator(Xi, Yi) for interpolator in interpolators]

    return Xi, Yi, v_i


@njit
def get_numba_meshgrid_3D(x, y, z):
    xx = np.empty(shape=(x.size, y.size, z.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size, z.size), dtype=y.dtype)
    zz = np.empty(shape=(x.size, y.size, z.size), dtype=z.dtype)
    for i in range(z.size):
        for j in range(y.size):
            for k in range(x.size):
                xx[i,j,k] = k  # change to x[k] if indexing xy
                yy[i,j,k] = j  # change to y[j] if indexing xy
                zz[i,j,k] = i  # change to z[i] if indexing xy
    return xx, yy, zz


def parse_projection_string(projection_str):
    # Parse cross-section type
    projection_str = projection_str.lower()
    first_var, second_var = projection_str

    first_index = 'xyz'.index(first_var)
    second_index = 'xyz'.index(second_var)

    section_index = int('012'.replace(str(first_index), '').replace(str(second_index), ''))
    section_var = 'xyz'[section_index]

    return first_var, second_var, first_index, second_index, section_index, section_var


class SplineWrapper(object):
    def __init__(self, X, y, degree=3, s=0, w=None):
        self.X = X
        self.y = y
        self.s = s
        self.w = w
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        self.degree = degree
        self.tck = None

        if isinstance(self.X, (pd.Series, pd.DataFrame)):
            self.X = self.X.values

        if isinstance(self.y, (pd.Series, pd.DataFrame)):
            self.y = self.y.values

    def fit(self):
        if self.X.ndim == 1:
            self.tck = splrep(y=self.X, x=self.y, w=self.w, s=self.s, k=self.degree)
        else:
            self.tck, _ = splprep(x=self.X, u=self.y, w=self.w, s=self.s, k=self.degree)

    def __call__(self, x, der=0, extrapolate=0):
        if self.tck is None:
            raise RuntimeError('must run fit method first')
        result = splev(x=x, tck=self.tck, der=der, ext=extrapolate)
        return np.array(result)


class UnivariateSplineWrapper(UnivariateSpline):
    def __init__(self, x, y, degree=3, s=0, w=None, weight_normalization=False):
        self.x = x
        self.degree = degree
        self.y = y
        self.s = s
        self.w = w
        self.weight_normalization = weight_normalization
        if self.weight_normalization and (self.w is not None):
            self.w = self.w / np.sum(self.w)
        self.x_min = np.min(x)
        self.x_max = np.max(x)
        self.degree = degree
        self.tck = None
        try:
            self.ret = super().__init__(self.x, self.y, k=self.degree, w=self.w, s=self.s)
        except UserWarning as e:
            raise

    @classmethod
    def from_tck(cls, tck, ext=0):
        self = super()._from_tck(tck, ext=ext)
        self.x = None
        self.degree = tck[-1]
        self.y = None
        self.s = None
        self.w = None
        self.weight_normalization = None
        if self.weight_normalization and (self.w is not None):
            self.w = self.w / np.sum(self.w)
        self.x_min = np.min(self.get_knots())
        self.x_max = np.max(self.get_knots())

        return self

    def _linear_interpolation(self, x, x0=None, nu=0):
        if x0 is None:
            #find the closest boundary
            x0 = np.where(np.argmin(np.stack([np.abs(x - self.x_min),
                                              np.abs(x - self.x_max)],
                                             axis=-1),
                                    axis=-1) == 0,
                          self.x_min,
                          self.x_max)
        f0 = super().__call__(x0, nu=nu, ext=0)
        slope = super().__call__(x0, nu=nu + 1, ext=0)
        result = f0 + slope * (x - x0)
        return result

    def get_tck(self):
        return self._eval_args


    def __call__(self, x, nu=0, extrapolate=0):
        if extrapolate == 'linear':
            result = np.where((x <= self.x_max) & (x >= self.x_min), super().__call__(x, nu=nu, ext=0),
                            self._linear_interpolation(x, nu=nu)
                         )
        elif isinstance(x, (np.ndarray, list)):
            if extrapolate == 'linear2':
                result = list(map(lambda elem: self(elem, nu=nu, extrapolate=extrapolate), x))
            else:
                result = super().__call__(x, nu=nu, ext=extrapolate)
        elif extrapolate == 'linear2':
            if x < self.x_min:
                result = self._linear_interpolation(x, self.x_min, nu)
            elif x > self.x_max:
                result = self._linear_interpolation(x, self.x_max, nu)
            else:
                result = super().__call__(x, nu=nu, ext=0)
        else:
            result = super().__call__(x, nu=nu, ext=extrapolate)

        return np.array(result)


def get_geometric_characteristics(df, state_vector_columns=None, window_size=3, smooth_radius=False):
    #df_calc = df_track.copy()

    if state_vector_columns is None:
        state_vector_columns = {'X': 'X_bird',
                                'Y': 'Y_bird',
                                'Z': 'Z_bird',
                                'Vx': 'dXdT_bird',
                                'Vy': 'dYdT_bird',
                                'Ax': 'd2XdT2_bird',
                                'Ay': 'd2YdT2_bird'}
    df_calc = df.copy()
    df_calc = df_calc[['time', 'bird_name'] + list(state_vector_columns.values())]
    df_calc.rename(columns={v: k for k, v in state_vector_columns.items()}, inplace=True)
    # for k, v in state_vector_columns.items():
    #     df_calc.loc[:, k] = df.loc[:, v]

    df_radius = pd.DataFrame(columns=['time', 'bird_name', #'curvature_center_X', 'curvature_center_Y',
                                      'radius_raw', 'radius'])
    list_of_birds = df_calc['bird_name'].unique()
    for bird in list_of_birds:
        current_bird = df_calc[df_calc['bird_name'] == bird]
        window_size = window_size  #* 10 + 1
        #
        # ls_result = rolling_apply_multiple_columns(calculate_circle_lulu, window_size, None,
        #                                            current_bird['X'].values,
        #                                            current_bird['Y'].values,
        #                                            drop_na=False, center=True)

        current_radius = current_bird[['time', 'bird_name']].copy()
        current_radius['curvature'] = np.vectorize(get_curvature_from_trajectory)(current_bird['Vx'].values,
                                                                                  current_bird['Vy'].values,
                                                                                  current_bird['Ax'].values,
                                                                                  current_bird['Ay'].values,)
        current_radius['radius_raw'] = 1 / current_radius['curvature']
        #current_radius['curvature_center_X'] = current_bird['Ax'].values / np.linalg.norm(current_bird[['Ax', 'Ay']].values, axis=1) *  current_radius['radius_raw']
        #current_radius['curvature_center_Y'] = current_bird['Ay'].values / np.linalg.norm(current_bird[['Ax', 'Ay']].values, axis=1) *  current_radius['radius_raw']
        #current_radius['curvature_center_X'] = ls_result[:, 0]
        #current_radius['curvature_center_Y'] = ls_result[:, 1]
        #current_radius['radius_raw'] = ls_result[:, 2]


        if smooth_radius:
            current_radius['radius'] = current_radius['radius_raw'].rolling(3, center=True, min_periods=1,
                                                                      win_type='gaussian').mean(std=1)
        else:
            current_radius['radius'] = current_radius['radius_raw']
        if df_radius.empty:
            df_radius = current_radius.copy()
        else:
            df_radius = pd.concat([df_radius, current_radius], ignore_index=True)


    #df_radius['curvature'] = 1 / df_radius['radius']
    df_calc = pd.merge(df_calc, df_radius, on=['bird_name', 'time'])
    # df_calc['radius_vector_x'] = np.abs(df_calc['radius']) * (df_calc['X'] - df_calc['curvature_center_X'])
    # df_calc['radius_vector_y'] = np.abs(df_calc['radius']) * (df_calc['Y'] - df_calc['curvature_center_Y'])
    #
    # df_calc['radius_unit_vector_x'] = df_calc['radius_vector_x'] / np.linalg.norm(df_calc[['radius_vector_x',
    #                                                                                        'radius_vector_y']],
    #                                                                               axis=1)
    # df_calc['radius_unit_vector_y'] = df_calc['radius_vector_y'] / np.linalg.norm(df_calc[['radius_vector_x',
    #                                                                                        'radius_vector_y']],
    #                                                                               axis=1)
    #
    # df_calc['radius_vector_x'] = np.abs(df_calc['radius']) * df_calc['radius_unit_vector_x']
    # df_calc['radius_vector_y'] = np.abs(df_calc['radius']) * df_calc['radius_unit_vector_y']
    # df_calc['velocity_unit_vector_x'] = - df_calc['radius_unit_vector_y']
    # df_calc['velocity_unit_vector_y'] = df_calc['radius_unit_vector_x']
    df_calc['velocity_unit_vector_x'] = df_calc['Vx'].values / np.linalg.norm(df_calc[['Vx', 'Vy']].values, axis=1) #np.sign(df_calc['radius']) * df_calc['velocity_unit_vector_x']
    df_calc['velocity_unit_vector_y'] = df_calc['Vy'].values / np.linalg.norm(df_calc[['Vx', 'Vy']].values, axis=1) #np.sign(df_calc['radius']) * df_calc['velocity_unit_vector_y']
    df_calc.drop(columns=list(state_vector_columns.keys()), inplace=True)
    return df_calc


def get_flight_characteristics(df_track, df_bird_parameters, state_vector_columns=None, radius_col='radius'):
    if state_vector_columns is None:
        state_vector_columns = {'X': 'X_bird',
                                'Y': 'Y_bird',
                                'Z': 'Z_bird',
                                'Vx': 'dXdT_bird',
                                'Vy': 'dYdT_bird',
                                'Ax': 'd2XdT2_bird',
                                'Ay': 'd2YdT2_bird'}
    df_calc = df_track.copy()
    df_calc = df_calc[['time', 'bird_name'] 
                      + list(state_vector_columns.values()) 
                      + ['velocity_unit_vector_x', 'velocity_unit_vector_y', 'radius']]
    df_calc.rename(columns={v: k for k, v in state_vector_columns.items()}, inplace=True)
    # for k, v in state_vector_columns.items():
    #     df_calc.loc[:, k] = df.loc[:, v]

    df_return = pd.DataFrame()

    for _, current_bird_parameters in df_bird_parameters.iterrows():
        current_bird_parameters = current_bird_parameters.to_dict()
        bird = current_bird_parameters['bird_name']
        current_bird = df_calc[df_calc['bird_name'] == bird].copy()

        current_bird_parameters.pop('bird_name')
        current_bird['bank_angle'] = current_bird['radius'].apply(get_bank_angle_from_radius, **current_bird_parameters)
    
        current_bird['Vh'] = current_bird['bank_angle'].apply(get_horizontal_velocity_from_bird_parameters,
                                                    **{'mass': current_bird_parameters['mass'],
                                                       'wing_area': current_bird_parameters['wing_area'],
                                                       'wing_loading': current_bird_parameters['wing_loading'],
                                                       'CL': current_bird_parameters['CL']})
    
        current_bird['min_sink_rate'] = current_bird['bank_angle'].apply(get_min_sink_rate_from_bank_angle,
                                                               **current_bird_parameters)

        current_bird['Vx'] = current_bird['velocity_unit_vector_x'] * current_bird['Vh']
        current_bird['Vy'] = current_bird['velocity_unit_vector_y'] * current_bird['Vh']
        current_bird['Vz'] = -current_bird['min_sink_rate']
        if df_return.empty:
            df_return = current_bird.copy()
        else:
            df_return = pd.concat([df_return, current_bird])

    df_return = df_return[['time', 'bird_name', 'bank_angle', 'Vh', 'min_sink_rate', 'Vx', 'Vy', 'Vz']]
    return df_return


@njit('float64[::1](float64[::1], float64, int32)')
def get_3_point_stencil_differentiation(data_array, dt, n=1):
    assert n in [1, 2], 'the order of differentiation n must be 1, 2, 3 or 4'
    if n == 1:
        weights = np.array([1, 0, -1])
        weights = weights / (2 * dt)
    elif n == 2:
        weights = np.array([1, -2, 1])
        weights = weights / (dt ** 2)
    N = data_array.size
    M = weights.size
    result_size = max(M, N) - min(M, N) + 1
    # numba does not support numpy.convolve with the 'mode' argument.
    # Therefore we need to cut the unwanted data manually
    result = np.empty(shape=result_size, dtype=float64)
    result = np.convolve(data_array, weights,
                         #mode='valid'
                         )[2:-2]
    return result
@njit('float64[::1](float64[::1], float64, int32)')
def get_5_point_stencil_differentiation(data_array, dt, n=1):
    assert n in [1, 2, 3, 4], 'the order of differentiation n must be 1, 2, 3 or 4'
    if n == 1:
        weights = np.array([-1, 8, 0, -8, 1])
        weights = weights / (12 * dt)
    elif n == 2:
        weights = np.array([-1, 16, -30, 16, -1])
        weights = weights / (12 * dt ** 2)
    elif n == 3:
        weights = np.array([1, -2, 0, 2, -1])
        weights = weights / (2 * dt ** 3)
    elif n == 4:
        weights = np.array([1, -4, 6, -4, 1])
        weights = weights / (1 * dt ** 4)
    N = data_array.size
    M = weights.size
    result_size = max(M, N) - min(M, N) + 1
    # numba does not support numpy.convolve with the 'mode' argument.
    # Therefore we need to cut the unwanted data manually
    result = np.empty(shape=result_size, dtype=float64)
    result = np.convolve(data_array, weights,
                         #mode='valid'
                         )[4:-4]
    return result


def get_smoothed_diff_per_partition(df, moving_average_params, dt, n=1, method='5point', partition_key='bird_name'):
    column_names_to_eval = list(moving_average_params.keys())
    unique_partitions = df[partition_key].unique()
    results_dict = {col: np.array([]) for col in column_names_to_eval}
    df_calc = df[column_names_to_eval + ['time', partition_key]]
    for idx, partition_value in enumerate(unique_partitions):
        df_filter = df_calc.loc[df_calc[partition_key] == partition_value]

        for col, ma_args in moving_average_params.items():
            # diff
            if method == '5point':
                current_diff = get_5_point_stencil_differentiation(df_filter[col].values, dt=dt, n=n)

                diff = df_filter[col].diff().values/dt
                current_diff = np.concatenate((diff[:2], current_diff, diff[-2:]))  # diff[0] = nan
            elif method == '3point':
                current_diff = get_3_point_stencil_differentiation(df_filter[col].values, dt=dt, n=n)

                diff = df_filter[col].diff().values/dt
                current_diff = np.concatenate((diff[:1], current_diff, diff[-1:]))  # diff[0] = nan
            elif method == 'spline':

                sp = UnivariateSpline(df_filter['time'].values, df_filter[col].values)
                current_diff = sp(df_filter['time'].values, nu=n)
            else:
                current_diff = df_filter[col].diff().values / dt

            # Linear Interpolation
            # (f2 - f1) / (t2-t1)
            # t2 - t1 = (index_2 - index_1) * dt = 1 * dt, but when t0 and t1 are calculated the dt would cancel out
            m = (current_diff[2] - current_diff[1])

            b = current_diff[1]
            d0 = b + m * (-1)
            current_diff[0] = d0

            # Smoothing
            if ma_args is not None:
                ma_args = ma_args.copy()
                if (ma_args is not None) and ('window_args' in ma_args.keys()):
                    window_args = ma_args.pop('window_args')
                else:
                    window_args = {}

                current_diff = pd.Series(current_diff).rolling(**ma_args).mean(**window_args).values

            results_dict[col] = np.concatenate([results_dict[col],  # Concatenate to add each bird to the right column
                                                current_diff])

    return results_dict




def integrate_bird_velocities(df):
    all_bird_positions = pd.DataFrame()
    for i_bird, bird in enumerate(df['bird_name'].unique()):

        current_bird = df[df['bird_name'] == bird]
        current_bird_trajectory = current_bird[['bird_name', 'time']].copy()


        for i_coord, coord in enumerate( ['X', 'Y', 'Z']):
            current_integrated_bird_trajectory = np.empty((1, 1))
            current_integrated_bird_trajectory[0, 0] =  0
            X_init = current_bird[f'{coord}_bird_TC'].values[0]
            current_bird.loc[:, f'd{coord}dT_bird_air'] = current_bird[f'd{coord}dT_bird_air']
            na_mask = ~current_bird[f'd{coord}dT_bird_air'].isna().values
            current_time = current_bird.loc[na_mask, 'time']
            current_dt = current_time.diff().dropna().values
            current_velocity = current_bird.loc[na_mask, f'd{coord}dT_bird_air'].values
            for i in range(0, len(current_dt)):
                v = current_velocity[i]
                dt = current_dt[i]
                current_integrated_bird_trajectory = np.concatenate([current_integrated_bird_trajectory,
                                                             (current_integrated_bird_trajectory[-1] + v * dt).reshape(-1, 1)])


            current_bird_positions = pd.DataFrame()
            current_bird_positions['time'] = current_time.values
            current_bird_positions['bird_name'] = bird
            current_bird_positions[f'{coord}_bird_air'] = current_integrated_bird_trajectory[:] + X_init

            current_bird_trajectory = pd.merge(current_bird_trajectory, current_bird_positions, how='left', on=['bird_name', 'time'])

        all_bird_positions = pd.concat([all_bird_positions, current_bird_trajectory])

    return all_bird_positions


def get_moving_average_per_bird(df, moving_average_params, partition_key='bird_name'):
    column_names_to_eval = list(moving_average_params.keys())

    unique_partitions = df[partition_key].unique()

    moving_average_results_dict = {col: np.array([]) for col in column_names_to_eval}

    for idx, partition_value in enumerate(unique_partitions):
        df_filter = df.loc[df[partition_key] == partition_value]
        df_filter = df_filter[column_names_to_eval]

        for col, ma_args in moving_average_params.items():

            if ma_args is not None:
                ma_args = ma_args.copy()
                if (ma_args is not None) and ('window_args' in ma_args.keys()):
                    window_args = ma_args.pop('window_args')
                else:
                    window_args = {}
                current_ma = df_filter[col].rolling(**ma_args).mean(**window_args)

                moving_average_results_dict[col] = np.concatenate([moving_average_results_dict[col],
                                                                   current_ma.values])

    return moving_average_results_dict


def get_diff_per_bird(df, column_names_to_eval, partition_key='bird_name'):
    # column_names_to_eval = list(moving_average_params.keys())

    unique_partitions = df[partition_key].unique()

    diff_results_dict = {col: np.array([]) for col in column_names_to_eval}

    for idx, partition_value in enumerate(unique_partitions):
        df_filter = df.loc[df[partition_key] == partition_value]
        df_filter = df_filter[column_names_to_eval]

        for col in column_names_to_eval:
            current_diff = df_filter[col].diff()

            # Linear Interpolation
            current_diff[current_diff.index[0]] = (current_diff[current_diff.index[1]]
                                                   - (current_diff[current_diff.index[2]]
                                                      - current_diff[current_diff.index[1]]
                                                      )
                                                   )

            current_diff = current_diff.fillna(0)
            diff_results_dict[col] = np.concatenate([diff_results_dict[col],
                                                     current_diff.values])

    return diff_results_dict


def calculate_per_partition(df, function, list_of_columns_to_eval, kwargs_per_col=None, partition_key='bird_name',
                            suffix='_new'):
    if kwargs_per_col is None:
        kwargs_per_col = {col: {} for col in list_of_columns_to_eval}

    unique_partitions = df[partition_key].unique()
    df_big = pd.DataFrame()

    for idx, partition_value in enumerate(unique_partitions):
        df_partition = df[df[partition_key] == partition_value]

        for col in list_of_columns_to_eval:
            df_partition[col + suffix] = df_partition.apply(function, **kwargs_per_col[col])
        if df_big.empty:
            df_big = df_partition.copy()
        else:
            df_big = pd.concat([df_big, df_partition])

    return df_big


def get_gradient(f, X, delta=1.0, N=10, ):

    x0, y0, z0 = X
    if N % 2 == 0:
        N = N + 1
    x = np.linspace(x0 - delta, x0 + delta, N, endpoint=True)
    y = np.linspace(y0 - delta, y0 + delta, N, endpoint=True)
    z = np.linspace(z0 - delta, z0 + delta, N, endpoint=True)
    d = 2 * delta / N
    x_mg, y_mg, z_mg = np.meshgrid(x, y, z)

    f_values = f(x_mg, y_mg, z_mg)

    grad = np.gradient(f_values, d)
    N_return = N//2 + 1
    return [grad[i][N_return, N_return, N_return] for i in range(3)]


def fill_nan_with_spline(x_array, f_array, n_max=None, extrapolate='linear'):
    na_mask = np.isnan(f_array)
    spline = UnivariateSplineWrapper(x_array[~na_mask],
                                     f_array[~na_mask],
                                     )

    resampled_f_array = spline(x_array, extrapolate=extrapolate)
    return resampled_f_array, spline


def get_na_mask(*arr):
    arr = np.array(arr)
    na_mask = np.logical_not(np.isnan(arr[0]))
    for a in arr[1:]:
        current_na_mask = np.logical_not(np.isnan(a))
        na_mask = np.logical_and(na_mask, current_na_mask)

    return na_mask


def prepare_tensor_for_periodic_boundary_4D(tensor):
    # Extend turbulence data to include periodic boundary conditions
    # fill 100 x 200 x 200 x 200
    tensor_shape = tensor.shape
    new_dataframe = np.empty(
        shape=(tensor_shape[0] + 1,  # time
               #tensor_shape[1],  # component
               tensor_shape[1] + 1,  # X
               tensor_shape[2] + 1,  # Y
               tensor_shape[3] + 1))  # Z
    new_dataframe[..., :-1, :-1, :-1, :-1] = tensor[..., :, :, :, :]

    # fill extra volumes 200 x 200 x 200, XYZ, TYZ, TXZ, TXY
    new_dataframe[..., -1, :-1, :-1, :-1] = tensor[..., 0, :, :, :]
    new_dataframe[..., :-1, -1, :-1, :-1] = tensor[..., :, 0, :, :]
    new_dataframe[..., :-1, :-1, -1, :-1] = tensor[..., :, :, 0, :]
    new_dataframe[..., :-1, :-1, :-1, -1] = tensor[..., :, :, :, 0]

    # fill extra faces  200 x 200, TX, TY, TZ, XY, YZ, XZ
    new_dataframe[..., -1, -1, :-1, :-1] = tensor[..., 0, 0, :, :]
    new_dataframe[..., -1, :-1, -1, :-1] = tensor[..., 0, :, 0, :]
    new_dataframe[..., -1, :-1, :-1, -1] = tensor[..., 0, :, :, 0]
    new_dataframe[..., :-1, -1, -1, :-1] = tensor[..., :, 0, 0, :]
    new_dataframe[..., :-1, :-1, -1, -1] = tensor[..., :, :, 0, 0]
    new_dataframe[..., :-1, -1, :-1, -1] = tensor[..., :, 0, :, 0]

    #fill extra edges, TX-TY, TX-TZ, TX-XY, *TX-YZ, *TX-XZ
    #                  TY-TZ, TY-XY, TY-YZ, *TY-XZ
    #                  *TZ-XY, TZ-YZ, *TZ-XY
    #                  XY-YZ, XY-XZ
    #                  YZ-XZ

    new_dataframe[..., -1, -1, -1, :-1] = tensor[..., 0, 0, 0, :]  # TX-TY
    new_dataframe[..., -1, -1, :-1, -1] = tensor[..., 0, 0, :, 0]  # TX-TZ
    new_dataframe[..., -1, :-1, -1, -1] = tensor[..., 0, :, 0, 0]  # TY-TZ
    new_dataframe[..., :-1, -1, -1, -1] = tensor[..., :, 0, 0, 0]  # XY-YZ

    #fill extra point
    new_dataframe[..., -1, -1, -1, -1] = tensor[..., 0, 0, 0, 0]
    return new_dataframe

def prepare_tensor_for_periodic_boundary(tensor):
    # Extend turbulence data to include periodic boundary conditions
    tensor_shape = tensor.shape
    if len(tensor_shape) > 3:
        new_dataframe = np.empty(
            shape=(tensor_shape[0],  # time
                   #tensor_shape[1],  # component
                   tensor_shape[1] + 1,  # X
                   tensor_shape[2] + 1,  # Y
                   tensor_shape[3] + 1))  # Z
    else:
        new_dataframe = np.empty(
            shape=(#tensor_shape[0],  # time
                   #tensor_shape[1],  # component
                   tensor_shape[0] + 1,  # X
                   tensor_shape[1] + 1,  # Y
                   tensor_shape[2] + 1))  # Z

    # fill 200x200x200
    new_dataframe[..., :-1, :-1, :-1] = tensor[..., :, :, :]

    # fill extra faces
    new_dataframe[..., -1, :-1, :-1] = tensor[..., 0, :, :]
    new_dataframe[..., :-1, -1, :-1] = tensor[..., :, 0, :]
    new_dataframe[..., :-1, :-1, -1] = tensor[..., :, :, 0]

    #fill extra edges
    new_dataframe[..., -1, -1, :-1] = tensor[..., 0, 0, :]
    new_dataframe[..., :-1, -1, -1] = tensor[..., :, 0, 0]
    new_dataframe[..., -1, :-1, -1] = tensor[..., 0, :, 0]

    #fill extra point
    new_dataframe[..., -1, -1, -1] = tensor[..., 0, 0, 0]

    return new_dataframe


def get_periodic_linear_interpolator(tensor, limits):
    tensor = prepare_tensor_for_periodic_boundary(tensor)
    #del turbulence_dataframe
    tensor = tensor
    tensor_shape = tensor.shape

    points = [np.linspace(l_min, l_max, tensor_shape[i], endpoint=True, dtype=tensor.dtype)
              for i, (l_min, l_max) in enumerate(limits)]

    interpolator = RegularGridInterpolator(points=points, values=tensor)
    return interpolator


def rolling_apply_multiple_columns(func, window_size: int, min_periods, *arrays: np.ndarray, n_jobs: int = 1,
                                   drop_na=True, center=False, **kwargs):
    import numpy_ext as npext
    with warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', category=DeprecationWarning)
        np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

        # From 0 to window_size - 1
        if min_periods is not None:
            return_arr_before = npext.expanding_apply(func,
                                                      min_periods,
                                                      *[arr[:window_size - 1] for arr in arrays],
                                                      n_jobs=n_jobs,
                                                      **kwargs
                                                      )
            # Remove the NaN from the beginning
            if drop_na and (not center):
                return_arr_before = return_arr_before[min_periods-1:]

            # From window_size to N-window_size
            return_arr = npext.rolling_apply(func,
                                             window_size,
                                             *arrays, # window_size - 1 [:-window_size + 1]
                                             n_jobs=n_jobs,
                                             **kwargs)

            # Remove the NaN from the beginning

            if drop_na and (not center):
                return_arr = return_arr[window_size-1:]

            # From N-window_size to N
            return_arr_after = npext.expanding_apply(func,
                                                     min_periods,
                                                     *[arr[-window_size + 1:][::-1] for arr in arrays],
                                                     n_jobs=n_jobs,
                                                     **kwargs
                                                     )
            # Remove the NaN from the beginning
            if drop_na and (not center):
                return_arr_after = return_arr_after[min_periods-1:]
            if center:
                return_arr = np.vstack([return_arr_before[min_periods//2:],
                                        return_arr[window_size:][window_size//2:],
                                        return_arr_after[::-1][:-(min_periods // 2)]
                                        ])
            else:
                return_arr = np.vstack([return_arr_before, return_arr, return_arr_after[::-1]])
        else:
            # From window_size to N-window_size
            return_arr = npext.rolling_apply(func,
                                             window_size,
                                             *arrays,  # window_size - 1 [:-window_size + 1]
                                             n_jobs=n_jobs,
                                             **kwargs)

            # Remove the NaN from the beginning

            if drop_na and (not center):
                return_arr = return_arr[window_size - 1:]
            if center:
                return_arr = np.vstack([return_arr[window_size // 2:],
                                        np.full(shape=(window_size // 2, return_arr.shape[-1]),
                                                fill_value=np.nan)],)

    return return_arr
