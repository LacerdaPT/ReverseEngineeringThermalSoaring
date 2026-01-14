import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

from calc.auxiliar import UnivariateSplineWrapper
from object.air import AirVelocityFieldBase


class ExtendedAirVelocityField():


    def __init__(self, base_avg: AirVelocityFieldBase):
        super().__init__()
        self.wind_function = None
        self.thermal_rotation_function = None
        self.thermal_profile_function = None
        self.velocity_functions_agg = {}
        self.components_agg = set()
        self.components = {'wind', 'thermal', 'rotation'}
        self.df_splines = None

        self.current_thermal_core_spline = {}
        self.thermal_profile_function_agg = None
        self.df_all_iterations = df[df['iteration'] != 0].copy()
        self.df_bins_all_iterations = df_bins[df_bins['iteration'] != 0].copy()

        if not isinstance(df_splines, pd.DataFrame):
            df_splines = pd.DataFrame.from_dict(df_splines)
        self.df_splines_all_iterations = df_splines[df_splines['iteration'] != 0].copy()
        self.list_of_iterations = self.df_all_iterations['iteration'].unique()
        #self.list_of_iterations = list(filter(lambda elem: elem != 0, self.list_of_iterations))
        self.list_of_iterations = sorted(self.list_of_iterations)
        self.current_iteration = self.list_of_iterations[0]
        if self.current_iteration == 0:
            self.current_iteration = 1
        self.df_bins = df_bins
        self.min_occupation_number = min_occupation_number
        self.aggregated = aggregated
        self.preprocessing()
        self.preprocessing_agg()
        # self.get_velocity = np.vectorize(self.get_velocity)

    def set_iteration(self, i):
        self.current_iteration = i
        self.preprocessing()
        self.preprocessing_agg()

    def preprocessing(self):
        self.preprocess_data()
        self.preprocess_wind()
        self.preprocess_thermal_rotation()
        self.preprocess_thermal_profile()
        self.set_thermal_core()
        self.preprocess_turbulence()
        # self.set_thermal_core_advection_time_dependent()

    def preprocess_data(self):
        self.df = self.df_all_iterations[self.df_all_iterations['iteration'] == self.current_iteration].copy()
        self.df_bins = self.df_bins_all_iterations[self.df_bins_all_iterations['iteration'] == self.current_iteration].copy()
        self.df_splines = self.df_splines_all_iterations[self.df_splines_all_iterations['iteration'] == self.current_iteration].copy()

        # READ SPLINES
        current_spline = self.df_splines.to_dict(orient='records')[0]
        self.current_thermal_core_spline = {}
        self.wind_spline = {}
        self.wind_correction_spline = {}
        for col in ['X', 'Y']:
            self.current_thermal_core_spline[col] = UnivariateSplineWrapper.from_tck(
                current_spline['thermal_core_positions'][f'{col}_avg']['tck'])
            #self.wind_correction_spline[col] = UnivariateSplineWrapper.from_tck(current_spline['wind'][f'wind_{col}']['tck'])
            self.wind_spline[col] = UnivariateSplineWrapper.from_tck(
                current_spline['wind'][f'wind_{col}']['tck'])

        self.df = self.df[self.df['Z'].between(self.current_thermal_core_spline['X'].x_min, self.current_thermal_core_spline['X'].x_max)]
        self.df_bins = self.df_bins[(self.df_bins['Z_bird_TC_min'] >= self.current_thermal_core_spline['X'].x_min)
                                    & (self.df_bins['Z_bird_TC_max'] <= self.current_thermal_core_spline['X'].x_max)]


        #self.df = self.df.sort_values(['Z']).dropna()

        self.df = self.df[np.abs(self.df['curvature']) < 0.1]

        self.df_bins['X_bird_TC_avg'] = self.df_bins['rho_bird_TC_avg'] * np.cos(self.df_bins['phi_bird_TC_avg'])
        self.df_bins['Y_bird_TC_avg'] = self.df_bins['rho_bird_TC_avg'] * np.sin(self.df_bins['phi_bird_TC_avg'])
        bin_cols = ['bin_index_rho_bird_TC', 'bin_index_phi_bird_TC', 'bin_index_Z_bird_TC']
        df_agg = self.df[bin_cols + [f'd{col}dT_thermal_ground' for col in ['X', 'Y', 'Z']]].groupby(bin_cols).agg(
            dXdT_thermal_ground_avg=('dXdT_thermal_ground', 'mean'),
            dYdT_thermal_ground_avg=('dYdT_thermal_ground', 'mean'),
            dZdT_thermal_ground_avg=('dZdT_thermal_ground', 'mean'),
            count=('dXdT_thermal_ground', 'count'),
        )
        df_agg = df_agg[df_agg['count'] >= self.min_occupation_number]
        self.df_agg = pd.merge(self.df_bins.drop(columns='iteration'), df_agg,
                               left_on=['bin_index_rho_bird_TC', 'bin_index_phi_bird_TC', 'bin_index_Z_bird_TC'],
                               right_index=True, how='left')

        self.df = pd.merge(self.df, self.df_agg,
                           on=['bin_index_rho_bird_TC', 'bin_index_phi_bird_TC', 'bin_index_Z_bird_TC'],
                           how='left')

        self.df['dXdT_thermal_ground_res'] = self.df['dXdT_thermal_ground'] - self.df['dXdT_thermal_ground_avg']
        self.df['dYdT_thermal_ground_res'] = self.df['dYdT_thermal_ground'] - self.df['dYdT_thermal_ground_avg']
        self.df['dZdT_thermal_ground_res'] = self.df['dZdT_thermal_ground'] - self.df['dZdT_thermal_ground_avg']

    def preprocessing_agg(self):
        self.preprocess_thermal_rotation_agg()
        self.preprocess_thermal_profile_agg()

    def preprocess_thermal_rotation(self):
        thermal_rotation_interpolator = [LinearNDInterpolator(self.df[['X_bird_TC',
                                                                       'Y_bird_TC',
                                                                       'Z_bird_TC']].values,
                                                              self.df[f'd{col}dT_thermal_ground'].values
                                                              ) for col in ['X', 'Y']]

        def thermal_rotation(X, t=0):
            import numpy as np
            n_points = X.shape[0]

            result = np.zeros(shape=list(X[..., 0].shape) + [3])

            result[..., 0] = thermal_rotation_interpolator[0](X)
            result[..., 1] = thermal_rotation_interpolator[1](X)

            return result


        self.velocity_functions['rotation'] = thermal_rotation
        self.components.add('rotation')
        self.var_per_component['rotation'] = 'air'

        # ====================================      PROFILE         ================================================== #

    def preprocess_thermal(self):

        thermal_interpolator = [LinearNDInterpolator(self.df[['X_bird_TC',
                                                              'Y_bird_TC',
                                                              'Z_bird_TC']].values,
                                                     self.df[f'd{col}dT_thermal_ground'].values
                                                     ) for col in ['X', 'Y', 'Z']]

        def thermal_function(X, t=0):
            import numpy as np
            n_points = X.shape[0]

            result = np.zeros(shape=list(X[..., 0].shape) + [3])

            result[..., 0] = thermal_interpolator[0](X)
            result[..., 1] = thermal_interpolator[1](X)
            result[..., 2] = thermal_interpolator[2](X)

            return result

        self.velocity_functions['thermal'] = thermal_function
        self.components.add('thermal')
        self.var_per_component['thermal'] = 'air'

    def preprocess_turbulence(self):
        self.velocity_functions['turbulence'] = self.get_velocity_fluctuations
        self.components.add('turbulence')
        self.var_per_component['turbulence'] = 'air'

    def preprocess_wind(self):

        def wind_function(X, t, corrected=True):
            self.wind_index_list = 2
            X = np.array(X)

            input_vars = X[..., self.wind_index_list]

            return_array = np.empty(shape=list(X[..., 0].shape) + [3])

            return_array[..., 0] = self.wind_spline['X'](input_vars, extrapolate='linear')
            return_array[..., 1] = self.wind_spline['Y'](input_vars, extrapolate='linear')
            return_array[..., 2] = 0

            return return_array

        self.velocity_functions['wind'] = wind_function
        self.components.add('wind')

        self.velocity_functions_agg['wind'] = wind_function
        self.components_agg.add('wind')
        self.var_per_component['wind'] = 'ground'

        # ====================================      PROFILE         ================================================== #

    def preprocess_thermal_agg(self):

        def thermal_function_agg(X, t=0, extrapolate=True):
            import numpy as np
            old_shape = X.shape
            in_hull_mask = self.is_in_hull(X)
            # if extrapolate:
            # else:
            #     in_hull_mask = np.full(X.shape[:-1], fill_value=True)
            result = np.zeros(shape=list(X[..., 0].shape) + [3])
            wind = self.velocity_functions['wind'](X[in_hull_mask],t=t)
            result[in_hull_mask, 0] = self.current_interpolator_dict['dXdT_air_TC'](X[in_hull_mask]) + self.current_TC_spline['X'](X[in_hull_mask,2]) - wind[:, 0]
            result[in_hull_mask, 1] = self.current_interpolator_dict['dYdT_air_TC'](X[in_hull_mask]) + self.current_TC_spline['Y'](X[in_hull_mask,2]) - wind[:, 1]
            result[in_hull_mask, 2] = self.current_interpolator_dict['dZdT_air_TC'](X[in_hull_mask]) + self.current_TC_spline['Z'](X[in_hull_mask,2])
            if np.any(~in_hull_mask) and extrapolate:
                result[~in_hull_mask] = np.nan # self.get_extrapolated_values(X=X[~in_hull_mask],t=t)
            return result


        self.velocity_functions_agg['thermal'] = thermal_function_agg
        self.components_agg.add('thermal')

    def set_thermal_core(self):


        def get_thermal_core_function(X, t=0, d=0):

            return_array = np.stack([self.current_thermal_core_spline['X'](X, nu=d, extrapolate='linear'),
                                     self.current_thermal_core_spline['Y'](X, nu=d, extrapolate='linear')],
                                    axis=-1).astype(np.float32)
            return return_array

        self.thermal_core_function = get_thermal_core_function

    def get_velocity_agg(self, X, t=0, include=None, exclude=None, relative_to_ground=True, return_components=False):
        X = np.array(X)
        single_point = X.ndim == 1
        if  single_point:
            X = X.reshape((1, 3))
        default_output = np.zeros(shape=X.shape)
        if exclude is None:
            exclude = set([])
        elif isinstance(exclude, str):
            exclude = {exclude}
        elif isinstance(exclude, (list, np.ndarray)):
            exclude = set(exclude)

        if include is None:
            include = set(self.components_agg)
        elif isinstance(include, str):
            include = {include}
        elif isinstance(include, (list, np.ndarray)):
            include = set(include)

        if relative_to_ground:
            X_air = self.change_frame_of_reference(X, t, ground_to_air=True)
            X_ground = X.copy()
        else:
            X_ground = self.change_frame_of_reference(X, t, ground_to_air=False)
            X_air = X.copy()

        to_include = include - exclude
        components = {}
        for comp in to_include:
            if comp in self.components_agg:
                X = X_air if self.var_per_component[comp] == 'air' else X_ground
                components[comp] = self.velocity_functions_agg[comp](X=X, t=t)
            else:
                components[comp] = default_output

        result = self.add_velocities(np.array(list(components.values())).T).T
        if single_point:
            result = result[0]
            components = {k: v[0] for k, v in components.items()}
        if return_components:
            return result, components
        else:
            return result

    def get_velocity_fluctuations(self, *args, **kwargs):

        raw_velocity = self.get_velocity(*args, exclude=['wind', 'turbulence'], relative_to_ground=False, **kwargs)
        aggregated_velocity = self.get_velocity_agg(*args, exclude=['wind', 'turbulence'],relative_to_ground=False,  **kwargs)
        return raw_velocity - aggregated_velocity
