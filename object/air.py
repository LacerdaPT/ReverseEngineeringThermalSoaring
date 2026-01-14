import numbers
import os
import dill as pickle
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from types import LambdaType, FunctionType
from typing import Iterable, Union, Dict

import h5py
import logging

import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from data.get_data import load_decomposition_data
import numpy as np

from calc.auxiliar import SplineWrapper, parse_projection_string, get_gradient, get_periodic_linear_interpolator, \
    UnivariateSplineWrapper, cart_to_cyl_matrix, cart_to_cyl_point, cyl_to_cart_point
from calc.thermal import get_hierarchical_turbulence, merge_dfs_with_spline

from plotting.auxiliar import get_cross_section_meshgrid


def apply_along_class_method_decorator(function, *args, **kwargs):
    def wrapper(*args, **kwargs):
        self = args[0]
        X = kwargs.pop('X')
        t = kwargs.pop('t')
        if X.ndim == 1:
            X = X.reshape((1, -1))
            return np.apply_along_axis(lambda x: function(self, x,t, **kwargs) , -1, X)[0]
        else:
            return np.apply_along_axis(lambda x: function(self, x, t, **kwargs), -1, X)
    return wrapper

def apply_along_decorator(function, *args, **kwargs):
    def wrapper(*args, **kwargs):
        X = kwargs.pop('X')
        t = kwargs.pop('t')
        if X.ndim == 1:
            X = X.reshape((1, -1))
            return np.apply_along_axis(lambda x: function(x,t, **kwargs) , -1, X)[0]
        else:
            return np.apply_along_axis(lambda x: function(x, t, **kwargs), -1, X)
    return wrapper

logger = logging.getLogger(__name__)

class AirVelocityFieldBase(ABC):

    config = {'t_start_max': 300,
              'time_resolution': 10,
              'z_max_limit': 3000,
              'dt_to_iterate': 0.1,
              'dt_to_save': 1,
              'thermal_core_initial_position': [0, 0, 0],
              }
    def __init__(self):
        self.components = set()
        self.var_per_component = {}
        self.velocity_functions = {}
        self.thermal_core_function = None

    def get_thermal_core(self, z, t=0, **kwargs):
        if isinstance(z, Iterable):
            z = np.array(z).copy()
            n_dim = z.ndim
            if n_dim == 1:
                n_points = z.shape[0]
                #z = z[..., np.newaxis]

            negative_z_indices = np.argwhere(z < 0).flatten()
        else:
            if z < 0:
                core = self.config['thermal_core_initial_position'][:2]
                return core

        core = self.thermal_core_function(z, t=t, **kwargs)
        return core
    def change_frame_of_reference(self, X, t, ground_to_air=True):
        core = self.get_thermal_core(X[..., -1], t)
        if ground_to_air:
            X_transformed = np.array([X[..., 0] - core[..., 0], X[..., 1] - core[..., 1], X[..., -1]])

        else:  # air to ground
            X_transformed = np.array([X[..., 0] + core[..., 0], X[..., 1] + core[..., 1], X[..., -1]])

        X_transformed = np.moveaxis(X_transformed, 0, -1)
        return X_transformed
    def get_velocity(self, X, t=0, include=None, exclude=None, relative_to_ground=True, return_components=False):
        X = np.array(X)
        single_point = X.ndim == 1
        if single_point:
            X = X.reshape((1, 3))
        default_output = np.zeros(shape=X.shape)
        if exclude is None:
            exclude = set([])
        elif isinstance(exclude, str):
            exclude = {exclude}
        elif isinstance(exclude, (list, np.ndarray)):
            exclude = set(exclude)

        if include is None:
            include = set(self.components)
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
            if comp in self.components:
                X = X_air if self.var_per_component[comp] == 'air' else X_ground
                components[comp] = self.velocity_functions[comp](X=X, t=t)
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

    @abstractmethod
    def set_thermal_core(self):
        pass
    @staticmethod
    def add_velocities(velocities_array):
        return np.sum(velocities_array, axis=-1)  # sum on scales


class AirVelocityField(AirVelocityFieldBase):

    def __init__(self, air_parameters, **config_args):
        super().__init__()
        self.turbulence_interpolator_parameters = None
        self.wind_index_list = []
        self.wind_var_list = []
        self.wind_time_var = False
        self._air_parameters = deepcopy(air_parameters)
        self.rho = air_parameters['rho']
        self.thermal_rotation_function = None
        self.thermal_profile_function = None
        self.wind_function = None
        self.turbulence_function = None
        self.thermal_core_function = None


        # properties

        self.config = AirVelocityField.config

        self.config.update(config_args)
        # Adding an extra resolution unit prevents issues with undefined thermal core
        # when time is close to the end of the simulation.
        self.config['t_start_max'] = self.config['t_start_max'] + self.config['time_resolution']

        self.turbulence_parameters = {}
        self.turbulence_interpolators = {}

        self.current_turbulence_time_limits = None

        self.legacy_handling()

        self.preprocessing()
        # self.get_velocity = np.vectorize(self.get_velocity)

    def legacy_handling(self):

        if isinstance(self._air_parameters['wind'], (list, np.ndarray)):
            self._air_parameters['wind'] = np.array(self._air_parameters['wind'])

            if self._air_parameters['wind'].ndim == 1:  # Constant wind
                self._air_parameters['wind'] = {'values': self._air_parameters['wind'],
                                                'used_vars': ''
                                                }
            elif self._air_parameters['wind'].ndim == 2:  # Z dependent by default
                z_array = np.linspace(0, self.config['z_max_limit'], self._air_parameters['wind'].shape[0])
                self._air_parameters['wind'] = {'values': self._air_parameters['wind'],
                                                'XYZT_values': z_array,
                                                'used_vars': 'z'}

    def preprocessing(self):
        self.preprocess_wind()
        self.preprocess_thermal_rotation()
        self.preprocess_thermal_profile()
        self.preprocess_turbulence()
        self.set_thermal_core()

    def preprocess_wind(self):
        # Custom Function
        if 'function' in self._air_parameters['wind']:
            @apply_along_decorator
            def wind_function(X, t=0):
                return self._air_parameters['wind']['function'](X, t, **self._air_parameters['wind']['args']) + [0]

        # Data driven wind
        else:
            # Check which variables are used in the data driven wind
            for i, coord in enumerate(self._air_parameters['wind']['used_vars'].lower()):
                index = ['x', 'y', 'z', 't'].index(coord)

                if coord == 't':
                    self.wind_time_var = True
                else:
                    self.wind_var_list.append(coord)
                    self.wind_index_list.append(index)

            n_used_vars = (len(self.wind_index_list) + int(self.wind_time_var))

            if n_used_vars == 0:
                @apply_along_decorator
                def wind_function(X, t=0): # Constant Wind
                    result = np.empty(shape=(3))
                    result[[0,1]] = self._air_parameters['wind']['values']
                    result[2] = 0

                    return result
            elif n_used_vars == 1:
                # Unidimensional

                wind_spline = SplineWrapper(X=self._air_parameters['wind']['values'].T,
                                            y=self._air_parameters['wind']['XYZT_values'], degree=3)
                wind_spline.fit()

                @apply_along_decorator
                def wind_function(X, t):

                    X = np.array(X)
                    n_points = X.shape[0]
                    if self.wind_time_var:
                        input_vars = np.hstack([X[self.wind_index_list], [t]])
                    else:
                        input_vars = X[self.wind_index_list]

                    return_array = np.empty((3), dtype=np.float32)
                    wind_values = wind_spline(input_vars)
                    return_array[0] = wind_values[0]
                    return_array[1] = wind_values[1]
                    return_array[2] = 0

                    return return_array
            else:  # Multidimensional
                interpolator = [LinearNDInterpolator(self._air_parameters['wind']['XYZT_values'],
                                                     self._air_parameters['wind']['values'][:, i]
                                                     ) for i in [0, 1]]

                interpolator = np.array(interpolator)
                @apply_along_decorator
                def wind_function(X: np.ndarray, t: float = 0) -> np.ndarray:
                    X = np.array(X)
                    if self.wind_time_var:
                        input_vars = np.hstack([X[self.wind_index_list], [t]])
                    else:
                        input_vars = X[self.wind_index_list]

                    return_array = np.array([interpolator[0](input_vars)[0],
                                             interpolator[1](input_vars)[0],
                                             0]
                                            ).astype(np.float32)

                    if np.any(np.isnan(return_array)):
                        pass
                    return return_array

        self.velocity_functions['wind'] = wind_function
        self.components.add('wind')
        self.var_per_component['wind'] = 'ground'

    def preprocess_thermal_rotation(self):
        # ====================================      ROTATION        ================================================== #
        if ('rotation' not in self._air_parameters['thermal']) or self._air_parameters['thermal']['rotation'] is None:
            thermal_rotation = lambda X, t: np.zeros(shape=np.array(X).shape)

        elif isinstance(self._air_parameters['thermal']['rotation'], (float, int)):
            @apply_along_decorator
            def thermal_rotation(X, t):
                import numpy as np
                rotation_magnitude = self._air_parameters['thermal']['rotation']


                x, y, z = X
                # Local coordinates
                r = np.linalg.norm([x, y])
                theta = np.arctan2(y, x)
                return [-rotation_magnitude * np.sin(theta),
                        rotation_magnitude * np.cos(theta),
                        0]

        else:  # elif isinstance(self._air_parameters['thermal']['rotation'], (LambdaType, FunctionType)):
            @apply_along_decorator
            def thermal_rotation(X, t=0):
                import numpy as np

                x, y, z = X
                # Local coordinates
                r = np.linalg.norm([x, y])
                theta = np.arctan2(y, x)

                return self._air_parameters['thermal']['rotation']['function'](r, theta, z, t,
                                                                               **self._air_parameters['thermal']['rotation']['args'])  + [0.0]


        self.velocity_functions['rotation'] = thermal_rotation
        self.components.add('rotation')
        self.var_per_component['rotation'] = 'air'
        # ====================================      PROFILE         ================================================== #

    def preprocess_thermal_profile(self):
        if ('profile' not in self._air_parameters['thermal']) or self._air_parameters['thermal']['profile'] is None:
            thermal_profile = lambda X, t: np.zeros(shape=(np.array(X).shape[0], 3))
        else:

            @apply_along_decorator
            def thermal_profile(X, t=0):
                import numpy as np

                x_relative, y_relative, z = X
                # Local coordinates

                r = np.linalg.norm([x_relative, y_relative])
                theta = np.arctan2(y_relative, x_relative)
                profile_args = self._air_parameters['thermal']['profile']['args']
                return [0, 0,
                        self._air_parameters['thermal']['profile']['function'](r, theta, z, t, **profile_args)]

        self.velocity_functions['thermal'] = thermal_profile
        self.components.add('thermal')
        self.var_per_component['thermal'] = 'air'

    def preprocess_turbulence(self):
        if 'turbulence' not in self._air_parameters.keys():
            self.turbulence_function = lambda X, t: np.zeros(shape=np.array(X).shape)
        elif not self._air_parameters['turbulence']:
            self.turbulence_function = lambda X, t: np.zeros(shape=np.array(X).shape)
        else:

            self.turbulence_parameters = deepcopy(self._air_parameters['turbulence'])
            self.turbulence_parameters['largest_spatial_scale'] = max(self.turbulence_parameters['scales'])
            self.turbulence_parameters['largest_velocity_scale'] = self.turbulence_parameters['scales'][self.turbulence_parameters['largest_spatial_scale']]

            # Normalize the scales
            if self.turbulence_parameters['normalization'] is None:
                normalization = self.turbulence_parameters['largest_velocity_scale']
            else:
                normalization = self.turbulence_parameters['normalization']

            if normalization:
                norm = np.sum(list(self.turbulence_parameters['scales'].values())) / normalization

                for spatial_scale in self.turbulence_parameters['scales'].keys():
                    self.turbulence_parameters['scales'][spatial_scale] /= norm

            with h5py.File(self.turbulence_parameters['data_path'], 'r') as hfd:
                n_time, _, n_x, n_y, n_z = hfd['data'].shape
                self.turbulence_parameters['time_limits'] = (0, n_time - 1)
                self.turbulence_parameters['n_grid'] = {'x': n_x // self.turbulence_parameters['downsampling_factor'],
                                                        'y': n_y // self.turbulence_parameters['downsampling_factor'],
                                                        'z': n_z // self.turbulence_parameters['downsampling_factor']}

        self.reset_turbulence_function(t=0)

    def reset_turbulence_function(self, t):

        if ('turbulence' not in self._air_parameters.keys()) or (self._air_parameters['turbulence'] is None):
            return
        else:
            del self.turbulence_interpolators

        logger.debug('resetting turbulence')
        # Fit interpolators
        largest_spatial_scale = self.turbulence_parameters['largest_spatial_scale']
        largest_velocity_scale = self.turbulence_parameters['scales'][largest_spatial_scale]

        L = {'x': largest_spatial_scale,
             'y': largest_spatial_scale,
             'z': largest_spatial_scale}

        t_min = np.floor(t).astype(int)
        t_max = int(t_min + self.turbulence_parameters['reset_period'])

        t_max = np.min([self.turbulence_parameters['time_limits'][-1],
                        t_max])

        self.turbulence_interpolator_parameters = {'spatial_scale': largest_spatial_scale,
                                                   'velocity_scale': largest_velocity_scale}
        if t_max == t_min:
            t_min = t_min - 1

        logger.debug(f'reading data file from {t_min:.1f} to {t_max:.1f}')
        with h5py.File(self.turbulence_parameters['data_path'], 'r') as hfd:
            DS = self.turbulence_parameters['downsampling_factor']
            turbulence_dataframe = hfd['data'][t_min: t_max + 1, :, ::DS, ::DS, ::DS]

        turbulence_dataframe = largest_velocity_scale * turbulence_dataframe
        logger.debug('done reading data file')
        self.current_turbulence_time_limits = [t_min, t_max]

        limits = [self.current_turbulence_time_limits,
                  [0, L['x']],
                  [0, L['y']],
                  [0, L['z']]]
        logger.debug(f'Interpolating turbulence data: spatial_grid:{self.turbulence_parameters["n_grid"]}, time: {t_max - t_min}')
        self.turbulence_interpolators = [get_periodic_linear_interpolator(turbulence_dataframe[:, i, :, :, :],
                                                                          limits=limits)
                                         for i in range(3)]
        del turbulence_dataframe
        logger.debug('Done interpolating turbulence data')

        def turbulence_function(X, t, scales=None, weighting_method=None, weighting_method_miltiplier=50,
                                velocity_component=None):
            periodic_time = t % self.turbulence_parameters['time_limits'][-1]
            if isinstance(velocity_component, int):
                list_of_components = [velocity_component]
            elif isinstance(velocity_component, (list, np.ndarray)):
                list_of_components = velocity_component.copy()
            else:
                list_of_components = np.arange(len(self.turbulence_interpolators))

            if weighting_method is None:
                weighting_method = self.turbulence_parameters['weighting_method']

            if not (self.current_turbulence_time_limits[0] <= periodic_time <= self.current_turbulence_time_limits[1]):
                self.reset_turbulence_function(periodic_time)

            if scales is None:
                list_of_scales = self.turbulence_parameters['scales'].keys()
            elif isinstance(scales, (numbers.Integral, numbers.Real)):
                list_of_scales = [scales]
            else:
                list_of_scales = scales

            scales_dict = {s: self.turbulence_parameters['scales'][s] for s in list_of_scales}
            interpolators_to_use = [self.turbulence_interpolators[i] for i in list_of_components]
            results = get_hierarchical_turbulence(X, periodic_time,
                                                  scales_dict=scales_dict,
                                                  list_of_interpolator=interpolators_to_use,
                                                  interpolator_parameters=self.turbulence_interpolator_parameters)

            sum = self.add_velocities(results)

            # Weighting
            if weighting_method == 'value':
                thermal_profile_value = self.velocity_functions['thermal'](X=X, t=t)
                thermal_profile_value = thermal_profile_value[-1]
                return_value = np.abs(thermal_profile_value) * sum
            elif weighting_method == 'gradient':

                #sigma = 15
                #thermal_profile_value = self.thermal_profile_function(X=X, t=t)
                #thermal_profile_gradient = - (np.linalg.norm(X[:2]) / sigma ** 2) * thermal_profile_value
                thermal_profile_gradient = self.get_velocity_gradient(X, t, include='thermal', N=5)

                return_value = weighting_method_miltiplier * np.linalg.norm(thermal_profile_gradient) * sum

            else:
                return_value = sum

            return return_value

        self.velocity_functions['turbulence'] = turbulence_function
        self.components.add('turbulence')
        self.var_per_component['turbulence'] = 'air'

    def set_thermal_core(self):
        zt_array = np.empty((0, 2), float)
        list_of_paths = np.empty((0, 3), float)
        first_t_array = []
        first_path = []
        for t_start in range(0, self.config['t_start_max'], self.config['time_resolution']):
            current_point = np.array(self.config['thermal_core_initial_position'])
            current_path = np.array([current_point], dtype=float)
            current_t_array = np.array([t_start], dtype=float)
            logger.debug(f't={t_start}')
            t = t_start
            while current_point[-1] < self.config['z_max_limit']:
                wind_function = self.velocity_functions['wind']
                thermal_profile_function = self.velocity_functions['thermal']
                dv = wind_function(X=np.array([current_point]),
                                        t=t)[0]
                if np.any(np.isnan(dv)):
                    logger.debug(f'{current_point=}, {t=}')
                    break
                dv += thermal_profile_function(X=np.array([0, 0, current_point[2]]), t=t
                                               )

                dl = dv * self.config['dt_to_iterate']
                current_point = current_point + dl
                t = t + self.config['dt_to_iterate']
                if np.isclose(t % self.config['dt_to_save'], 0, atol=1e-4) or np.isclose(t % self.config['dt_to_save'], 1, atol=1e-4):
                    current_path = np.append(current_path, [current_point], axis=0)
                    current_t_array = np.append(current_t_array, [t], axis=0)

            list_of_paths = np.append(list_of_paths, current_path, axis=0)
            z_array = current_path[:, -1]

            current_zt = np.array([z_array, current_t_array]).T
            zt_array = np.append(zt_array, current_zt, axis=0)

            if len(first_t_array) == 0:
                first_t_array = current_t_array.copy()
                first_path = current_path.copy()

        # This fills the list with the instants that are not expressed in the previous iterations, e.g. thermal core
        # for z=0.1 only exists for t=0, for z=200 only exists for, say, t > 50 for a thermal with v_max = 4 m/s
        # This fills the gaps as if before t=0 everything was stationary.
        logger.debug('doing before')
        time_step = int(round(self.config['time_resolution'] / self.config['dt_to_save']))
        for (x, y, z), t_max in zip(first_path[1::time_step],
                                    first_t_array[1::time_step]):  # the first element would be repeated
            logger.debug(f't={t_max}')
            current_t_array = np.arange(0, t_max, self.config['dt_to_save'])
            current_path = np.array([[x, y, z]] * len(current_t_array))

            list_of_paths = np.append(list_of_paths, current_path, axis=0)
            z_array = current_path[:, -1]

            current_zt = np.array([z_array, current_t_array]).T
            zt_array = np.append(zt_array, current_zt, axis=0)

        logger.debug(f'using {len(zt_array)} points')
        thermal_core_interpolator = [LinearNDInterpolator(zt_array,
                                                          list_of_paths[:, i]
                                                          ) for i in [0, 1]]

        # thermal_core = SplineWrapper(current_path[:, :2].T, current_path[:, -1].T)
        # thermal_core.fit()

        def thermal_core_function(z: float, t: float = 0) -> np.ndarray:

            input_vars = [z, t]

            return_array = np.stack([thermal_core_interpolator[0](*input_vars),
                                      thermal_core_interpolator[1](*input_vars)], axis=-1).astype(np.float32)

            if np.any(np.isnan(return_array)):
                pass
            return return_array

        self.thermal_core_function = thermal_core_function



    def get_velocity_gradient(self, X, t, include, delta=1.0, N=11):

        x0, y0, z0 = X
        if N % 2 == 0:
            N = N + 1
        x = np.linspace(x0 - delta, x0 + delta, N, endpoint=True)
        y = np.linspace(y0 - delta, y0 + delta, N, endpoint=True)
        z = np.linspace(z0 - delta, z0 + delta, N, endpoint=True)
        d = 2 * delta / N
        x_mg, y_mg, z_mg = np.meshgrid(x, y, z)

        X_array = np.array(list(zip(x_mg.flatten(),
                                    y_mg.flatten(),
                                    z_mg.flatten())))

        f_values = self.get_velocity(X_array, t=t, include=include, relative_to_ground=False)[:, -1]
        f_values = f_values.reshape(x_mg.shape)
        grad = np.gradient(f_values, d)  # , axis=[0,1,2]
        N_return = N // 2 + 1
        gradient_return = [grad[i][N_return, N_return, N_return] for i in range(3)]
        return gradient_return


class AirVelocityFieldVisualization(object):
    def __init__(self, air_velocity_obj: AirVelocityFieldBase):
        self.air_velocity_field = air_velocity_obj

    @classmethod
    def from_air_parameters(cls, air_parameters, **config_args):
        air_velocity_obj = AirVelocityField(air_parameters, **config_args)
        return cls(air_velocity_obj=air_velocity_obj)

    def plot_thermal_profile(self, ax, plot_type, max_rho, Z_level, t_value=0, resolution=30,
                             include_turbulence=False, add_colorbar=True,  cross_section_type='XY',
                             kwargs=None, velocity_kwargs=None):

        if kwargs is None:
            kwargs = {}
        if velocity_kwargs is None:
            velocity_kwargs = {'relative_to_ground': False}
        limits = [[-max_rho, max_rho],
                  [-max_rho, max_rho]]
        to_include = ['thermal']
        if include_turbulence:
            to_include += ['turbulence']
        ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=Z_level,
                                                                  include=to_include, t_value=t_value,
                                                                  cross_section_type=cross_section_type,
                                                                  plot_type=plot_type,
                                                                  plotting_function=2,
                                                                  color_function=None,
                                                                  resolution=resolution,
                                                                  plot_kwargs=kwargs,
                                                                  velocity_kwargs=velocity_kwargs,
                                                                  add_colorbar=add_colorbar)
        return ax, artist, cbar

    def plot_thermal_core_3d(self, ax, t_value=0, resolution=30, Z_array=None, kwargs=None):

        if kwargs is None:
            kwargs = {}

        if Z_array is None:
            z_max=1000
            Z_array = np.linspace(0.1, z_max, resolution)
        t_array = t_value * np.ones(shape=Z_array.shape)

        # ZT_array = np.stack([Z_array, t_array], axis=-1)
        XY_core = self.air_velocity_field.get_thermal_core(z=Z_array, t=t_value)

        core = np.hstack([XY_core, Z_array.reshape((Z_array.shape[0], 1))])

        artist = ax.plot(xs=core[:, 0],
                         ys=core[:, 1],
                         zs=core[:, 2], **kwargs)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        ax.plot(core[:, 0], core[:, 1], zs=np.min(core[:,-1]), zdir='z', color='k', alpha=0.3)

        return ax, artist

    def plot_thermal_rotation(self, ax, max_rho, Z_level, t_value, resolution,  include_turbulence=False,
                              add_colorbar=True, kwargs=None,  velocity_kwargs=None):

        if kwargs is None:
            kwargs = {}

        if velocity_kwargs is None:
            velocity_kwargs = {'relative_to_ground': False}
        limits = [[-max_rho, max_rho],
                  [-max_rho, max_rho]]
        to_include = ['rotation']
        if include_turbulence:
            to_include += ['turbulence']
        ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=Z_level,
                                                                  include=to_include, t_value=t_value,
                                                                  cross_section_type='XY',
                                                                  plot_type='streamplot',
                                                                  plotting_function=[0, 1],
                                                                  color_function=[0, 1],
                                                                  resolution=resolution,
                                                                  plot_kwargs=kwargs,
                                                                  velocity_kwargs=velocity_kwargs,
                                                                  add_colorbar=add_colorbar)

        return ax, artist, cbar

    def plot_turbulence(self, ax, plot_type, limits, t=0, cross_section_type='XY', section_level=0, resolution=20,
                        add_colorbar=True, plotting_function=None, color_function=None,
                        velocity_function_kwargs=None, plot_kwargs=None):

        if color_function is None:
            color_function = [0, 1]
        if plotting_function is None:
            plotting_function = [0, 1]
        if velocity_function_kwargs is None:
            velocity_function_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=section_level,
                                                                  include='turbulence', t_value=t,
                                                                  cross_section_type=cross_section_type,
                                                                  plot_type=plot_type,
                                                                  plotting_function=plotting_function,
                                                                  color_function=color_function,
                                                                  resolution=resolution,
                                                                  plot_kwargs=plot_kwargs,
                                                                  velocity_kwargs=velocity_function_kwargs,
                                                                  add_colorbar=add_colorbar)
        return ax, artist, cbar

    def plot_wind(self, ax, limits, t=0, section_level=0, resolution=20, cross_section_type='XZ',
                  add_colorbar=True, kwargs=None):

        if kwargs is None:
            kwargs = {}

        ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=section_level,
                                                                  include='wind', t_value=t,
                                                                  cross_section_type=cross_section_type,
                                                                  plot_type='streamplot',
                                                                  plotting_function=lambda v: [v[0], 0],
                                                                  color_function=0,
                                                                  #linewidth_function=lambda v: 3 *v[0],
                                                                  resolution=resolution,
                                                                  plot_kwargs=kwargs,
                                                                  velocity_kwargs={'relative_to_ground': True},
                                                                  add_colorbar=add_colorbar)

        ax.set_aspect('equal')

        return ax, artist, cbar

    def plot_components(self, ax, limits, components, t=0, section_level=0, resolution=20, cross_section_type='XZ',
                        plot_type='streamplot', add_colorbar=True, plotting_function=None, color_function=None,
                        velocity_kwargs=None, kwargs=None):

        if velocity_kwargs is None:
            velocity_kwargs = {}
        if kwargs is None:
            kwargs = {}

        ax, artist, cbar = self._plot_per_component_and_plot_type(ax, limits=limits, section_value=section_level,
                                                                  include=components, t_value=t,
                                                                  cross_section_type=cross_section_type,
                                                                  plot_type=plot_type,
                                                                  plotting_function=plotting_function,
                                                                  color_function=color_function,
                                                                  resolution=resolution,
                                                                  plot_kwargs=kwargs,
                                                                  velocity_kwargs=velocity_kwargs,
                                                                  add_colorbar=add_colorbar)

        ax.set_aspect('equal')

        return ax, artist, cbar

    def plot_all(self, z_value=200, section_value=0, t_value=0, include_turbulence=False, ax=None,cbar_ax=None,
                 kwargs_list=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(12, 10), constrained_layout=True)
            ax = [fig.add_subplot(221),
                  fig.add_subplot(222, projection='3d'),
                  fig.add_subplot(223, projection='3d'),
                  fig.add_subplot(224)]
        else:
            fig = ax[0].get_figure()
        if cbar_ax is None:
            cbar_ax = [None] * len(ax)
        if kwargs_list is None:
            kwargs_list = [{}] * len(ax)
        # ==============================================================================================================
        # ========================================      WIND      ======================================================
        # ==============================================================================================================
        limits = [[10, 1000],
                  [0.1, 1000]]

        _, art, cbar = self.plot_wind(ax=ax[0], limits=limits, t=t_value, section_level=section_value,
                                      add_colorbar=cbar_ax[0], kwargs=kwargs_list[0])
        ax[0].set_aspect('equal')
        ax[0].set_title('Wind')
        if cbar is not None:
            cbar.set_label('$V_{Horizontal}$ (m/s)')


        # ==============================================================================================================
        # ===================================      THERMAL PROFILE       ===============================================
        # ==============================================================================================================

        _, art, cbar = self.plot_thermal_profile(ax=ax[1], plot_type='surface', max_rho=100, Z_level=z_value, t_value=t_value,
                                                 resolution=100, include_turbulence=include_turbulence,
                                                 add_colorbar=cbar_ax[1],
                                                 kwargs=kwargs_list[1] | {'linewidth': 0}, )
        if cbar is not None:
            cbar.set_label('$V_{Z}$ (m/s)')

        ax[1].set_title('Thermal Profile')
        ax[1].set_zlabel('$V_Z$ (m/s)')

        # ==============================================================================================================
        # ===================================      THERMAL CORE       ==================================================
        # ==============================================================================================================
        _, art = self.plot_thermal_core_3d(ax=ax[2], t_value=t_value, kwargs=kwargs_list[2] )

        ax[2].set_title('Thermal Core')

        # ==============================================================================================================
        # ===================================      THERMAL ROTATION      ===============================================
        # ==============================================================================================================
        _, art, cbar = self.plot_thermal_rotation(ax=ax[3], max_rho=100, Z_level=z_value, t_value=t_value, resolution=15,
                                                  add_colorbar=cbar_ax[3], include_turbulence=include_turbulence, kwargs=kwargs_list[3])
        ax[3].set_aspect('equal')
        ax[3].set_title('Thermal Rotation')
        if cbar:
            cbar.set_label('$V_{Horizontal}$ (m/s)')
        fig.suptitle(f't={t_value:.1f}')
        return fig, ax

    def plot_all_over_time(self, t_steps, destination_folder):

        import matplotlib.pyplot as plt
        plt.ioff()
        n_digits = np.ceil(np.log10(max(t_steps))).astype(int)
        try:
            os.makedirs(destination_folder)
        except FileExistsError:
            pass

        for t in t_steps:
            fig, ax = self.plot_all(t_value=t)
            filename = 't=' + ('0' * n_digits + str(t))[-n_digits:]
            full_path = os.path.join(destination_folder, f'{filename}.png')
            fig.savefig(full_path)
            print(f'saved to {full_path}')
            plt.close(fig)

    def _plot_per_component_and_plot_type(self, ax, limits, section_value, include,
                                          plot_type, t_value, plotting_function, color_function=None, linewidth_function=None,
                                          cross_section_type='XY',
                                          resolution=15, velocity_kwargs=None, plot_kwargs=None, add_colorbar=True):
        if velocity_kwargs is None:
            velocity_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        (XYZ_meshgrid,
         plotting_vars, plotting_indices,
         section_var, section_index) = get_cross_section_meshgrid(limits=limits,
                                                                  cross_section_type=cross_section_type,
                                                                  n_points=resolution,
                                                                  section_value=section_value)

        mg_shape = XYZ_meshgrid.shape
        X_meshgrid = XYZ_meshgrid[..., plotting_indices[0]]
        Y_meshgrid = XYZ_meshgrid[..., plotting_indices[1]]
        Z_meshgrid = XYZ_meshgrid[..., section_index]

        XYZ_flat = XYZ_meshgrid.reshape((mg_shape[0] * mg_shape[1], mg_shape[-1]))
        velocities_array = self.air_velocity_field.get_velocity(XYZ_flat, t=t_value, include=include, **velocity_kwargs)

        velocities_array = velocities_array.reshape(XYZ_meshgrid.shape)
        #plotting_function

        if isinstance(plotting_function, (FunctionType, LambdaType)):
            plot_array = np.apply_along_axis(plotting_function, -1, velocities_array)
        elif isinstance(plotting_function, Iterable):
            plot_array = velocities_array[..., plotting_function]
        else:
            plot_array = velocities_array[..., plotting_function]

        if isinstance(color_function, (FunctionType, LambdaType)):
            color_array = np.apply_along_axis(color_function, -1, velocities_array)
        elif np.isscalar(color_function):
            color_array = velocities_array[..., color_function]
        elif color_function is None:
            color_array = None
        else:  # Defaults to norm of the vector with indices in plotting_function,
            # e.g., plotting_function=[0, 1] will yield the horizontal velocity
            color_array = np.linalg.norm(velocities_array[..., color_function], axis=-1)

        if isinstance(linewidth_function, (FunctionType, LambdaType)):
            linewidth_array = np.apply_along_axis(linewidth_function, -1, velocities_array)
        elif np.isscalar(linewidth_function):
            linewidth_array = velocities_array[..., color_function]
        elif linewidth_function is None:
            linewidth_array = None
        else:  # Defaults to norm of the vector with indices in plotting_function,
            # e.g., plotting_function=[0, 1] will yield the horizontal velocity
            linewidth_array = np.linalg.norm(velocities_array[..., linewidth_function], axis=-1)

        if plot_type == 'quiver':
            if np.allclose(plot_array, 0):
                artist = None
            else:
                if color_array is not None:
                    artist = ax.quiver(X_meshgrid,
                                       Y_meshgrid,
                                       plot_array[..., 0],
                                       plot_array[..., 1],
                                       color_array,
                                       **plot_kwargs)
                else:
                    artist = ax.quiver(X_meshgrid,
                                       Y_meshgrid,
                                       plot_array[..., 0],
                                       plot_array[..., 1],
                                       **plot_kwargs)
        elif plot_type == 'contour':
            if plot_array.ndim > 2:
                plot_array = np.linalg.norm(plot_array, axis=-1)
            artist = ax.contourf(X_meshgrid,
                                 Y_meshgrid,
                                 plot_array,
                                 **plot_kwargs
                                 )
        elif plot_type == 'streamplot':
            if np.allclose(color_array, 0):
                artist = None
            else:
                artist = ax.streamplot(x=X_meshgrid,
                                   y=Y_meshgrid,
                                   u=plot_array[..., 0],
                                   v=plot_array[..., 1],
                                       linewidth=linewidth_array,
                                   color=color_array, **plot_kwargs)
        elif plot_type == 'imshow':
            if plot_array.ndim > 2:
                plot_array = np.linalg.norm(plot_array, axis=-1)
            artist = ax.imshow(plot_array, origin='lower', extent=(limits[0][0],
                                                                    limits[0][1],
                                                                    limits[1][0],
                                                                    limits[1][1]),
                               **plot_kwargs
                               )
        elif plot_type == 'surface':
            if plot_array.ndim > 2:
                plot_array = np.linalg.norm(plot_array, axis=-1)
            artist = ax.plot_surface(X_meshgrid,
                                     Y_meshgrid,
                                     plot_array, facecolors=color_array, **plot_kwargs)

        if add_colorbar and (artist is not None):
            import matplotlib.pyplot as plt
            if isinstance(add_colorbar, bool):
                if plot_type == 'streamplot':
                    cbar = plt.colorbar(artist.lines, ax=ax)
                else:
                    cbar = plt.colorbar(artist, ax=ax)
            else:
                if plot_type == 'streamplot':
                    cbar = plt.colorbar(artist.lines, cax=add_colorbar)
                else:
                    cbar = plt.colorbar(artist, cax=add_colorbar)

        else:
            cbar = None

        ax.set_xlabel(plotting_vars[0] + ' (m)')
        ax.set_ylabel(plotting_vars[1] + ' (m)')
        return ax, artist, cbar



class DecomposedAirVelocityField(AirVelocityFieldBase):


    def __init__(self, df, df_bins, df_splines, aggregated=True, min_occupation_number=4, **config_args):
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

    def preprocess_thermal_profile(self):
        df_filter = self.df[~self.df['dZdT_thermal_ground'].isna()]
        thermal_profile_interpolator = LinearNDInterpolator(df_filter[['X_bird_TC',
                                                                       'Y_bird_TC',
                                                                       'Z_bird_TC']].values,
                                                            df_filter[f'dZdT_thermal_ground'].values
                                                            )

        def thermal_profile(X, t=0):
            import numpy as np
            n_points = X.shape[0]

            result = np.zeros(shape=list(X[..., 0].shape) + [3])

            result[..., 2] = thermal_profile_interpolator(X)

            return result

        self.velocity_functions['thermal'] = thermal_profile
        self.components.add('thermal')
        self.var_per_component['thermal'] = 'air'

    def preprocess_turbulence(self):
        self.velocity_functions['turbulence'] = self.get_velocity_fluctuations
        self.components.add('turbulence')
        self.var_per_component['turbulence'] = 'air'

    def preprocess_wind(self):

        def wind_function(X, t, corrected=False):
            self.wind_index_list = 2
            X = np.array(X)

            input_vars = X[..., self.wind_index_list]

            return_array = np.empty(shape=list(X[..., 0].shape) + [3])

            return_array[..., 0] = self.wind_spline['X'](input_vars)
            return_array[..., 1] = self.wind_spline['Y'](input_vars)
            if corrected:
                return_array[..., 0] += self.wind_correction_spline['X'](X[..., -1], extrapolate='linear')
                return_array[..., 1] += self.wind_correction_spline['Y'](X[..., -1], extrapolate='linear')
            return_array[..., 2] = 0

            return return_array

        self.velocity_functions['wind'] = wind_function
        self.components.add('wind')

        self.velocity_functions_agg['wind'] = wind_function
        self.components_agg.add('wind')
        self.var_per_component['wind'] = 'ground'

    def preprocess_thermal_rotation_agg(self):
        df_filter = self.df_agg[~self.df_agg['dXdT_thermal_ground_avg'].isna()]
        df_filter = df_filter[~df_filter['dYdT_thermal_ground_avg'].isna()]
        thermal_rotation_interpolator = [LinearNDInterpolator(df_filter[['X_bird_TC_avg',
                                                                           'Y_bird_TC_avg',
                                                                           'Z_bird_TC_avg']].values,
                                                              df_filter[f'd{col}dT_thermal_ground_avg'].values
                                                              ) for col in ['X', 'Y']]
        def thermal_rotation(X, t=0):
            import numpy as np
            n_points = X.shape[0]

            result = np.zeros(shape=list(X[..., 0].shape) + [3])

            result[..., 0] = thermal_rotation_interpolator[0](X)
            result[..., 1] = thermal_rotation_interpolator[1](X)

            return result


        self.velocity_functions_agg['rotation'] = thermal_rotation
        self.components_agg.add('rotation')

        # ====================================      PROFILE         ================================================== #

    def preprocess_thermal_profile_agg(self):
        df_filter = self.df_agg[~self.df_agg['dZdT_thermal_ground_avg'].isna()]
        thermal_profile_interpolator = LinearNDInterpolator(df_filter[['X_bird_TC_avg',
                                                                         'Y_bird_TC_avg',
                                                                         'Z_bird_TC_avg']].values,
                                                            df_filter[f'dZdT_thermal_ground_avg'].values
                                                            )

        def thermal_profile(X, t=0):
            import numpy as np

            result = np.zeros(shape=list(X[..., 0].shape) + [3])

            result[..., 2] = thermal_profile_interpolator(X)

            return result


        self.velocity_functions_agg['thermal'] = thermal_profile
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



class ReconstructedAirVelocityField(AirVelocityFieldBase):

    def __init__(self, df_splines: dict,
                 interpolator_dict: Dict, max_extrapolated_distance: float=20, **config_args):
        super().__init__()
        self.wind_spline = None
        self.wind_function = None
        self.current_interpolator_dict = interpolator_dict
        self.valid_region = None
        self.thermal_rotation_function = None
        self.thermal_profile_function = None
        self.components = {'wind', 'thermal'}
        self.df_splines = None
        self.extrapolate = max_extrapolated_distance > 0
        self.max_extrapolated_distance = max_extrapolated_distance
        self.current_thermal_core_spline = {}
        self.current_TC_spline = {}
        self.thermal_profile_function_agg = None
        self.current_spline = df_splines
        self.hull_per_bin = []
        self.boundaries_per_bin = []
        self.boundary_values_per_bin = []
        self.gradients_per_bin = []
        self.preprocessing()
        # self.get_velocity = np.vectorize(self.get_velocity)

    @classmethod
    def from_path(cls, path_to_files: str, iteration: Union[int, str]=None, aggregated=True,
                  max_extrapolated_distance: float =20, **config_args):

        dec = load_decomposition_data(path_to_files, iteration=None,
                                      list_of_files=['splines.yaml',
                                                     ])

        with open(os.path.join(path_to_files, 'interpolators.pkl'), 'rb') as f:
            current_interpolator_dict = pickle.load(f)


        return ReconstructedAirVelocityField(df_splines=dec['splines'][0],
                                             interpolator_dict=current_interpolator_dict,
                                             max_extrapolated_distance=max_extrapolated_distance,
                                             **config_args)

    def preprocessing(self):
        self.preprocess_data()
        self.preprocess_wind()
        self.preprocess_thermal()
        self.set_thermal_core()
        # self.preprocess_turbulence()

        # self.set_thermal_core_advection_time_dependent()

    def preprocess_data(self):
        # READ SPLINES

        self.current_thermal_core_spline = {}
        self.valid_region = self.current_interpolator_dict[f'valid_region']
        self.wind_spline = {}
        for col in ['X', 'Y']:
            self.current_thermal_core_spline[col] = UnivariateSplineWrapper.from_tck(
                self.current_spline['thermal_core_positions'][f'{col}_avg']['tck'])
            self.current_TC_spline[col] = UnivariateSplineWrapper.from_tck(
                self.current_spline['thermal_core_velocities'][f'd{col}dT_TC_ground']['tck'])

            self.wind_spline[col] = UnivariateSplineWrapper.from_tck(self.current_spline['wind'][f'wind_{col}']['tck'])
        self.current_TC_spline['Z'] = UnivariateSplineWrapper.from_tck(
            self.current_spline['thermal_core_velocities'][f'dZdT_TC_ground']['tck'])


    def preprocess_turbulence(self):
        self.velocity_functions['turbulence'] = self.get_velocity_fluctuations
        self.components.add('turbulence')
        self.var_per_component['turbulence'] = 'air'

    def preprocess_wind(self):

        def wind_function(X, t):
            self.wind_index_list = 2
            X = np.array(X)

            input_vars = X[..., self.wind_index_list]

            return_array = np.empty(shape=list(X[..., 0].shape) + [3])

            return_array[..., 0] = self.wind_spline['X'](input_vars, extrapolate='linear' if self.extrapolate else 3)
            return_array[..., 1] = self.wind_spline['Y'](input_vars, extrapolate='linear' if self.extrapolate else 3)
            return_array[..., 2] = 0

            return return_array

        self.velocity_functions['wind'] = wind_function
        self.components.add('wind')
        self.var_per_component['wind'] = 'ground'

        # ====================================      PROFILE         ================================================== #

    def preprocess_thermal(self):

        def thermal_function(X, t=0):
            import numpy as np
            old_shape = X.shape
            in_valid_region_mask = self.is_in_valid_region(X)

            result = np.full(shape=list(X[..., 0].shape) + [3], fill_value=np.nan)
            for i_col, col in enumerate(['X', 'Y', 'Z',]):
                result[in_valid_region_mask, i_col] = self.current_interpolator_dict[f'd{col}dT_thermal_ground'](X[in_valid_region_mask])

            if np.any(~in_valid_region_mask) and self.extrapolate:
                result[~in_valid_region_mask] = self.get_extrapolated_velocities(X=X[~in_valid_region_mask],t=t)
            return result


        self.velocity_functions['thermal'] = thermal_function
        self.components.add('thermal')
        self.var_per_component['thermal'] = 'air'


    def is_in_valid_region(self, X):

        in_valid_region_mask = np.linalg.norm(X[...,:2], axis=-1) <= self.valid_region(X[...,2],
                                                                                       np.arctan2(X[...,1], X[...,0])
                                                                                       )

        return in_valid_region_mask

    def set_thermal_core(self):

        def get_thermal_core_function(X, t=0, d=0):

            return_array = np.stack([self.current_thermal_core_spline['X'](X, nu=d, extrapolate='linear'),
                                     self.current_thermal_core_spline['Y'](X, nu=d, extrapolate='linear')],
                                    axis=-1).astype(np.float32)
            return return_array

        self.thermal_core_function = get_thermal_core_function

    @apply_along_class_method_decorator
    def get_extrapolated_velocities(self, X: np.ndarray,t, n_points=5, dr_norm=0.5, dl=1):
        X_cyl = cart_to_cyl_point(X)

        closest_hull_point_cyl = np.copy(X_cyl)
        closest_hull_point_cyl[0] = self.valid_region(X_cyl[2], X_cyl[1])

        dtheta = np.arctan(dl/closest_hull_point_cyl[0])
        list_of_convex_hull_points_cyl = []
        for i_theta in np.arange(-(n_points - 1) // 2, (n_points - 1) // 2 + 1):
            current_theta = np.mod(closest_hull_point_cyl[1] + i_theta * dtheta + np.pi, 2 * np.pi) - np.pi
            list_of_convex_hull_points_cyl.append([self.valid_region(closest_hull_point_cyl[2], current_theta),
                                                   current_theta,
                                                   closest_hull_point_cyl[2]])
        list_of_convex_hull_points_cyl = np.array(list_of_convex_hull_points_cyl)
        list_of_convex_hull_points = np.apply_along_axis(cyl_to_cart_point, axis=1, arr=list_of_convex_hull_points_cyl)

        delta_r_avg_cyl = X_cyl - closest_hull_point_cyl
        if delta_r_avg_cyl[0] > self.max_extrapolated_distance:
            return np.zeros(3)

        hull_v_avg = np.empty((n_points, 3))
        grad_cyl_avg = np.empty((n_points, 3, 3))
        for i_point, current_hull_point in enumerate(list_of_convex_hull_points):
            # iterate on velocity components
            current_hull_point_cyl = cart_to_cyl_point(current_hull_point)

            current_v = np.empty((3,))
            current_jacobian = np.empty((3, 3))
            for i_velo_comp, velo_coord in enumerate( ['X', 'Y', 'Z']):
                current_v[i_velo_comp] = self.current_interpolator_dict[f'd{velo_coord}dT_thermal_ground'](current_hull_point.reshape(1, -1))[0]
                # iterate on the direction
                for j_coord in range(3):
                    current_delta = np.zeros(3)
                    current_delta[j_coord] = dr_norm
                    current_hull_point_aux = current_hull_point + current_delta
                    current_v_i_aux =  self.current_interpolator_dict[f'd{velo_coord}dT_thermal_ground'](current_hull_point_aux.reshape(1, -1))[0]
                    current_jacobian[i_velo_comp, j_coord] = (current_v[i_velo_comp] - current_v_i_aux) / (0 - dr_norm)

            rho = current_hull_point_cyl[0]
            theta = current_hull_point_cyl[1]
            transformation_matrix = cart_to_cyl_matrix(theta)
            transformation_matrix[1, :] /= rho
            grad_cylindrical = (transformation_matrix @ current_jacobian.T).T
            hull_v_avg[i_point] = current_v
            grad_cyl_avg[i_point] = grad_cylindrical

        hull_v_avg = np.mean(hull_v_avg, axis=0)
        grad_cyl_avg = np.mean(grad_cyl_avg, axis=0)

        max_abs_grad = np.abs((hull_v_avg - 0) / self.max_extrapolated_distance)

        grad_cyl_avg_process = np.copy(grad_cyl_avg)
        grad_cyl_avg_process[:, 0] = - np.sign(hull_v_avg) * np.max(np.stack([np.abs(grad_cyl_avg[:, 0]),
                                                                              max_abs_grad], axis=-1),
                                                                    axis=1)
        delta_r_avg_cyl[1] = 0
        delta_v_processed = (grad_cyl_avg_process @ delta_r_avg_cyl)
        processed_v_at_point = hull_v_avg + delta_v_processed
        v = np.where(np.sign(hull_v_avg) != np.sign(processed_v_at_point), 0, processed_v_at_point)

        return v


class IterativeReconstructedAirVelocityField:

    def __init__(self,  path_to_files, iteration='best',max_extrapolated_distance=20, **config_args):
        self.df = None
        self.current_hull = None
        self.list_of_iterations = None
        self.current_interpolator_dict = None
        self.current_iteration = None
        self.df_splines_all_iterations = None
        self.path_to_files = path_to_files
        self.max_extrapolated_distance = max_extrapolated_distance
        self.config_args = config_args
        dec = load_decomposition_data(path_to_files, iteration=iteration,
                                      list_of_files=['iterations.csv',
                                                     'splines.yaml',
                                                     ])
        self.path_to_convexhull = os.path.join(path_to_files, 'convex_hulls')
        self.path_to_interpolator = os.path.join(path_to_files, 'interpolators')
        self.df_splines = None
        self.max_extrapolated_distance = max_extrapolated_distance
        self.thermal_profile_function_agg = None
        self.df_all_iterations = dec['iterations']
        if not isinstance(dec['splines'], pd.DataFrame):
            self.df_splines_all_iterations = pd.DataFrame.from_records(dec['splines'])
        else:
            self.df_splines_all_iterations = dec['splines']
        self.preprocessing()

    def set_iteration(self, i):
        self.current_iteration = i

        self.preprocess_data()
        return ReconstructedAirVelocityField(df=self.df, df_splines=self.df_splines, convexhull=self.current_hull,
                                             interpolator_dict=self.current_interpolator_dict,
                                             max_extrapolated_distance=self.max_extrapolated_distance,
                                             **self.config_args)

    def preprocessing(self):

        self.df_all_iterations = self.df_all_iterations[self.df_all_iterations['iteration'] != 0].copy()
        self.df_splines_all_iterations = self.df_splines_all_iterations[self.df_splines_all_iterations['iteration'] != 0].copy()
        self.list_of_iterations = self.df_splines_all_iterations['iteration'].unique()
        #self.list_of_iterations = list(filter(lambda elem: elem != 0, self.list_of_iterations))
        self.list_of_iterations = sorted(self.list_of_iterations)
        self.current_iteration = self.list_of_iterations[0]
        if self.current_iteration == 0:
            self.current_iteration = 1

    def preprocess_data(self):
        self.df = self.df_all_iterations[self.df_all_iterations['iteration'] == self.current_iteration].copy()
        self.df_splines = self.df_splines_all_iterations[
            self.df_splines_all_iterations['iteration'] == self.current_iteration].copy()

        with open(self.path_to_interpolator, 'rb') as d:
            self.current_interpolator_dict = pickle.load(d)
        with open(self.path_to_convexhull, 'rb') as d:
            self.current_hull = pickle.load(d)
