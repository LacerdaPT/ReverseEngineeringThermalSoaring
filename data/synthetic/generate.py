import logging
import pprint

import numpy as np
from matplotlib.colors import Normalize

from calc.flight import get_radius_from_bank_angle, get_horizontal_velocity_from_bird_parameters, \
    get_min_sink_rate_from_bank_angle
from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame
from object.flight import BirdPoint, Trajectory


logger = logging.getLogger(__name__)


def get_initial_conditions(bird_parameters, control_parameters, air_field_obj):
    bank_angle_init = np.random.uniform(-control_parameters['general_args']['bank_angle_max'],
                                        control_parameters['general_args']['bank_angle_max'])  # degrees
    bank_angle_init = bank_angle_init * np.pi / 180  # Radians

    radius_init = get_radius_from_bank_angle(bank_angle=bank_angle_init, mass=bird_parameters['mass'],
                                             wing_area=bird_parameters['wing_area'], CL=bird_parameters['CL'])

    if 'Z_init' not in bird_parameters.keys():
        Z_init = np.random.uniform(50, 100)
    else:
        Z_init = bird_parameters['Z_init']

    X_core, Y_core = air_field_obj.get_thermal_core(Z_init)
    distance_from_core = np.abs(np.random.uniform(0, 20))
    angle_from_core = np.random.uniform(0, 2 * np.pi)

    if 'X_init' not in bird_parameters.keys():
        X_init = X_core + distance_from_core * np.cos(angle_from_core)
    else:
        X_init = bird_parameters['X_init']

    if 'Y_init' not in bird_parameters.keys():
        Y_init = Y_core + distance_from_core * np.sin(angle_from_core)
    else:
        Y_init = bird_parameters['Y_init']

    theta_init = angle_from_core + np.pi

    # theta_init = np.random.uniform(0, 2 * np.pi)  # np.pi/4
    Vh_init = get_horizontal_velocity_from_bird_parameters(bank_angle=bank_angle_init,
                                                           mass=bird_parameters['mass'],
                                                           wing_area=bird_parameters['wing_area'],
                                                           CL=bird_parameters['CL'],
                                                           rho=air_field_obj.rho)

    Vbird_x_init = Vh_init * np.cos(theta_init)
    Vbird_y_init = Vh_init * np.sin(theta_init)
    Vbird_z_init = -get_min_sink_rate_from_bank_angle(bank_angle_init,
                                                      mass=bird_parameters['mass'],
                                                      wing_area=bird_parameters['wing_area'],
                                                      CL=bird_parameters['CL'],
                                                      rho=air_field_obj.rho,
                                                      CD=bird_parameters['CD'])

    bird_air_init = BirdPoint(V=[Vbird_x_init, Vbird_y_init, Vbird_z_init],
                              X=[X_init, Y_init, Z_init],
                              A=[0, 0, 0],
                              t=0,
                              bank_angle=bank_angle_init,
                              CL=bird_parameters['CL'],
                              CD=bird_parameters['CD'],
                              mass=bird_parameters['mass'],
                              wing_area=bird_parameters['wing_area'],
                              rho=air_field_obj.rho)

    V_air_init, components_init = air_field_obj.get_velocity(bird_air_init['X'], t=0, return_components=True)
    V_air_wind_init = components_init['wind']
    V_air_rotation_init = components_init['rotation']
    V_air_thermal_init = components_init['thermal']
    # Define Arrays

    bird_ground_init = BirdPoint(V=bird_air_init.V + V_air_init,
                                 X=[X_init, Y_init, Z_init],
                                 A=[0, 0, 0], t=0,
                                 bank_angle=bank_angle_init,
                                 CL=bird_parameters['CL'],
                                 CD=bird_parameters['CD'],
                                 mass=bird_parameters['mass'],
                                 wing_area=bird_parameters['wing_area'],
                                 rho=air_field_obj.rho)

    initial_conditions = {'X': X_init,
                          'Y': Y_init,
                          'Z': Z_init,
                          'Vair_X': bird_air_init['V'][0],
                          'Vair_Y': bird_air_init['V'][1],
                          'Vair_Z': bird_air_init['V'][2],
                          'Vground_X': bird_ground_init['V'][0],
                          'Vground_Y': bird_ground_init['V'][1],
                          'Vground_Z': bird_ground_init['V'][2],
                          'Vairfield_X': V_air_init[0],
                          'Vairfield_Y': V_air_init[1],
                          'Vairfield_Z': V_air_init[2]}

    return bird_air_init, bird_ground_init, (V_air_init, V_air_wind_init,
                                             V_air_rotation_init, V_air_thermal_init), initial_conditions


def get_synthetic_flock(duration, dt, air_velocity_field_obj, list_of_bird_air_init, list_of_control_parameters,
                        debug=False):
    # SET INITIAL CONDITIONS
    air_velocity_field_obj.reset_turbulence_function(t=0)
    # INITIALIZE VARIABLES
    n_birds = len(list_of_bird_air_init)
    list_of_trajectory_air = []
    list_of_trajectory_ground = []
    list_of_thermal_core_array = []
    list_of_V_air_array = []
    list_of_V_air_components_arrays = []
    list_of_df = []
    list_of_is_landed = [False] * n_birds
    for bird_air_init, control_parameters in zip(list_of_bird_air_init, list_of_control_parameters):
        V_air_init, components_init = air_velocity_field_obj.get_velocity(bird_air_init['X'], t=0, return_components=True)

        bird_ground_init = BirdPoint(V=bird_air_init.V + V_air_init,
                                     X=bird_air_init.X,
                                     A=[0, 0, 0], t=0,
                                     bank_angle=bird_air_init['bank_angle'],
                                     CL=bird_air_init['CL'],
                                     CD=bird_air_init['CD'],
                                     mass=bird_air_init['mass'],
                                     wing_area=bird_air_init['wing_area'],
                                     rho=air_velocity_field_obj.rho)

        trajectory_air = Trajectory(bird_air_init, control_parameters)
        trajectory_ground = Trajectory(bird_ground_init, control_parameters)
        thermal_core_position_init = air_velocity_field_obj.get_thermal_core(z=trajectory_ground.get_last_N_point()['X'][-1],
                                                                             t=0)
        thermal_core_array = np.array([thermal_core_position_init])
        V_air_array = np.array([V_air_init])
        V_air_components_arrays = {comp: np.array([v_comp]) for comp, v_comp in components_init.items()}

        list_of_trajectory_air.append(trajectory_air)
        list_of_trajectory_ground.append(trajectory_ground)
        list_of_thermal_core_array.append(thermal_core_array)
        list_of_V_air_array.append(V_air_array)
        list_of_V_air_components_arrays.append(V_air_components_arrays)

    time_array = np.arange(0, duration, dt)
    # START SIMULATION
    for i, t in enumerate(time_array[1:]):

        for i_bird in range(len(list_of_bird_air_init)):
            if list_of_is_landed[i_bird]:
                continue
            if np.isclose(t - round(t), 0):
                logger.info(f'{t=} - bird={i_bird}')
            else:
                logger.debug(f'{t=} - bird={i_bird}')
            trajectory_air = list_of_trajectory_air[i_bird]
            trajectory_ground = list_of_trajectory_ground[i_bird]
            thermal_core_array = list_of_thermal_core_array[i_bird]
            V_air_array = list_of_V_air_array[i_bird]
            V_air_components_arrays = list_of_V_air_components_arrays[i_bird]

            bank_angle_new, flight_mode_new, thermalling_mode_new = trajectory_ground.get_control(dt=dt)
            logger.debug(f'{bank_angle_new=}')
            V_air_new, components_new = air_velocity_field_obj.get_velocity(X=trajectory_ground.get_last_N_point()['X'],
                                                                            t=t, return_components=True)
            thermal_core_position_new = air_velocity_field_obj.get_thermal_core(z=trajectory_ground.get_last_N_point()['X'][-1],
                                                                                t=t)

            for line in pprint.pformat(components_new).split('\n'):
                logger.log(level=logging.DEBUG, msg=line)

            trajectory_air.insert_by_bank_angle(bank_angle=bank_angle_new, dt=dt, flight_mode=flight_mode_new,
                                                thermalling_mode=thermalling_mode_new)
            trajectory_ground.insert_by_velocity(trajectory_air.get_last_N_point()['V'] + V_air_new,
                                                 dt=dt, flight_mode=flight_mode_new, thermalling_mode=thermalling_mode_new)
            trajectory_ground.get_last_N_point().bank_angle = bank_angle_new

            list_of_V_air_array[i_bird] = np.vstack([V_air_array, V_air_new])
            for comp, v_component in components_new.items():
                list_of_V_air_components_arrays[i_bird][comp] = np.vstack([V_air_components_arrays[comp], v_component])

            list_of_thermal_core_array[i_bird] = np.vstack([thermal_core_array, thermal_core_position_new])
            # Check if the bird is landed
            if ((trajectory_ground.get_last_N_point()['X'][2] < 0)
                    or (np.isclose(trajectory_ground.get_last_N_point()['X'][2], 0, atol=0.1))):
                logger.info(f'landed at {t}')
                trajectory_ground.get_last_N_point()['X'][2] = 0
                # trajectory_ground.land(dt)
                # trajectory_air.land(dt)
                list_of_is_landed[i_bird] = True

    # ==================================================================================================================
    #                                      COMPILE AND POST-CALCULATION DATA
    # ==================================================================================================================

    for i_bird in range(len(list_of_bird_air_init)):
        trajectory_air = list_of_trajectory_air[i_bird]
        trajectory_ground = list_of_trajectory_ground[i_bird]
        thermal_core_array = list_of_thermal_core_array[i_bird]
        V_air_array = list_of_V_air_array[i_bird]
        V_air_components_arrays = list_of_V_air_components_arrays[i_bird]

        df_ground = trajectory_ground.to_dataframe()
        df_air = trajectory_air.to_dataframe('bird_real', include_thermal_core_estimate=False)

        df = df_ground.join(df_air.drop(columns='time'))

        df['thermal_core_X_real'] = thermal_core_array[:, 0]
        df['thermal_core_Y_real'] = thermal_core_array[:, 1]
        for i_coord, coord in enumerate(['X', 'Y', 'Z']):
            df[f'd{coord}dT_air_real'] = V_air_array[:, i_coord]
            for comp, v_comp_array in V_air_components_arrays.items():

                df[f'd{coord}dT_air_{comp}_real'] = v_comp_array[:, i_coord]

        df['curvature_bird_real'] = 1 / df['radius_bird_real']
        df['curvature'] = 1 / df['radius']
        df['X_thermal_real'] = df['X'] - df['thermal_core_X_real']
        df['Y_thermal_real'] = df['Y'] - df['thermal_core_Y_real']
        df['rho_thermal_real'] = np.linalg.norm(df[['X_thermal_real', 'Y_thermal_real']], axis=1)
        df['phi_thermal_real'] = np.arctan2(df['Y_thermal_real'], df['X_thermal_real'])

        for velocity_type in ['air_rotation', 'air_thermal', 'bird']:
            df[[f'V_rho_rotating_{velocity_type}_real',
                f'V_phi_rotating_{velocity_type}_real']] = df[['X_thermal_real', 'Y_thermal_real',
                                                               f'dXdT_{velocity_type}_real',
                                                               f'dYdT_{velocity_type}_real'
                                                               ]].apply(lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
                                                                        axis=1, result_type='expand')
            df[f'V_H_{velocity_type}_real'] = np.linalg.norm(df[[f'dXdT_{velocity_type}_real',
                                                                 f'dYdT_{velocity_type}_real']], axis=1)

        for velocity_type in ['air', 'air_wind']:

            df[[f'V_rho_{velocity_type}_real',
                f'V_phi_{velocity_type}_real']] = df[['X', 'Y',
                                                      f'dXdT_{velocity_type}_real',
                                                      f'dYdT_{velocity_type}_real'
                                                      ]].apply(lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
                                                               axis=1, result_type='expand')
            df[f'V_H_{velocity_type}_real'] = np.linalg.norm(df[[f'dXdT_{velocity_type}_real',
                                                                 f'dYdT_{velocity_type}_real']], axis=1)
        df[f'V_H'] = np.linalg.norm(df[['dXdT', 'dYdT']], axis=1)

        list_of_df.append(df)

    return list_of_df, list_of_is_landed


def get_synthetic_bird(duration, dt, air_velocity_field_obj, bird_air_init, control_parameters, debug=False):
    # SET INITIAL CONDITIONS
    air_velocity_field_obj.reset_turbulence_function(t=0)
    # INITIALIZE VARIABLES

    V_air_init, components_init = air_velocity_field_obj.get_velocity(bird_air_init['X'], t=0, return_components=True)

    bird_ground_init = BirdPoint(V=bird_air_init.V + V_air_init,
                                 X=bird_air_init.X,
                                 A=[0, 0, 0], t=0,
                                 bank_angle=bird_air_init['bank_angle'],
                                 CL=bird_air_init['CL'],
                                 CD=bird_air_init['CD'],
                                 mass=bird_air_init['mass'],
                                 wing_area=bird_air_init['wing_area'],
                                 rho=air_velocity_field_obj.rho)

    trajectory_air = Trajectory(bird_air_init, control_parameters)
    trajectory_ground = Trajectory(bird_ground_init, control_parameters)
    thermal_core_position_init = air_velocity_field_obj.get_thermal_core(z=trajectory_ground.get_last_N_point()['X'][-1],
                                                                         t=0)
    thermal_core_array = np.array([thermal_core_position_init])
    V_air_array = np.array([V_air_init])
    V_air_components_arrays = {comp: np.array([v_comp]) for comp, v_comp in components_init.items()}

    time_array = np.arange(0, duration, dt)

    is_landed = False
    for i, t in enumerate(time_array[1:]):
        if np.isclose(t - round(t), 0):
            logger.info(f'{t=}')
        else:
            logger.debug(f'{t=}')

        bank_angle_new, flight_mode_new, thermalling_mode_new = trajectory_ground.get_control(dt=dt)
        logger.debug(f'{bank_angle_new=}')
        V_air_new, components_new = air_velocity_field_obj.get_velocity(X=trajectory_ground.get_last_N_point()['X'],
                                                                        t=t, return_components=True)
        thermal_core_position_new = air_velocity_field_obj.get_thermal_core(z=trajectory_ground.get_last_N_point()['X'][-1],
                                                                            t=t)

        for line in pprint.pformat(components_new).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)

        trajectory_air.insert_by_bank_angle(bank_angle=bank_angle_new, dt=dt, flight_mode=flight_mode_new,
                                            thermalling_mode=thermalling_mode_new)
        trajectory_ground.insert_by_velocity(trajectory_air.get_last_N_point()['V'] + V_air_new,
                                             dt=dt, flight_mode=flight_mode_new, thermalling_mode=thermalling_mode_new)
        trajectory_ground.get_last_N_point().bank_angle = bank_angle_new
        trajectory_ground.set_thermal_core_estimation(1, 50)

        V_air_wind_new = components_new['wind']
        V_air_rotation_new = components_new['rotation']
        V_air_thermal_new = components_new['thermal']
        V_air_turbulence_new = components_new['turbulence']

        V_air_array = np.vstack([V_air_array, V_air_new])
        for comp, v_component in components_new.items():
            V_air_components_arrays[comp] = np.vstack([V_air_components_arrays[comp], v_component])

        thermal_core_array = np.vstack([thermal_core_array, thermal_core_position_new])
        # Check if the bird is landed
        if ((trajectory_ground.get_last_N_point()['X'][2] < 0)
                or (np.isclose(trajectory_ground.get_last_N_point()['X'][2], 0, atol=0.1))):
            logger.info(f'landed at {t}')
            trajectory_ground.get_last_N_point()['X'][2] = 0
            # trajectory_ground.land(dt)
            # trajectory_air.land(dt)
            is_landed = True
            break

        if control_parameters['general_args']['debug'] and np.isclose(t % control_parameters['general_args']['period'],
                                                                      0) and False:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            aaa = ax.scatter(trajectory_ground['X'][:, 0], trajectory_ground['X'][:, 1],
                             c=trajectory_ground['bank_angle'],
                             norm=Normalize(vmin=-0.5, vmax=0.5),
                             cmap='Spectral_r', label='track')
            ax.scatter([0], [0], marker='o', label='real core')
            ax.set_aspect('equal')
            plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
            plt.colorbar(aaa)
            plt.show(block=True)
    # ==================================================================================================================
    #                                               STORE DATA
    # ==================================================================================================================

    df_ground = trajectory_ground.to_dataframe()
    df_air = trajectory_air.to_dataframe('bird_real', include_thermal_core_estimate=False)

    df = df_ground.join(df_air.drop(columns='time'))

    df['thermal_core_X_real'] = thermal_core_array[:, 0]
    df['thermal_core_Y_real'] = thermal_core_array[:, 1]
    for i_coord, coord in enumerate(['X', 'Y', 'Z']):
        df[f'd{coord}dT_air_real'] = V_air_array[:, i_coord]
        for comp, v_comp_array in V_air_components_arrays.items():

            df[f'd{coord}dT_air_{comp}_real'] = v_comp_array[:, i_coord]

    df['curvature_bird_real'] = 1 / df['radius_bird_real']
    df['curvature'] = 1 / df['radius']
    df['X_thermal_real'] = df['X'] - df['thermal_core_X_real']
    df['Y_thermal_real'] = df['Y'] - df['thermal_core_Y_real']
    df['rho_thermal_real'] = np.linalg.norm(df[['X_thermal_real', 'Y_thermal_real']], axis=1)
    df['phi_thermal_real'] = np.arctan2(df['Y_thermal_real'], df['X_thermal_real'])

    for velocity_type in ['air_rotation', 'air_thermal', 'bird']:
        df[[f'V_rho_rotating_{velocity_type}_real',
            f'V_phi_rotating_{velocity_type}_real']] = df[['X_thermal_real', 'Y_thermal_real',
                                                           f'dXdT_{velocity_type}_real',
                                                           f'dYdT_{velocity_type}_real'
                                                           ]].apply(lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
                                                                    axis=1, result_type='expand')
        df[f'V_H_{velocity_type}_real'] = np.linalg.norm(df[[f'dXdT_{velocity_type}_real',
                                                             f'dYdT_{velocity_type}_real']], axis=1)

    for velocity_type in ['air', 'air_wind']:

        df[[f'V_rho_{velocity_type}_real',
            f'V_phi_{velocity_type}_real']] = df[['X', 'Y',
                                                  f'dXdT_{velocity_type}_real',
                                                  f'dYdT_{velocity_type}_real'
                                                  ]].apply(lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
                                                           axis=1, result_type='expand')
        df[f'V_H_{velocity_type}_real'] = np.linalg.norm(df[[f'dXdT_{velocity_type}_real',
                                                             f'dYdT_{velocity_type}_real']], axis=1)
    df[f'V_H'] = np.linalg.norm(df[['dXdT', 'dYdT']], axis=1)
    return df, is_landed
