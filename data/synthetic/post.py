import pandas as pd


def postprocess_generated_datasets(df_real, df_data):
    df_real['time'] = round(df_real['time'], 3)
    df_data['time'] = round(df_data['time'], 3)
    for col in ['dXdT_air_turbulence_real', 'dYdT_air_turbulence_real', 'dZdT_air_turbulence_real']:
        if col not in df_real.columns:
            df_real[col] = 0.0
    df_data_full = pd.merge(df_real, df_data[['bird_name', 'time']],
                            on=['bird_name', 'time'])

    df_data_full['dXdT_air_thermal_real'] = df_data_full['dXdT_air_rotation_real'].values
    df_data_full['dYdT_air_thermal_real'] = df_data_full['dYdT_air_rotation_real'].values

    df_data_full.drop(columns=['dXdT_air_rotation_real',
                               'dYdT_air_rotation_real',
                               'dZdT_air_rotation_real'],
                      inplace=True)
    df_data_full.rename(columns={'thermal_core_X_real': 'X_TC_ground',
                                 'thermal_core_Y_real': 'Y_TC_ground',
                                 'X_thermal_real':      'X_bird_TC',
                                 'Y_thermal_real':      'Y_bird_TC',
                                 'rho_thermal_real':    'rho_bird_TC',
                                 'phi_thermal_real':    'phi_bird_TC', },
                        inplace=True)

    bird_dict = {'X_bird_real':                'X_bird_air',
                 'Y_bird_real':                'Y_bird_air',
                 'Z_bird_real':                'Z_bird_air',
                 'dXdT_bird_real':             'dXdT_bird_air',
                 'dYdT_bird_real':             'dYdT_bird_air',
                 'dZdT_bird_real':             'dZdT_bird_air',
                 'd2XdT2_bird_real':           'd2XdT2_bird_air',
                 'd2YdT2_bird_real':           'd2YdT2_bird_air',
                 'd2ZdT2_bird_real':           'd2ZdT2_bird_air',
                 'dXdT':                       'dXdT_bird_ground',
                 'dYdT':                       'dYdT_bird_ground',
                 'dZdT':                       'dZdT_bird_ground',
                 'V_H':                        'V_H_bird_ground',
                 'd2XdT2':                     'd2XdT2_bird_ground',
                 'd2YdT2':                     'd2YdT2_bird_ground',
                 'd2ZdT2':                     'd2ZdT2_bird_ground',
                 'bearing_bird_real':          'bearing_bird_air',
                 'radius_bird_real':           'radius_bird_air',
                 'bank_angle_bird_real':       'bank_angle_bird_air',
                 'curvature_bird_real':        'curvature_bird_air',
                 'curvature':                  'curvature_bird_ground',
                 'bearing':                    'bearing_bird_ground',
                 'radius':                     'radius_bird_ground',
                 'bank_angle':                 'bank_angle_bird_ground',
                 'flight_mode':                'flight_mode_bird_ground',
                 'thermalling_mode':           'thermalling_mode_bird_ground',
                 'V_rho_rotating_bird_real':   'V_rho_rotating_bird_air',
                 'V_phi_rotating_bird_real':   'V_phi_rotating_bird_air',
                 'V_H_bird_real':              'V_H_bird_air',
                 'flight_mode_bird_real':      'flight_mode_bird_air',
                 'thermalling_mode_bird_real': 'thermalling_mode_bird_air',
                 }

    air_dict = {'dXdT_air_real':                    'dXdT_air_ground',
                'dXdT_air_turbulence_real':         'dXdT_turbulence_ground',
                'dXdT_air_wind_real':               'wind_X',
                'dXdT_air_thermal_real':            'dXdT_thermal_ground',
                'dYdT_air_real':                    'dYdT_air_ground',
                'dYdT_air_turbulence_real':         'dYdT_turbulence_ground',
                'dYdT_air_wind_real':               'wind_Y',
                'dYdT_air_thermal_real':            'dYdT_thermal_ground',
                'dZdT_air_real':                    'dZdT_air_ground',
                'dZdT_air_turbulence_real':         'dZdT_turbulence_ground',
                'dZdT_air_wind_real':               'wind_Z',
                'dZdT_air_thermal_real':            'dZdT_thermal_ground',
                'V_rho_air_real':                   'V_rho_air_ground',
                'V_phi_air_real':                   'V_phi_air_ground',
                'V_H_air_real':                     'V_H_air_ground',
                'V_rho_air_wind_real':              'V_rho_wind_real',
                'V_phi_air_wind_real':              'V_phi_wind_real',
                'V_H_air_wind_real':                'V_H_wind_real',
                'V_rho_rotating_air_rotation_real': 'V_rho_rotating_air_rotation_ground',
                'V_phi_rotating_air_rotation_real': 'V_phi_rotating_air_rotation_ground',
                'V_H_air_rotation_real':            'V_H_rotation_ground',
                'V_rho_rotating_air_thermal_real':  'V_rho_rotating_air_thermal_ground',
                'V_phi_rotating_air_thermal_real':  'V_phi_rotating_air_thermal_ground',
                'V_H_air_thermal_real':             'V_H_thermal_ground',
                'curvature_bird_real':              'curvature_bird_air',
                'curvature':                        'curvature_bird_ground',
                }

    common_columns = ['bird_name', 'time', 'X', 'Y', 'Z',
                      'X_TC_ground', 'Y_TC_ground',
                      'X_bird_TC', 'Y_bird_TC',
                      'rho_bird_TC', 'phi_bird_TC',
                      ]
    df_air = df_data_full[common_columns + list(air_dict.keys())].copy()
    df_bird = df_data_full[common_columns + list(bird_dict.keys())].copy()

    df_data_full.rename(columns=air_dict | bird_dict, inplace=True)
    df_air.rename(columns=air_dict, inplace=True)
    df_bird.rename(columns=bird_dict, inplace=True)

    return df_data_full, df_air, df_bird
