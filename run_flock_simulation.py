import datetime
import logging
import os
import pprint
import sys
import time
from shutil import copyfile, SameFileError

import dill as pickle
import numpy as np
import pandas as pd

from data.auxiliar import downsample_dataframe
from data.synthetic.generate import get_synthetic_flock
from data.synthetic.auxiliar import get_run_parameters_from_yaml,  get_bird_parameters_from_yaml, get_air_velocity
from data.synthetic.post import postprocess_generated_datasets
from misc.auxiliar import config_logger, flatten_dict, sanitize_dict_for_yaml
from object.air import AirVelocityFieldVisualization
from object.flight import BirdPoint
from plotting.plot import inspect_flock


def main(path_to_yaml):
    with open(path_to_yaml, 'r') as file:
        import yaml
        parameters_dict = yaml.load(file, Loader=yaml.FullLoader)

    run_parameters = get_run_parameters_from_yaml(parameters_dict['run'])
    air_velocity_field_obj, air_parameters = get_air_velocity(parameters_dict['air'],
                                                              duration=run_parameters['duration'])

    (list_of_bird_names,
     list_of_init_condition,
     list_of_physical_parameters,
     list_of_control_parameters) = get_bird_parameters_from_yaml(parameters_dict['bird'],
                                                                 air_velocity_field_obj=air_velocity_field_obj,
                                                                 n_birds=run_parameters['n_birds'])

    run_time = datetime.datetime.isoformat(datetime.datetime.now()).split('.')[0]
    if run_parameters['save_folder']:
        destination_folder = os.path.join(run_parameters['save_folder'], f'{run_parameters["prefix"]}')
        try:
            os.makedirs(destination_folder)
        except FileExistsError as e:
            destination_folder = os.path.join(run_parameters['save_folder'], f'{run_parameters["prefix"]}_{run_time}')
            os.makedirs(destination_folder)
    else:
        destination_folder = ''

    #verbosity_level = logging.WARNING
    logger = logging.getLogger()

    config_logger(logger, output_dir=destination_folder,
                  verbosity=run_parameters['verbosity'], log_to_file=bool(run_parameters['save_folder']))

    if run_parameters['inspect_before_running']:
        import matplotlib
        matplotlib.use('QtAgg')
        import matplotlib.pyplot as plt
        air_velocity_field_vis = AirVelocityFieldVisualization(air_velocity_field_obj)
        air_velocity_field_vis.plot_all()
        plt.show(block=True)
    start_time = time.time()
    synthetic_bird_parameters = []
    df_all_birds = pd.DataFrame()


    list_of_bird_air_init = [BirdPoint.from_bank_angle_and_bearing(bank_angle=init_condition['bank_angle'],
                                                                   bearing=init_condition['bearing'],
                                                                   X=[init_condition['x'],
                                                                      init_condition['y'],
                                                                      init_condition['z'],
                                                                      ],
                                                                   A=[0, 0, 0], t=0,
                                                                   CL=physical_parameters['CL'],
                                                                   CD=physical_parameters['CD'],
                                                                   mass=physical_parameters['mass'],
                                                                   wing_area=physical_parameters['wing_area'])

                             for init_condition, physical_parameters in
                             zip(list_of_init_condition, list_of_physical_parameters)]
    (list_of_df, list_of_is_landed) = get_synthetic_flock(duration=run_parameters['duration'],
                                                          dt=run_parameters['dt'],
                                                          air_velocity_field_obj=air_velocity_field_obj,
                                                          list_of_bird_air_init=list_of_bird_air_init,
                                                          list_of_control_parameters=list_of_control_parameters,
                                                          debug=False
                                                          )

    for i in range(run_parameters['n_birds']):
        current_bird = list_of_df[i]
        bird_name = list_of_bird_names[i]
        init_condition = list_of_init_condition[i]
        physical_parameters = list_of_physical_parameters[i]
        control_parameters = list_of_control_parameters[i]
        current_bird['bird_name'] = bird_name
        current_bird['bird_name'] = current_bird['bird_name'].astype('string')
        synthetic_bird_parameters.append({'bird_name': bird_name,
                                          'initial_conditions': init_condition,
                                          'physical_parameters': physical_parameters,
                                          'control_parameters': control_parameters})
        if df_all_birds.empty:
            df_all_birds = current_bird.copy()
        else:
            df_all_birds = pd.concat([df_all_birds, current_bird])

    # =======================================            SAVING             ======================================= #

    all_birds = list(map(lambda elem: elem['bird_name'], synthetic_bird_parameters))
    birds_to_keep = all_birds
    if run_parameters['inspect_before_saving']:
        _, _, _, _, exclude_list = inspect_flock(df_all_birds, X_col='X', Y_col='Y', Z_col='Z', time_col='time',
                                                 bird_label_col='bird_name', color=None,
                                                 air_velocity_field=air_velocity_field_obj)
        birds_to_keep = [bird for bird in all_birds if bird not in exclude_list]

    if run_parameters['skip_landed']:
        landed_bird_names = [all_birds[i] for i, landed in enumerate(list_of_is_landed) if landed]
        birds_to_keep = [bird for bird in birds_to_keep if bird not in landed_bird_names]
    synthetic_bird_parameters = list(filter(lambda x: x['bird_name'] in birds_to_keep, synthetic_bird_parameters))
    df_all_birds = df_all_birds[df_all_birds['bird_name'].isin(birds_to_keep)]

    df_output = df_all_birds[['bird_name', 'time', 'X', 'Y', 'Z']].copy()
    df_output = downsample_dataframe(df_output, round(run_parameters['dt_to_save'] / run_parameters['dt']),
                                     partition_key='bird_name')
    for coord in ['X', 'Y', 'Z']:
        df_output[coord] = df_output[coord] \
                           + run_parameters['noise_level'] * np.random.standard_normal(df_output[coord].count())

    if run_parameters['save_folder']:
        try:
            copyfile(sys.argv[1], os.path.join(destination_folder, 'parameters.yml'))
        except SameFileError:
            pass

        with open(os.path.join(destination_folder, 'air_velocity_field.pkl'), 'wb') as f:
            pickle.dump(air_velocity_field_obj, f)
        with open(os.path.join(destination_folder, 'air_parameters.pkl'), 'wb') as f:
            pickle.dump(air_parameters, f)

        with open(os.path.join(destination_folder, 'bird_parameters.yml'), 'w') as f:
            import yaml
            yaml.dump(list(map(sanitize_dict_for_yaml, synthetic_bird_parameters)), f, default_flow_style=False)

        synthetic_bird_parameters = [flatten_dict(d) for d in synthetic_bird_parameters]
        df_synthetic_bird_parameters = pd.DataFrame(synthetic_bird_parameters)
        df_synthetic_bird_parameters.to_csv(os.path.join(destination_folder, 'bird_parameters.csv'),
                                            index=False, sep=',')

        df_all_birds.to_csv(os.path.join(destination_folder, 'data_real.csv'), index=False, sep=',')
        df_output.to_csv(os.path.join(destination_folder, 'data.csv'), index=False, sep=',')

        df_data_full, df_air, df_bird = postprocess_generated_datasets(df_all_birds, df_output)

        df_data_full.to_csv(os.path.join(destination_folder, 'data_full.csv'), index=False)
        df_air.to_csv(os.path.join(destination_folder, 'air.csv'), index = False)
        df_bird.to_csv(os.path.join(destination_folder, 'bird.csv'), index = False)
        logger.info(f'saved to {destination_folder}')

    run_duration = time.time() - start_time
    logger.info(f'this took {round(run_duration/60.,1)}')


if __name__ == '__main__':
    main(sys.argv[1])
