from typing import Iterable

import numpy as np
import pandas as pd
import os
import dill as pickle
import yaml

from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame

import shelve
from dill import Pickler, Unpickler

shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

def get_single_bird_burst(path_to_file, sanitize=True):
    df = pd.read_csv(path_to_file, sep='\t', header=1)
    df.reset_index(inplace=True)
    new_columns = df.columns.values[1:].tolist() + ['bird_name']
    df.columns = new_columns

    # column names sanitizing
    if sanitize:
        column_names = df.columns
        column_names = [
            col.replace('[', '(').replace(']', ')').replace('(m/s)', '').replace('(m/s2)', '').replace('(m)', '')
            for col in column_names]

        df.columns = column_names
        df.rename(columns={'#t(centisec)': 'time'}, inplace=True)

    return df


def get_burst(root_path, sanitize=True):
    df = pd.DataFrame()

    for file in os.listdir(root_path):
        current_burst = get_single_bird_burst(os.path.join(root_path, file), sanitize=False)
        if df.empty:
            df = current_burst.copy()
        else:
            df = pd.concat([df, current_burst])

    # column names sanitizing
    column_names = df.columns
    column_names = [
        col.replace('[', '(').replace(']', ')').replace('(m/s)', '').replace('(m/s2)', '').replace('(m)', '')
        for col in column_names]

    df.columns = column_names
    df.rename(columns={'#t(centisec)': 'time'}, inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'bird_time_index'}, inplace=True)
    return df


def load_decomposition_data(path, list_of_files=None, iteration=None):
    if iteration == 'best':
        df_loss = pd.read_csv(os.path.join(path, 'losses.csv'), index_col=0, low_memory=False)
        iteration = df_loss.loc[df_loss['loss'].argmin(), 'iteration']
    if iteration is not None:
        if np.isscalar(iteration):
            iteration = [iteration]
    if list_of_files is None:
        list_of_files = ['iterations.csv',
                         'bins.csv',
                         'thermal_core.csv',
                         'vz_max.csv',
                         'aerodynamic_parameters.csv',
                         'splines.yaml',
                         'aggregated.csv',
                         'parameters.yml',
                         #'convex_hulls_reconstructed.db'
                         #'interpolators_reconstructed.db'
                         'losses.csv']

    df_dict = {}
    for key in list_of_files:
        file_name, file_extension = key.split('.')
        try:
            if file_extension == 'csv':

                df_dict[file_name] = pd.read_csv(os.path.join(path, f'{key}'), index_col=0, low_memory=False)
            elif (file_extension == 'yaml') or (file_extension == 'yml'):
                with open(os.path.join(path, f'{key}'), 'r') as f:
                    df_dict[file_name] = yaml.load(f, yaml.FullLoader)
            else:
                with open(os.path.join(path, f'{key}'), 'rb') as f:
                    df_dict[file_name] = pickle.load(f)


            if isinstance(df_dict[file_name], dict):
                df_dict[file_name] = pd.DataFrame.from_dict([df_dict[file_name]], orient='columns')
            # elif isinstance(df_dict[file_name], list):
            #     df_dict[file_name] = pd.DataFrame.from_dict(df_dict[file_name], orient='columns')
            if iteration is not None:
                try:
                    if isinstance(df_dict[file_name], list):
                        df_dict[file_name] = list(filter(lambda elem: elem['iteration'] in iteration, df_dict[file_name]))
                    else:
                        df_dict[file_name] = df_dict[file_name][df_dict[file_name]['iteration'].isin(iteration)]
                except KeyError:
                    pass
        except FileNotFoundError:
            pass
    try:
        if os.path.exists(os.path.join(path, 'decomposition_args.yml')):
            with open(os.path.join(path, 'decomposition_args.yml'), 'r') as f:
                df_dict['decomposition_args'] = yaml.load(f, yaml.FullLoader)
        else:
            with open(os.path.join(path, 'decomposition_args.yaml'), 'r') as f:
                df_dict['decomposition_args'] = yaml.load(f, yaml.FullLoader)
    except FileNotFoundError:
        pass

    return df_dict


def load_synthetic_data(path, list_of_object=None):

    if list_of_object is None:
        list_of_object = ['air_parameters.pkl',
                          'air_velocity_field.pkl',
                          'data.csv',
                          'data_real.csv',
                          'synthetic_run_params.pkl',
                          'bird_parameters.yml',
                          'bird_parameters.pkl',
                          'parameters.yml'
                          ]
    return_dict = {}
    for file in list_of_object:
        if not os.path.exists(os.path.join(path, file)):
            continue
        filetype = file.split('.')[-1]
        file_name = '.'.join(file.split('.')[:-1])
        if filetype == 'pkl':
            with open(os.path.join(path, file), 'rb') as f:
                return_dict[file_name] = pickle.load(f)
        elif filetype == 'csv':
            return_dict[file_name] = pd.read_csv(os.path.join(path, file), low_memory=False)
        else:
            try:
                with open(os.path.join(path, file), 'r') as f:
                    return_dict[file_name] = yaml.load(f, Loader=yaml.FullLoader)
            except:
                continue

    return return_dict


def load_synthetic_and_decomposed(path_to_decomposition, list_of_files=None, input_folder=None, iteration=None):

    decomposition_dict = load_decomposition_data(path_to_decomposition, list_of_files=list_of_files,
                                                 iteration=iteration)

    if input_folder is None:
        decomposition_args = decomposition_dict['decomposition_args']
        path_to_synthetic_data = decomposition_args['run_parameters']['input_folder']
    else:
        path_to_synthetic_data = input_folder

    synthetic_data_dict = load_synthetic_data(path_to_synthetic_data, list_of_object=list_of_files)

    return synthetic_data_dict, decomposition_dict


def get_standardized_decomposed_and_real_data(df_iteration, df_real, AirVelocityField):

    df_iteration['d2ZdT2_bird'] = df_iteration['d2ZdT2_bird_3']


    return df_iteration, df_real
