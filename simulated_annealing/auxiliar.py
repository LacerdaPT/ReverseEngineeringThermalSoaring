import argparse
import datetime
import logging
import os
import pprint
from copy import deepcopy

import yaml

from data.auxiliar import get_args_from_yaml_with_default
from decomposition.auxiliar import decomposition_preparation
from misc.auxiliar import config_logger, deep_dictionary_update

logger = logging.getLogger(__name__)


default_sim_annealing_parameters = {'run_parameters': {'n_iterations': 15,
                                                       'verbosity': 2,
                                                       'save': True,
                                                       'output_folder': None,
                                                       'save_all_iterations': True,
                                                       'n_processors': None,
                                                       'debug': False},
                                    'annealing_parameters': {'max_iter': 20,
                                                             'temperature': 10,
                                                             'max_fiter': 10000,
                                                             'annealing_type': 'cauchy',
                                                             'minimizer_kwargs': None,
                                                             'x0': None,
                                                             },
                                    'search_parameters': {'parameters_to_search': ['mass'],
                                                          'individual_search': False,
                                                          'search_limits': {'mass': (2, 4),
                                                                            'CD': (0.01, 1),
                                                                            'CL': (0.8, 3),
                                                                            'wing_area': (0.1, 2)
                                                                            },
                                                          'loss_function': 'horizontal_air_velocity_median'
                                                          },
                                    'decomposition_parameters':{
                                        'file': 'config/default/decomposition_parameters.default.yaml'
                                    }
                                    }


def parse_sim_annealing_arguments(parse_args):
    args = {'yaml': parse_args['yaml'],
            'run_parameters': {'verbosity': parse_args['verbosity'],
                               'save': parse_args['save'],
                               'output_folder': parse_args['output_folder'],
                               'true_values_cols': parse_args['true_columns'],
                               'n_processors': None,
                               'debug': parse_args['debug']},
            'annealing_parameters': {'max_iter': parse_args['n_iterations'],
                                     'temperature': None,
                                     'no_local_search': None,
                                     'annealing_type': None,
                                     'n_consecutive_exception_max': None,
                                     'max_fiter': None,
                                     'minimizer_kwargs': None,
                                     'x0': parse_args['initial_guess']},
            'search_parameters': {'parameters_to_search': parse_args['parameters_to_search'],
                                  'search_limits': {'mass': parse_args['CL'],
                                                    'CD': parse_args['CD'],
                                                    'CL': parse_args['mass'],
                                                    'wing_area': parse_args['wing_area']
                                                    },
                                  },
            'decomposition_parameters': {
                'file': parse_args['decomposition_yaml']
            }
            }

    return args


def get_sim_annealing_args_from_yaml(path_to_yaml):
    with open(path_to_yaml, 'r') as f:
        yaml_dict = yaml.load(f, yaml.FullLoader)
    parameter_dict = default_sim_annealing_parameters.copy()

    for parameter_set in default_sim_annealing_parameters.keys():
        if parameter_set in yaml_dict:
            if yaml_dict[parameter_set] is not None:
                parameter_dict[parameter_set].update(yaml_dict[parameter_set])
            else:
                parameter_dict[parameter_set] = yaml_dict[parameter_set]
    return parameter_dict


def get_sim_annealing_args(cmd_line_args):
    # Read from yaml
    if cmd_line_args['yaml']:
        parameter_dict = get_args_from_yaml_with_default(cmd_line_args['yaml'], default_sim_annealing_parameters)
    else:
        parameter_dict = default_sim_annealing_parameters.copy()

    # Read from Command line - OVERWRITES YAML!!

    deep_dictionary_update(parameter_dict, cmd_line_args, condition_for_update=lambda val: val is not None)

    if isinstance(parameter_dict['run_parameters']['verbosity'], int):
        parameter_dict['run_parameters']['verbosity'] = parameter_dict['run_parameters']['verbosity'] * 10
    parameter_dict['run_parameters']['run_time'] = datetime.datetime.isoformat(datetime.datetime.now()).split(".")[0]

    return parameter_dict


def sim_annealing_preparation(parse_args):

    sim_annealing_args = parse_sim_annealing_arguments(parse_args)
    parameter_dict = get_sim_annealing_args(sim_annealing_args)
    run_parameters = parameter_dict['run_parameters']
    annealing_parameters = parameter_dict['annealing_parameters']
    search_parameters = parameter_dict['search_parameters']
    decomposition_parameters = parameter_dict['decomposition_parameters']
    if isinstance(decomposition_parameters['file'], str):
        decomposition_parameters['file'] = [decomposition_parameters['file']]

    decomposition_args_list = []
    for f in decomposition_parameters['file']:
        (df,
         df_true,
         decomposition_run_parameters,
         data_parameters,
         thermal_core_ma_args,
         smoothing_ma_args,
         physical_parameters,
         binning_parameters,
         spline_parameters,
         debug_dict) = decomposition_preparation({'yaml':f})
        decomposition_args_dict = {'df': df,
                                   'df_true': df_true,
                                   'run_parameters': decomposition_run_parameters,
                                   'thermal_core_ma_args': thermal_core_ma_args,
                                   'smoothing_ma_args': smoothing_ma_args,
                                   'physical_parameters': physical_parameters,
                                   'binning_parameters': binning_parameters,
                                   'spline_parameters': spline_parameters
                                   }
        if run_parameters['save']:
            if run_parameters['output_folder'] is None:
                run_parameters['output_folder'] = os.path.join(decomposition_args_dict['run_parameters']['output_folder'],
                                                               'sim_annealing')

            destination_folder = run_parameters['output_folder']
            os.makedirs(destination_folder, exist_ok=True)

        else:
            destination_folder = ''

        decomposition_args_list.append(deepcopy(decomposition_args_dict))
        del decomposition_args_dict

    return run_parameters, annealing_parameters, search_parameters, decomposition_args_list


def print_fun(x_previous, f_previous, x_candidate, f_candidate, accept_probability, accepted, new_global_minimum):
    if new_global_minimum:
        logger.info(f"NEW GLOBAL MINIMUM at {x_previous}, with loss {f_previous}")
    else:
        logger.info(f"at minimum {x_previous}, with loss {f_previous} - with {accept_probability}")


def add_sim_annealing_parser():
    parser = argparse.ArgumentParser('Decompose air velocity from bird velocity')

    parser.add_argument('-y', '--yaml', dest='yaml', type=str,
                        help='path to yaml file to read configuration from')
    parser.add_argument('-dy', '--decomposition_yaml', dest='decomposition_yaml', type=str,
                        help='path to yaml file to read decomposition configuration from')
    parser.add_argument('-o', '--output', dest='output_folder', type=str, default=None,
                        help='folder to write iterations.pkl, thermal_core.pkl and bins.pkl to')
    parser.add_argument('-n', '--n-iterations', dest='n_iterations', type=int, default=None,
                        help='number of iterations to calculate')
    parser.add_argument('-L', '--CL', dest='CL', type=float, help='CL. Overrides decomposition CL')
    parser.add_argument('-D', '--CD', dest='CD', type=float, help='CD. Overrides decomposition CD')
    parser.add_argument('-M', '--mass', dest='mass', type=float, help='mass. Overrides decomposition mass')
    parser.add_argument('-W', '--wing-area', dest='wing_area', type=float,
                        help='wing_area. Overrides decomposition wing_area')
    parser.add_argument('-ig', '--initial-guess', dest='initial_guess', nargs='*', type=float, default=None,
                        help='initial-guess')
    parser.add_argument('-tc', '--true-columns', dest='true_columns', nargs='*',
                        help='true_columns'
                             'Can take several values from the following list: CL, CD, mass, wing_area')
    parser.add_argument('-p', '--parameters', dest='parameters_to_search', nargs='*',
                        help='parameters to search for on simulated annealing. '
                             'Can take several values from the following list: CL, CD, mass, wing_area')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true', default=None,
                        help='whether or not to show debugging features')
    parser.add_argument('-dr', '--dryrun', dest='save', action='store_false', default=None,
                        help='whether or not to show debugging features')
    parser.add_argument('-v', '--verbose', dest='verbosity', type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help='Level of verbosity')

    return parser