import concurrent.futures
import logging
import os
from copy import deepcopy
from itertools import repeat

import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.optimize import dual_annealing

from calc.auxiliar import get_na_mask
from decomposition.auxiliar import decomposition_preparation, parse_decomposition_arguments, \
    calculate_decomposition_loss
from decomposition.core import start_decomposition
from simulated_annealing.auxiliar import print_fun
from simulated_annealing.calc import my_annealing

logger = logging.getLogger(__name__)


def velocity_decomposition_wrapper_multiple_thermals(p, parameters_to_iterate, decomposition_kwargs_list, list_of_unique_birds,
                                                     save_iterations, loss_function='horizontal_air_velocity_median', na_penalty=0,
                                                     true_values_list=None,
                                                     individual_search=False, history=None, n_processors=None):
    # ===============     PREPARE DECOMPOSITION     ===============
    if true_values_list is None:
        true_values_list = [None, None]

    current_physical_parameters = pd.DataFrame(list_of_unique_birds,columns=['bird_name'])
    n_birds = len(current_physical_parameters['bird_name'])
    if individual_search:
        for i, param in enumerate(parameters_to_iterate):
            current_physical_parameters[param] = p[i * n_birds: (i + 1) * n_birds]
    else:
        for i, param in enumerate(parameters_to_iterate):
            current_physical_parameters[param] = float(p[i])


    logger.info(current_physical_parameters)
    current_decomposition_kwargs_list = []
    for i in range(len(decomposition_kwargs_list)):

        current_decomposition_kwargs = deepcopy(decomposition_kwargs_list[i])

        if current_decomposition_kwargs['run_parameters']['output_folder'] is not None:
            folder_name = '_'.join([f'{parameter}={p[i]:.6g}' for i, parameter in enumerate(parameters_to_iterate)])
            current_decomposition_kwargs['run_parameters']['output_folder'] = os.path.join(current_decomposition_kwargs['run_parameters']['output_folder'],
                                                                       folder_name)
        current_decomposition_kwargs['run_parameters']['save'] = save_iterations

        current_decomposition_kwargs['initial_physical_parameters'] = pd.merge(
            current_decomposition_kwargs['initial_physical_parameters'].drop(columns=parameters_to_iterate),
            current_physical_parameters, on='bird_name')
        current_decomposition_kwargs_list.append(current_decomposition_kwargs)

    loss_list = []
    if len(current_decomposition_kwargs_list) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_processors) as executor:
            result = executor.map(velocity_decomposition_wrapper,
                                  current_decomposition_kwargs_list,
                                  repeat(loss_function),
                                  repeat(na_penalty),
                                  true_values_list)

            for l in result:
                loss_list.append(l)
                logger.debug(f'loss={l:.4f}')
    else:
        loss_list.append(velocity_decomposition_wrapper(current_decomposition_kwargs_list[0], loss_function, na_penalty, true_values_list[0]))

    loss = np.max(loss_list)
    if history is not None:
        history.append([current_physical_parameters[param] for param in parameters_to_iterate] + [loss])
    logger.debug(loss_list)
    logger.info(f'{loss=:.4f}')
    return loss

def velocity_decomposition_wrapper(run_args,
                                   loss_function='horizontal_air_velocity_median', na_penalty=0,
                                   true_values=None):
    max_allowed_curvature = 0.10
    # ===============     RUN DECOMPOSITION     ===============
    return_tuple = start_decomposition(**run_args)


    # ===============     CALCULATE LOSS     ===============

    list_of_losses = return_tuple[-1]
    list_of_losses = np.array(list_of_losses)
    best_iteration = np.argmin(list_of_losses[:, 0])
    loss, N_NA, N_total = list_of_losses[best_iteration]

    loss = loss + na_penalty * (N_NA / N_total)

    return loss


def simulated_annealing_for_parameter_search(run_parameters, annealing_parameters, search_parameters,
                                             decomposition_args_list):
    search_limits = []
    list_of_parameters = []
    list_of_birds = []
    for decomposition_args_dict in decomposition_args_list:

        list_of_birds += decomposition_args_dict['physical_parameters']['bird_name'].values.tolist()

    list_of_birds = np.unique(list_of_birds)
    list_of_birds = np.sort(list_of_birds)
    n_birds = len(list_of_birds)
    for param in search_parameters['parameters_to_search']:
        if search_parameters['individual_search']:
            search_limits += [search_parameters['search_limits'][param]] * n_birds
            list_of_parameters += [f'{param}_{bird_name}' for bird_name in list_of_birds]
        else:
            search_limits += [search_parameters['search_limits'][param]]
            list_of_parameters += [param]

    history_list = []  # np.empty((1,4))

    annealing_parameters['bounds'] = search_limits
    annealing_parameters['kwargs'] = {'parameters_to_iterate': search_parameters['parameters_to_search'],
                                      'loss_function': search_parameters['loss_function'],
                                      'na_penalty': search_parameters['na_penalty'],
                                      'individual_search': search_parameters['individual_search'],
                                      'history': history_list,  # history=
                                      'save_iterations': run_parameters['save_all_iterations'],
                                      'n_processors': run_parameters['n_processors']
                                      }
    if annealing_parameters['minimizer_kwargs'] is None:
        annealing_parameters['no_local_search'] = True
        annealing_parameters['minimizer_kwargs'] = {}


    annealing_parameters['kwargs']['list_of_unique_birds'] = list_of_birds #.append(decomposition_args_dict['physical_parameters'])
    annealing_parameters['kwargs']['true_values_list'] = []
    annealing_parameters['kwargs']['decomposition_kwargs_list'] = []

    for decomposition_args_dict in decomposition_args_list:
        if 'df_true' in decomposition_args_dict:
            df_true = decomposition_args_dict.pop('df_true')
        else:
            df_true = None
        decomposition_args_dict['run_parameters']['output_folder'] = run_parameters['output_folder']

        annealing_parameters['kwargs']['true_values_list'].append(df_true)  # true_values
        annealing_parameters['kwargs']['decomposition_kwargs_list'].append({'df': decomposition_args_dict['df'],
                 'run_parameters': decomposition_args_dict['run_parameters'],
                 'initial_physical_parameters': decomposition_args_dict['physical_parameters'],
                 'thermal_core_ma_args': decomposition_args_dict['thermal_core_ma_args'],
                 'smoothing_ma_args': decomposition_args_dict['smoothing_ma_args'],
                 'binning_parameters': decomposition_args_dict['binning_parameters'],
                 'spline_parameters': decomposition_args_dict['spline_parameters'], })


    path_to_history = os.path.join(run_parameters['output_folder'], 'annealing_history.csv')
    if os.path.exists(path_to_history) and run_parameters['continue_annealing']:
        history = pd.read_csv(path_to_history)
        with open(os.path.join(run_parameters['output_folder'], 'sim_annealing_results.yaml'), 'r') as f:
            sim_result = yaml.safe_load(f)
            annealing_parameters['i_best'] = sim_result['i']
            annealing_parameters['x_best'] = sim_result['x']
            annealing_parameters['f_best'] = sim_result['fun']

        last_iteration = len(history) - 1
        annealing_parameters['x0'] = history.tail(1)[list_of_parameters].values[0]
        annealing_parameters['start_from'] = last_iteration + 1

    if annealing_parameters['x0'] is None:
        annealing_parameters['x0'] = [np.random.uniform(*b) for b in search_limits]
    elif np.isscalar(annealing_parameters['x0']) and search_parameters['individual_search']:
        annealing_parameters['x0'] = [annealing_parameters['x0']] * n_birds

    optimize_res, history = my_annealing(velocity_decomposition_wrapper_multiple_thermals, **annealing_parameters, new_best_callback=print_fun)

    history = pd.DataFrame(history, columns=list_of_parameters + ['loss']
                                             + [p + '_candidate' for p in list_of_parameters] + ['loss_candidate']
                                             + ['acceptance_probability', 'accept',
                                                'new_global_minimum', 'temperature'])
    return optimize_res, history
    # ret = basinhopping(loss_func, x0, minimizer_kwargs=minimizer_kwargs, disp=True,
    #                    callback=lambda a, b, c: print_fun(a, b, c, history=history_list),
    #                    niter=3)
