import datetime
import logging
import os
import pprint

import numpy as np
import pandas as pd
import yaml

from decomposition.auxiliar import parse_decomposition_arguments, decomposition_preparation
from decomposition.core import start_decomposition
from misc.auxiliar import config_logger, NoAliasDumper, sanitize_dict_for_yaml
from simulated_annealing.auxiliar import add_sim_annealing_parser, parse_sim_annealing_arguments, \
    sim_annealing_preparation
from simulated_annealing.core import simulated_annealing_for_parameter_search

def main(parser):
    run_time = datetime.datetime.isoformat(datetime.datetime.now()).split('.')[0]
    parse_args = parser.parse_args().__dict__

    run_parameters,  annealing_parameters, search_parameters, decomposition_args_list = sim_annealing_preparation(parse_args)

    logger = logging.getLogger()
    config_logger(logger, output_dir=run_parameters['output_folder'],
                  verbosity=run_parameters['verbosity'], log_to_file=run_parameters['save'])

    logger.info('RUN PARAMETERS')
    for line in pprint.pformat(run_parameters).split('\n'):
        logger.log(level=logging.INFO, msg=line)

    logger.info('ANNEALING PARAMETERS')
    for line in pprint.pformat(annealing_parameters).split('\n'):
        logger.log(level=logging.INFO, msg=line)
    logger.info('SEARCH PARAMETERS')
    for line in pprint.pformat(search_parameters).split('\n'):
        logger.log(level=logging.INFO, msg=line)
    #decomposition_args = parse_decomposition_arguments({'yaml': decomposition_parameters['file']})

    optimize_res, history = simulated_annealing_for_parameter_search(run_parameters,  annealing_parameters,
                                                                     search_parameters, decomposition_args_list)

    x0 = optimize_res.x
    print(x0)
    print(history)
    if run_parameters['save']:
        simulated_annealing_root_path = run_parameters['output_folder']
        annealing_parameters.pop('kwargs')
        path_to_history_file = os.path.join(simulated_annealing_root_path, 'annealing_history.csv')
        path_to_parameters_file = os.path.join(simulated_annealing_root_path,
                                               f'sim_annealing_decomposition_parameters_{run_time}.yaml')
        history_exists = os.path.exists(path_to_history_file)
        should_append = history_exists and run_parameters['continue_annealing']
        if not should_append:  # (OVER)WRITE

            history.to_csv(path_to_history_file, mode='w', index=False, )
            with open(path_to_parameters_file, 'w') as f:
                yaml.dump(sanitize_dict_for_yaml({'run_parameters': run_parameters,
                                                  'annealing_parameters': annealing_parameters,
                                                  'search_parameters': search_parameters,
                                                  }),
                          f, Dumper=NoAliasDumper)
        else:  # APPEND

            history.to_csv(path_to_history_file, mode='a', index=False, header=False)

            with open(path_to_parameters_file, 'w') as f:
                yaml.dump(sanitize_dict_for_yaml({'run_parameters': run_parameters,
                                                  'annealing_parameters': annealing_parameters,
                                                  'search_parameters': search_parameters,
                                                  }),
                          f, Dumper=NoAliasDumper)
        with open(os.path.join(simulated_annealing_root_path, f'sim_annealing_results.yaml'), 'w') as f:
            yaml.dump(sanitize_dict_for_yaml(dict(optimize_res)),
                      f, Dumper=NoAliasDumper)

    full_list_of_birds = []
    for col in history.columns[:len(x0)]:
        for i, parameter in enumerate(search_parameters['parameters_to_search']):
            col = col.replace(parameter + '_', '')
        full_list_of_birds.append(col)
    x0 = list(map(float, x0))
    n_birds = len(full_list_of_birds)
    if search_parameters['individual_search']:
        df_x0 = pd.DataFrame(index=full_list_of_birds)
        for i, parameter in enumerate(search_parameters['parameters_to_search']):
            df_x0[parameter] = x0[i * n_birds: (i + 1) * n_birds]

    for i_decomp, decomposition_args_dict in enumerate(decomposition_args_list):
        if not run_parameters['save']:
            decomposition_args_dict['run_parameters']['output_folder'] =  ''
        else:
            simulated_annealing_root_path = run_parameters['output_folder']
            decomposition_args_dict['run_parameters']['output_folder'] = os.path.join(simulated_annealing_root_path, str(i_decomp))

        # Decomposition with best result
        if search_parameters['individual_search']:
            decomposition_args_dict['physical_parameters'] = pd.merge(decomposition_args_dict['physical_parameters'].drop(columns=search_parameters['parameters_to_search']), df_x0,
                                                                      left_on=['bird_name'], right_index=True, how='inner')
        else:
            for i, parameter in enumerate(search_parameters['parameters_to_search']):
                decomposition_args_dict['physical_parameters'][parameter] = float(x0[i])
    
        _ = start_decomposition(decomposition_args_dict['df'],
                                decomposition_args_dict['run_parameters'],
                                decomposition_args_dict['physical_parameters'],
                                decomposition_args_dict['thermal_core_ma_args'],
                                decomposition_args_dict['smoothing_ma_args'],
                                decomposition_args_dict['binning_parameters'],
                                decomposition_args_dict['spline_parameters'],
                                )

        with open(os.path.join(decomposition_args_dict['run_parameters']['output_folder'], 'decomposition_args.yaml'), 'w') as f:

            yaml.dump({'run_parameters':       decomposition_args_dict['run_parameters'],
                       'thermal_core_ma_args':       decomposition_args_dict['thermal_core_ma_args'],
                       'smoothing_ma_args':          decomposition_args_dict['smoothing_ma_args'],
                       'physical_parameters':        decomposition_args_dict['physical_parameters'].set_index('bird_name').to_dict(orient='index'),
                       'binning_parameters':         decomposition_args_dict['binning_parameters'],
                       'spline_parameters':          decomposition_args_dict['spline_parameters']
                       },
                      f, default_flow_style=False)


if __name__ == '__main__':
    parser = add_sim_annealing_parser()

    main(parser)