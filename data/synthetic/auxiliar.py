import logging
import numbers
import pprint
import time

import numpy as np
import yaml
import dill as pickle
from calc.thermal import get_ND_random_walk
from object.air import AirVelocityField

logger = logging.getLogger(__name__)
def get_flock_stats(list_of_bird_parameters, return_data=False):
    if isinstance(list_of_bird_parameters, str):
        with open(list_of_bird_parameters, 'r') as f:
            list_of_bird_parameters = yaml.load(f, yaml.FullLoader)

    one_bird = list_of_bird_parameters[00]

    stats_dict = {}
    for parameter_set in one_bird:
        if not isinstance(one_bird[parameter_set], (dict, numbers.Number)):
            continue
        if isinstance(one_bird[parameter_set], dict):
            stats_dict[parameter_set] = {}
        for parameter in one_bird[parameter_set]:

            if isinstance(one_bird[parameter_set][parameter], numbers.Number):

                parameter_data = list(map(lambda x: x[parameter_set][parameter],
                                          list_of_bird_parameters))

                parameter_stats = {'mean': np.mean(parameter_data),
                                   'std': np.std(parameter_data),
                                   'N': len(parameter_data)}
                if return_data:
                    parameter_stats['sample'] = parameter_data
                stats_dict[parameter_set][parameter] = parameter_stats.copy()
            elif isinstance(one_bird[parameter_set][parameter], dict):
                stats_dict[parameter_set][parameter] = {}
                for subparameter in one_bird[parameter_set][parameter]:
                    parameter_data = list(map(lambda x: x[parameter_set][parameter][subparameter],
                                              list_of_bird_parameters))

                    parameter_stats = {'mean': np.mean(parameter_data),
                                       'std': np.std(parameter_data),
                                       'N': len(parameter_data)}
                    if return_data:
                        parameter_stats['sample'] = parameter_data

                    stats_dict[parameter_set][parameter][subparameter] = parameter_stats.copy()

    return stats_dict


DEFAULT_YAML_PATH = 'config/default/bird_generate.default.yaml'
PARAMETERS_DICT = {
    'run': ['save_folder', 'n_birds', 'dt', # in seconds,
            'dt_to_save',# in seconds
            'duration',  # in seconds,
            'noise_level',
            'prefix',
            'skip_landed',
            'random_seed',
            'inspect_before_saving',
            'inspect_before_running',
            'verbosity'],
    'initial_conditions': ['rho', 'phi', 'z', 'bank_angle', 'bearing'],
    'physical_parameters': ['CL', 'CD', 'mass', 'wing_area'],
    'control_parameters': {'general_args': ['period',
                                            'bank_angle_max',
                                            'delta_bank_angle_max',
                                            'sigma_noise_degrees',
                                            'N_out_of_thermal',
                                            'exploration_exploitation_ratio',
                                            'debug'],
                           'exploration': ['alpha_degrees', 'time'],
                           'exploitation': ['K', 'thermalling_bank_angle'],
                           'glide': ['trigger', 'direction']},
    'air': [{'thermal': ['rotation', 'profile']},
            'wind',
            'rho',
            'turbulence']
}


def get_defaults_for_air(component):
    with open(DEFAULT_YAML_PATH) as f:
        default_yaml = yaml.load(f, Loader=yaml.FullLoader)

    if component != 'thermal':
        if component in default_yaml['air']:
            return parse_air_yaml(default_yaml['air'][component])
        else:
            return None
    else:
        if component == 'rotation':
            parameters = {'radius': 20,
                          'A': 2}
        else:
            parameters = {'radius': 20,
                          'A': 3}

        if component == 'rotation':
            def function(r, theta, z, t):
                import numpy as np
                radius = parameters['radius']
                A_rotation = parameters['A']

                # K is the constant so that magnitude is A_rotation at r=radius/2
                K = 4 * A_rotation / radius ** 2
                if r > radius:
                    return [0, 0]
                else:
                    magnitude = K * r * (radius - r)

                    return [-magnitude * np.sin(theta),
                            magnitude * np.cos(theta)]
        else:
            def function(r, theta, z, t):
                import numpy as np
                A = parameters['A']
                radius = parameters['radius']

                return A * np.exp(-np.power(r, 2.) / (2 * np.power(radius, 2.)))
        return function


def get_defaults(parameter, parameter_set):
    with open(DEFAULT_YAML_PATH) as f:
        default_yaml = yaml.load(f, Loader=yaml.FullLoader)

    if parameter_set == 'run':
        return parse_bird_yaml(default_yaml['run'][parameter])
    if parameter_set == 'initial_conditions':
        return parse_bird_yaml(default_yaml['bird']['initial_conditions'][parameter])
    if parameter_set == 'physical_parameters':
        return parse_bird_yaml(default_yaml['bird']['physical_parameters'][parameter])
    if 'control' in parameter_set:
        parameter_subset = parameter_set.split('_')[1:]
        parameter_subset = '_'.join(parameter_subset)
        if (parameter_subset not in default_yaml['bird']['control_parameters']) or (default_yaml['bird']['control_parameters'][parameter_subset] is None):
            return None
        else:
            if isinstance(default_yaml['bird']['control_parameters'][parameter_subset], dict):
                return parse_bird_yaml(default_yaml['bird']['control_parameters'][parameter_subset][parameter])
            else:
                return parse_bird_yaml(default_yaml['bird']['control_parameters'][parameter_subset])
    if parameter_set == 'air':
        return get_defaults_for_air(parameter)


def parse_bird_yaml(yaml_struct, bird_idx=None):
    # CONSTANT
    if isinstance(yaml_struct, (int, float, str, type(None))):
        return yaml_struct
    # FROM LIST
    elif isinstance(yaml_struct, list):
        return yaml_struct[bird_idx]
    else:
        if 'distribution' in yaml_struct:
            # FROM RANDOM DISTRIBUTION
            try:
                distribution = getattr(np.random, yaml_struct['distribution'])
            except AttributeError as e:
                raise Exception("distribution keyword must match one distribution provided by numpy. " +
                                "Check https://numpy.org/doc/1.16/reference/routines.random.html#distributions")

            return distribution(**yaml_struct['parameters'])
        else:
            return yaml_struct


def generate_air_data(generate_yaml, shape=2):
    limits = generate_yaml['limits']
    n_steps = generate_yaml['n_steps']
    n_vars = len(limits)
    used_vars = list(limits.keys())
    used_vars = ''.join(used_vars)
    if n_vars > 1:
        mgs = np.meshgrid(*[np.linspace(*limits, generate_yaml['n_steps']) for limits in limits.values()])
        xyzt_array = np.stack([mg.flatten() for mg in mgs], axis=-1)
    else:
        xyzt_array = np.linspace(*(limits[used_vars]), generate_yaml['n_steps'])
    if generate_yaml['method'] == 'random_walk':
        generate_yaml.pop('method')
        values = get_ND_random_walk(generate_yaml['mean'],
                                    generate_yaml['std'],n_vars,n_steps)

    elif generate_yaml['method'] == 'random':

        if 'distribution' in generate_yaml:
            # FROM RANDOM DISTRIBUTION
            try:
                distribution = getattr(np.random, generate_yaml['distribution'])
            except AttributeError as e:
                raise Exception("distribution keyword must match one distribution provided by numpy. " +
                                "Check https://numpy.org/doc/1.16/reference/routines.random.html#distributions")
            if 'multivariate' in generate_yaml['distribution']:
                values = distribution(size=(n_steps ** n_vars), **generate_yaml['parameters'])
            else:
                values = distribution(size=(n_steps ** n_vars, shape), **generate_yaml['parameters'])

    return {'XYZT_values': xyzt_array, 'values': values, 'used_vars': used_vars}


def get_air_function(yaml_struct):

    from importlib import import_module
    try:
        my_mod = import_module(yaml_struct['file'].replace('/', '.').replace('.py', ''))
        my_function = getattr(my_mod, yaml_struct['function'] )
    except AttributeError as e:
        raise Exception(f"function {yaml_struct['function']} unrecognized")
    return {'function': my_function,
            'args': yaml_struct.get('parameters', {})}



def parse_air_yaml(yaml_struct):
    # CONSTANT
    if isinstance(yaml_struct, (int, float, str, type(None), list)):
        return yaml_struct
    else:
        if 'generate' in yaml_struct:
            return generate_air_data(yaml_struct['generate'])
        # METHOD - FUNCTION
        elif 'from_function' in yaml_struct:
            return get_air_function(yaml_struct['from_function'])
        elif 'from_data' in yaml_struct:
            yaml_struct['from_data']['XYZT_values'] = np.array(yaml_struct['from_data']['XYZT_values'])
            yaml_struct['from_data']['values'] = np.array(yaml_struct['from_data']['values'])
            return yaml_struct['from_data']
        else:
            return yaml_struct


def get_initial_condition_from_yaml(initial_conditions_yaml, i=None):
    initial_conditions = {}
    for coord in PARAMETERS_DICT['initial_conditions']:

        if coord in initial_conditions_yaml:
            initial_conditions[coord] = parse_bird_yaml(initial_conditions_yaml[coord], i)
        else:
            initial_conditions[coord] = get_defaults(coord, 'initial_conditions')

    initial_conditions['bank_angle'] *= np.pi / 180

    return initial_conditions


def get_physical_parameters_from_yaml(physical_parameters_yaml, i=None):
    physical_parameters = {}
    for parameter in PARAMETERS_DICT['physical_parameters']:
        if parameter in physical_parameters_yaml:
            physical_parameters[parameter] = parse_bird_yaml(physical_parameters_yaml[parameter], i)
        else:
            physical_parameters[parameter] = get_defaults(parameter, 'physical_parameters')

    return physical_parameters


def get_control_parameters_from_yaml(control_parameters_yaml, i=None):
    list_parameter_sets = list(PARAMETERS_DICT['control_parameters'].keys())
    control_parameters = {parameter_set: {} for parameter_set in list_parameter_sets}
    # GENERAL ARGS

    for parameter_set in list_parameter_sets:
        # If parameter_set is not in yaml, then go to default
        if parameter_set not in control_parameters_yaml:
            for parameter in PARAMETERS_DICT['control_parameters'][parameter_set]:
                control_parameters[parameter_set][parameter] = get_defaults(parameter, f'control_{parameter_set}')
        else:
            # If parameter_set is False, then return None
            if control_parameters_yaml[parameter_set] is False:
                control_parameters[parameter_set] = None
            # If parameter_set is present and not None, then parse yaml
            elif control_parameters_yaml[parameter_set] is not None:
                for parameter in PARAMETERS_DICT['control_parameters'][parameter_set]:
                    if parameter in control_parameters_yaml[parameter_set]:
                        control_parameters[parameter_set][parameter] = parse_bird_yaml(
                            control_parameters_yaml[parameter_set][parameter], i)
                    else:
                        control_parameters[parameter_set][parameter] = get_defaults(parameter, f'control_{parameter_set}')
            else:
                # If parameter_set is None, then go to default
                for parameter in PARAMETERS_DICT['control_parameters'][parameter_set]:
                    control_parameters[parameter_set][parameter] = get_defaults(parameter, f'control_{parameter_set}')


    return control_parameters


def get_air_parameters_from_yaml(air_yaml):
    air_parameters = {}

    for component in PARAMETERS_DICT['air']:
        if not isinstance(component, dict):  # This is a proxy for thermal
            if component not in air_yaml:
                air_parameters[component] = get_defaults_for_air(component)
            else:
                air_parameters[component] = parse_air_yaml(air_yaml[component])
        else:
            air_parameters['thermal'] = {}

            for parameter in PARAMETERS_DICT['air'][0]['thermal']:
                if parameter not in air_yaml['thermal']:
                    air_parameters['thermal'][parameter] = get_defaults_for_air(parameter)
                else:
                    air_parameters['thermal'][parameter] = parse_air_yaml(air_yaml['thermal'][parameter])

    return air_parameters


def get_air_velocity(air_yaml, duration):
    if 'from_file' in air_yaml:
        with open(air_yaml['from_file'], 'rb') as f:
            air_velocity_field_obj = pickle.load(f)
            air_parameters = air_velocity_field_obj._air_parameters
    else:
        air_parameters = get_air_parameters_from_yaml(air_yaml)

        air_velocity_field_obj = AirVelocityField(air_parameters=air_parameters,
                                                  t_start_max=duration + AirVelocityField.config['time_resolution'], z_max_limit=1000)
    return air_velocity_field_obj, air_parameters


def get_run_parameters_from_yaml(run_yaml):
    run_parameters = {}

    for parameter in PARAMETERS_DICT['run']:

        if parameter in run_yaml:
            run_parameters[parameter] = parse_bird_yaml(run_yaml[parameter])
        else:
            run_parameters[parameter] = get_defaults(parameter, 'run')

    if ('random_seed' in run_parameters) and (run_parameters['random_seed'] is not None):
        np.random.seed(run_parameters['random_seed'])
    else:
        random_seed = int(time.time())
        np.random.seed(random_seed)
        run_parameters['random_seed'] = random_seed

    return run_parameters


def get_bird_parameters_from_file(path_to_file):
    with open(path_to_file, 'r') as f:
        parameters = yaml.safe_load(f)
    list_of_bird_names = list(map(lambda x: x['bird_name'], parameters))
    list_of_init_condition = list(map(lambda x: x['initial_conditions'], parameters))
    list_of_physical_parameters = list(map(lambda x: x['physical_parameters'], parameters))
    list_of_control_parameters = list(map(lambda x: x['control_parameters'], parameters))

    return list_of_bird_names, list_of_init_condition, list_of_physical_parameters, list_of_control_parameters


def get_bird_parameters_from_yaml(parameters_dict, **kwargs):
    if 'from_file' in parameters_dict:
        return get_bird_parameters_from_file(parameters_dict['from_file'])
    else:
        return generate_bird_parameters_from_yaml(parameters_dict, **kwargs)

def generate_bird_parameters_from_yaml(parameters_dict, air_velocity_field_obj, n_birds):
    list_of_bird_names = []
    list_of_init_condition = []
    list_of_physical_parameters = []
    list_of_control_parameters = []
    i = 0
    n_characters = int(np.ceil(np.log10(n_birds)))
    while i < n_birds:

        init_condition, physical_parameters, control_parameters = generate_single_bird_parameters_from_yaml(parameters_dict,
                                                                                                            i)

        core = air_velocity_field_obj.get_thermal_core(init_condition['z'], 0)
        init_condition['x'] = float(core[0] + init_condition['rho'] * np.cos(init_condition['phi']))
        init_condition['y'] = float(core[1] + init_condition['rho'] * np.sin(init_condition['phi']))

        for line in pprint.pformat(init_condition).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)
        for line in pprint.pformat(physical_parameters).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)
        for line in pprint.pformat(control_parameters).split('\n'):
            logger.log(level=logging.DEBUG, msg=line)

        bird_name = ('0' * n_characters + str(i))
        bird_name = 'sb_' + bird_name[-n_characters:]

        list_of_bird_names.append(bird_name)
        list_of_init_condition.append(init_condition)
        list_of_physical_parameters.append(physical_parameters)
        list_of_control_parameters.append(control_parameters)
        i = i + 1

    return list_of_bird_names, list_of_init_condition, list_of_physical_parameters, list_of_control_parameters

def generate_single_bird_parameters_from_yaml(yaml_dict=None, i=None):

    if yaml_dict is None:
        bird_yaml = {'initial_conditions': {},
                     'physical_parameters': {},
                     'control_parameters': {}}
    else:
        bird_yaml = yaml_dict

    initial_conditions = get_initial_condition_from_yaml(bird_yaml['initial_conditions'], i=i)
    physical_parameters = get_physical_parameters_from_yaml(bird_yaml['physical_parameters'], i=i)
    control_parameters = get_control_parameters_from_yaml(bird_yaml['control_parameters'], i=i)

    return initial_conditions, physical_parameters, control_parameters