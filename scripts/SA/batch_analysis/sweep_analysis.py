
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import PermutationMethod

from data.get_data import load_synthetic_data
from decomposition.post import  get_metrics_from_path
# sweep_var_dict = {'EER_sweep_avg':    'EER_avg',
#                   'EER_sweep_std':    'EER_std',
#                   'constant_wind':    'wind',
#                   'random_walk':      'wind_avg',
#                   'turbulence_noise': 'turbulence',
#                   'turbulence_noise': 'noise_level',
#                   'sweep_offset_CD':  'CD',
#                   'sweep_offset_CL':  'CL',
#                'rotation': 'rot_int',
#                 'WL_sweep_avg_6_std_06': 'WL',
#                 'wing_loading_sweep_std': 'WL_std',
#                   }
wild_card_dict = {'EER_sweep_avg':    'E*/*',
                  'EER_sweep_std':    'E*/*',
                  'constant_wind':    'w*/*',
                  'random_walk':      'w*/*',
                  'rotation':  'r*/*/*',
                  'turbulence_noise': 'n*/*/*',
                  'sweep_offset_CD':  'C*/*',
                  'sweep_offset_CL':  'C*/*',
                'WL_sweep_avg_6_std_06': 'W*/*',
                'wing_loading_sweep_std': 'W*/*',
                  }

wild_card_dict = {k:v for k,v in wild_card_dict.items() if k == 'turbulence_noise'}
sweep_parameters_functions_dict= {'EER_avg': lambda synth: np.mean(list(map(lambda x: x['control_parameters']['general_args']['exploration_exploitation_ratio'],
                                                                            synth['bird_parameters']))
                                                                   ),
                                  'EER_std':  lambda synth: np.std(list(map(lambda x: x['control_parameters']['general_args']['exploration_exploitation_ratio'],
                                                                            synth['bird_parameters']))
                                                                   ),
                                  'wind': lambda synth: synth['parameters']['air']['wind'][0] if isinstance(synth['parameters']['air']['wind'], (list, tuple)) else None,
                                  'wind_avg':  lambda synth: synth['parameters']['air']['wind']['generate']['mean'][0] if 'generate' in synth['parameters']['air']['wind'] else None,
                                  'turbulence': lambda synth: synth['parameters']['air']['turbulence']['normalization'] if synth['parameters']['air']['turbulence'] is not None else 0,
                                  'noise_level': lambda synth: synth['parameters']['run']['noise_level'],
                                  'rot_int':  lambda synth: synth['parameters']['air']['thermal'][
                                      'rotation']['from_function']['parameters']['A_rotation'] if synth['parameters']['air']['thermal']['rotation'] is not None else 0,
                                  'rot_radius': lambda synth: synth['parameters']['air']['thermal'][
                                      'rotation']['from_function']['parameters']['radius'] if synth['parameters']['air']['thermal']['rotation'] is not None else 0,
                                  'CD': lambda synth: np.mean(list(map(lambda x: x['physical_parameters']['CD'], synth['bird_parameters']))),
                                  'CL': lambda synth: np.mean(list(map(lambda x: x['physical_parameters']['CL'], synth['bird_parameters']))),
                                  }



parameter_dict_lambda = {'EER_sweep_avg':    lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'EER_sweep_std':    lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'constant_wind':    lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'random_walk':      lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'rotation':      lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[1].name.split('=')[0],
                                    'parameter_value': float(x.parents[1].name.split('=')[1]), },
                  'turbulence_noise': lambda x: {'realization':     x.name,
                                    'parameter_name_1':  x.parents[0].name.split('=')[0],
                                    'parameter_value_1': float(x.parents[0].name.split('=')[1]),
                                    'parameter_name_2':  x.parents[1].name.split('=')[0],
                                    'parameter_value_2': float(x.parents[1].name.split('=')[1]), },
                  'sweep_offset_CD':  lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'sweep_offset_CL':  lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'WL_sweep_avg_6_std_06':  lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), },
                  'wing_loading_sweep_std':  lambda x: {'realization':     x.name,
                                    'parameter_name':  x.parents[0].name.split('=')[0],
                                    'parameter_value': float(x.parents[0].name.split('=')[1]), }
                  }
root = '/home/pedro/PycharmProjects/ThermalModelling'
save = True
for parameter_sweep_type in wild_card_dict.keys():

    path_to_save = os.path.join(root, 'synthetic_data/from_atlasz/newdata', parameter_sweep_type, 'results')
    if save:
        os.makedirs(path_to_save, exist_ok=True)
    p = Path(os.path.join(root, 'synthetic_data/from_atlasz/newdata', parameter_sweep_type))


    df_sa_results = pd.DataFrame()
    df_correlations = pd.DataFrame()

    for ss in p.glob(wild_card_dict[parameter_sweep_type]):
        current_realization_path = str(ss)
        if not os.path.isdir(current_realization_path):
            continue

        current_path_to_annealing = os.path.join(current_realization_path, 'decomposition', 'average')
        current_path_to_history = os.path.join(current_path_to_annealing, 'annealing_history.csv')
        path_to_annealing_results = os.path.join(current_path_to_annealing, 'sim_annealing_results.yaml')
        if not os.path.exists(current_path_to_history):
            continue
        print(current_realization_path)
        syn = load_synthetic_data(current_realization_path)
        current_correlations, current_correlations_per_bird = get_metrics_from_path(decomposition_path=os.path.join(current_path_to_annealing, '0', 'final', 'reconstructed'),
                                                                                    iteration='best',
                                                                                    synthetic_path=current_realization_path,
                                                                                    do_per_bird=False,
                                                                                    method=PermutationMethod(1000,1000))


        current_correlations.to_csv(os.path.join(current_path_to_annealing, 'rms_correlations_post.csv'))
        if current_correlations_per_bird is not None:
            current_correlations_per_bird.to_csv(os.path.join(current_path_to_annealing, 'rms_correlations_per_bird_post.csv'))
        current_masses = np.array(list(map(lambda x: x['physical_parameters']['mass'], syn['bird_parameters'])))
        current_wing_areas = np.array(list(map(lambda x: x['physical_parameters']['wing_area'], syn['bird_parameters'])))
        current_WL = current_masses / current_wing_areas
        WL_avg = np.mean(current_WL)
        WL_std = np.std(current_WL)
        wind_steps = np.nan
        parameter_dict = parameter_dict_lambda[parameter_sweep_type](ss)

        sweep_parameters_values_dict = {}
        for sweep_var, f in sweep_parameters_functions_dict.items():
            sweep_parameters_values_dict[sweep_var] = f(syn)
        try:
            history = pd.read_csv(current_path_to_history, delimiter=',', )
        except FileNotFoundError as e:
            print(e)
            continue
        with open(path_to_annealing_results, 'r') as f:
            current_sa_result = yaml.safe_load(f)

        last_iteration = len(history) - 1
        current_row = {'WL_avg':     WL_avg,
                       'WL_std':     WL_std,
                       'WL_SA': current_sa_result['x'][0],
                       'loss': current_sa_result['fun'],
                       } | sweep_parameters_values_dict | parameter_dict
        for k,v in current_row.items():
            current_correlations[k] = v

        #current_correlations['turb_intensity'] = turb_intensity
        #current_correlations['noise_level'] = noise_level
        df_sa_results = pd.concat([df_sa_results, pd.Series(current_row).to_frame().T], ignore_index=True)
        df_correlations = pd.concat([df_correlations, current_correlations])

    if save:
        df_sa_results.to_csv(os.path.join(path_to_save, 'sweep_annealing_result_post.csv'))
        df_correlations.to_csv(os.path.join(path_to_save, 'sweep_correlations_post.csv'))
