
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.get_data import load_synthetic_data, load_synthetic_and_decomposed
from decomposition.post import get_rms_and_correlations, get_rms_and_correlations_per_bird, get_metrics_from_path
from object.air import DecomposedAirVelocityField

from plotting.sim_annealing import plot_sim_annealing_summary_from_history, plot_sim_annealing_scatter_hist_2d

save = True
root = '/home/pedro/PycharmProjects/ThermalModelling'
very_base_path = os.path.join(root, 'synthetic_data/from_atlasz/newdata/turbulence_noise')
sweep_var_nominal = ['turbulence','noise_level']
sweep_parameters_functions_dict= {'EER_avg': lambda synth: np.mean(list(map(lambda x: x['control_parameters']['general_args']['exploration_exploitation_ratio'],
                                                                            synth['bird_parameters']))
                                                                   ),
                                  'EER_std':  lambda synth: np.std(list(map(lambda x: x['control_parameters']['general_args']['exploration_exploitation_ratio'],
                                                                            synth['bird_parameters']))
                                                                   ),
                                  'wind': lambda synth: synth['parameters']['air']['wind'][0] if isinstance(synth['parameters']['air']['wind'], (list, tuple)) else None,
                                  'wind_avg':  lambda synth: synth['parameters']['air']['wind']['generate']['mean'][0] if 'generate' in synth['parameters']['air']['wind'] else None,
                                  'turbulence': lambda synth: np.sqrt(0.37331674 ** 2 + 0.43644384 ** 2 + 0.3794553 ** 2) * synth['parameters']['air']['turbulence']['normalization'] if synth['parameters']['air']['turbulence'] is not None else 0.0,
                                  'noise_level': lambda synth: synth['parameters']['run']['noise_level'],
                                  'CD': lambda synth: np.mean(list(map(lambda x: x['physical_parameters']['CD'], synth['bird_parameters']))),
                                  'CL': lambda synth: np.mean(list(map(lambda x: x['physical_parameters']['CL'], synth['bird_parameters']))),
                                  }

df_sa_results = pd.DataFrame()
df_correlations = pd.DataFrame()


pp = Path(very_base_path)

for config_1 in pp.glob('n*'): #os.listdir(very_base_path):
    current_config = [str(config_1)]
    current_config_1_path = os.path.join(very_base_path, config_1)
    for config in os.listdir(current_config_1_path):
        current_config.append(config)
        current_config_2_path = os.path.join(current_config_1_path, config)
        for realization in os.listdir(current_config_2_path):
            current_realization_path = os.path.join(current_config_2_path, realization) # os.path.join(root,'synthetic_data/from_atlasz/turbulence_noise/noise=1.0/turbulence=1.8/0') #
            if not os.path.isdir(current_realization_path):
                continue

            print(current_realization_path)
            current_path_to_annealing = os.path.join(current_realization_path, 'decomposition', 'average')
            current_path_to_decomposition = os.path.join(current_path_to_annealing, '0')
            current_path_to_history = os.path.join(current_path_to_annealing, 'annealing_history.csv')
            path_to_annealing_results = os.path.join(current_path_to_annealing, 'sim_annealing_results.yaml')
            if not os.path.exists(current_path_to_history):
                continue
            syn = load_synthetic_data(current_realization_path)
            current_correlations, current_correlations_per_bird = get_metrics_from_path(current_path_to_decomposition,
                                                                                        iteration='best',
                                                                                        synthetic_path=current_realization_path)

            if save:
                current_correlations.to_csv(os.path.join(current_path_to_annealing, 'rms_correlations_post.csv'))
                current_correlations_per_bird.to_csv(os.path.join(current_path_to_annealing, 'rms_correlations_per_bird_post.csv'))

            #current_masses = np.array(syn['parameters']['bird']['physical_parameters']['mass'])
            #current_wing_areas = np.array(syn['parameters']['bird']['physical_parameters']['wing_area'])
            #current_WL = current_masses / current_wing_areas
            #WL_avg = np.round(np.mean(current_WL), 2)
            #WL_std = np.round(np.std(current_WL), 2)
            #turb_intensity = syn['parameters']['air']['turbulence']['normalization']
            #noise_level = syn['parameters']['run']['noise_level']
            current_masses = np.array(list(map(lambda x: x['physical_parameters']['mass'], syn['bird_parameters'])))
            current_wing_areas = np.array(list(map(lambda x: x['physical_parameters']['wing_area'], syn['bird_parameters'])))
            current_WL = current_masses / current_wing_areas
            WL_avg = np.mean(current_WL)
            WL_std = np.std(current_WL)
            wind_steps = np.nan
            sweep_var_nominal_values = float(config.split('=')[-1])
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

            current_row = {'realization': realization,
                           'WL_avg':     WL_avg,
                           'WL_std':     WL_std,
                           'WL_SA': current_sa_result['x'][0],
                           'loss': current_sa_result['fun'],
                           } | sweep_parameters_values_dict
            for k,v in current_row.items():
                current_correlations[k] = v

            #current_correlations['turb_intensity'] = turb_intensity
            #current_correlations['noise_level'] = noise_level
            df_sa_results = pd.concat([df_sa_results, pd.Series(current_row).to_frame().T], ignore_index=True)
            df_correlations = pd.concat([df_correlations, current_correlations])

if save:
    df_sa_results.to_csv(os.path.join(very_base_path, 'sweep_annealing_result_post.csv'))
    df_correlations.to_csv(os.path.join(very_base_path, 'sweep_correlations_post.csv'))

