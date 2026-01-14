import os.path
from copy import deepcopy
from itertools import product, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from calc.stats import get_all_permutated_rms
from misc.constants import root_path



save = True
with_average = True
p = Path(os.path.join(root_path, 'results/turbulence/turbulence_noise/mesoscale_new_centers/air'))
path_to_csv = str(p)

if with_average:
    turbulence_per_realization_filename = 'turbulence_correlation_grid_size=2_with_average_avg_per_realization.csv'
    turbulence_avg_filename = 'turbulence_correlation_grid_size=2_with_average_avg.csv'
    p_values_filename = 'p_values_turbulence_with_average.csv'
    p_values_noise_filename = 'p_values_noise_with_average.csv'
else:
    turbulence_per_realization_filename = 'turbulence_correlation_grid_size=2_avg_per_realization.csv'
    turbulence_avg_filename = 'turbulence_correlation_grid_size=2_avg.csv'
    p_values_filename = 'p_values_turbulence.csv'
    p_values_noise_filename = 'p_values_noise.csv'

df_all_correlations_avg_per_realization = pd.read_csv(os.path.join(path_to_csv, turbulence_per_realization_filename), index_col=False)


save_folder = path_to_csv
n_permutations = 1000

#df_all_correlations['turbulence'] *= 0.6314318
#df_all_correlations['noise'] *= np.sqrt(3)
df_all_correlations_avg_per_realization['turbulence'] *= 0.6314318
df_all_correlations_avg_per_realization['noise'] *= np.sqrt(3)

list_of_components = ['inner', 'x', 'y', 'z']

df_all_correlations_avg = df_all_correlations_avg_per_realization.groupby(['noise' ,'turbulence'] + ['datatype', 'size', 'delta_R']
                                                  ).agg(**{f'{col}_{stat}': (f'{col}_{stat}', stat)
                                                           for col in list_of_components
                                                           for stat in ['mean', 'median', 'std', 'count']
                                                           }).reset_index()

for i_comp, comp in enumerate(list_of_components):
    df_all_correlations_avg[f'{comp}_sem'] = df_all_correlations_avg[f'{comp}_std'].values / np.sqrt(df_all_correlations_avg[f'{comp}_count'].values)


df_all_correlations_avg.to_csv(os.path.join(path_to_csv, turbulence_avg_filename), index=False)



list_of_turbulence = df_all_correlations_avg['turbulence'].sort_values().unique()
list_of_noise = df_all_correlations_avg['noise'].sort_values().unique()
list_of_sizes = df_all_correlations_avg['size'].unique()
list_to_permute = list_of_noise


df_all_correlations_avg = df_all_correlations_avg[df_all_correlations_avg['delta_R'] >= 10]


dict_null_rms_noise = {}
dict_null_rms_turbulence = {}

df_corr_avg_no_turb = df_all_correlations_avg[df_all_correlations_avg['turbulence'] == 0.0]

p_value_list_no_turb = []
for i_comp, comp in enumerate(list_of_components):
    print(comp)
    for i_size, size in enumerate(list_of_sizes):
        df_correlations_avg_current_size = df_corr_avg_no_turb[df_corr_avg_no_turb['size']== size]
        list_of_diff, dict_null_rms_noise[comp] = get_all_permutated_rms(df_correlations_avg_current_size[df_correlations_avg_current_size['datatype'] == 'gt'],
                                                                         df_correlations_avg_current_size[df_correlations_avg_current_size['datatype'] == 'dec'],
                                                                          f'{comp}_mean', 'noise', list_to_permute=list_of_noise,
                                                                         n_permutations = n_permutations)

        current_p_value = (np.array(list_of_diff) < dict_null_rms_noise[comp]).sum() / len(list_of_diff)
        p_value_list_no_turb.append([comp, size, current_p_value, len(list_of_diff)])

if save:
    pd.DataFrame(p_value_list_no_turb, columns=['comp', 'size', 'current_p_value', 'n']
                 ).to_csv(os.path.join(save_folder, p_values_noise_filename), index=False)

df_corr_avg_no_noise = df_all_correlations_avg[df_all_correlations_avg['noise'] == 0.0]

p_value_list_no_noise = []
for i_comp, comp in enumerate(list_of_components):
    print(comp)
    for i_size, size in enumerate(list_of_sizes):
        df_correlations_avg_current_size = df_corr_avg_no_noise[df_corr_avg_no_noise['size']== size]
        list_of_diff, dict_null_rms_turbulence[comp] = get_all_permutated_rms(df_correlations_avg_current_size[df_correlations_avg_current_size['datatype'] == 'gt'],
                                                                              df_correlations_avg_current_size[df_correlations_avg_current_size['datatype'] == 'dec'],
                                               f'{comp}_mean', 'turbulence', list_of_turbulence, n_permutations
                                                                              )

        current_p_value = (np.array(list_of_diff) < dict_null_rms_turbulence[comp]).sum() / len(list_of_diff)
        p_value_list_no_noise.append([comp, size, current_p_value, len(list_of_diff)])

if save:
    pd.DataFrame(p_value_list_no_noise, columns=['comp', 'size', 'current_p_value', 'n']
                 ).to_csv(os.path.join(save_folder, p_values_filename), index=False)