import os.path
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import yaml

from calc.analysis.turbulence import get_local_fluctuations
from data.get_data import load_decomposition_data
from misc.constants import root_path

save = True

list_of_radii = [5, 10, 15,20] # 10, 20, 30,
base_path = Path(os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/same_flock_WL=6.0'))
glob_string = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'

air_component = 'air'
v_cols = [f'dXdT_{air_component}_ground', f'dYdT_{air_component}_ground', f'dZdT_{air_component}_ground']

save_folder = os.path.join(root_path, f'results/turbulence/same_flock_WL=6.0/small_scale/{air_component}')

if save:
    os.makedirs(save_folder, exist_ok=True)


sigmas_dict = []
#* 0.6314318
min_occupation = 3
df_fluctuations = pd.DataFrame()
df_stats = pd.DataFrame()
df_sigmas = pd.DataFrame()
#df_sigmas = pd.read_csv(os.path.join(save_folder, 'turbulence_sigmas.csv'), index_col=False)
#df_fluctuations = pd.read_csv(os.path.join(save_folder, 'fluctuations.csv'), index_col=False)
#df_stats = pd.read_csv(os.path.join(save_folder, 'stats_per_radius_per_sphere.csv'), index_col=False)
label_function = lambda pt: {'bin_z_size': pt.parents[3].name.split('=')[-1],
                             'thermal':    pt.parents[6].name}

for i_t, current_path in enumerate(base_path.glob(glob_string)):

    label_dict = label_function(current_path)
    label_keys = sorted(label_dict.keys())
    path_to_decomposition = str(current_path)
    if ((not df_sigmas.empty)
            and (not df_sigmas[np.all(df_sigmas[sorted(label_dict.keys())].astype(str) == [str(label_dict[k]) for k in sorted(label_dict.keys())],
                                                axis=1) & (df_sigmas['datatype'] == 'gt')].empty)):
        print('SKIPPING - ', str(path_to_decomposition))
        continue
    for datatype in ['dec', 'gt']:

        if datatype == 'gt':
            path_to_decomposition = os.path.join(path_to_decomposition, 'ground_truth_reconstructed')
        print(path_to_decomposition)

        df = pd.read_csv(os.path.join(path_to_decomposition, 'thermal.csv'), index_col=False)
        df = df[(~np.any(df[['interpolated_thermal_X', 'interpolated_thermal_Y', 'interpolated_thermal_Z']], axis=1))]
        if 'in_hull' in df.columns:
            df = df[df['in_hull']]
        my_data = df[
            ['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC'] + v_cols]

        (current_individual_stats,
         current_fluctuations) = get_local_fluctuations(my_data, list_of_radii=list_of_radii,
                                                        v_cols=v_cols,
                                                        min_occupation_number=min_occupation)
        current_individual_stats = pd.DataFrame(current_individual_stats,
                                                 columns=['radius', 'center_x', 'center_y', 'center_z',
                                                          'avg_x', 'avg_y', 'avg_z', 'avg_xyz',
                                                          'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xyz', 'count'])
        current_individual_stats['datatype'] = datatype
        for k,v in label_dict.items():
            current_individual_stats[k] = v
            current_fluctuations[k] = v
        current_individual_stats['datatype'] = datatype
        current_fluctuations['datatype'] = datatype
        df_stats = pd.concat([df_stats, current_individual_stats])
        df_fluctuations= pd.concat([df_fluctuations, current_fluctuations])

    df_sigmas = df_fluctuations.groupby(label_keys + ['datatype', 'radius']).std().reset_index()

    if save:
        df_fluctuations.to_csv(os.path.join(save_folder, f'fluctuations.csv'), index=False)
        df_stats.to_csv(os.path.join(save_folder, f'stats_per_radius_per_sphere.csv'), index=False)

        df_sigmas.to_csv(os.path.join(save_folder, f'turbulence_sigmas.csv'), index=False)

