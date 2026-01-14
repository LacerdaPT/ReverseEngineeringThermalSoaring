import os.path
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd


turbulence_lookup_mpers = {0: 0.22,
                           1: 0.30,
                           2: 0.28,
                           3: 0.42,
                           4: 0.41,
                           5: 0.61, }
pd.options.mode.chained_assignment = None
root_path = '/home/pedro/PycharmProjects/ThermalModelling'
air_component = 'air'
path_to_csvs = os.path.join(root_path, f'results/turbulence/same_flock_WL=6.0/small_scale', air_component)

selected_radius = 10
min_occupation = 5

save_folder = os.path.join(path_to_csvs, f'min-occ_{min_occupation}_radius_{selected_radius}')
os.makedirs(save_folder, exist_ok=True)

df_all_fluctuations = pd.read_csv(f'{path_to_csvs}/fluctuations.csv', index_col=False)
df_all_spheres = pd.read_csv(f'{path_to_csvs}/stats_per_radius_per_sphere.csv', index_col=False)

df_all_spheres.rename(columns={'count': 'n'}, inplace=True)

df_all_fluctuations = df_all_fluctuations[df_all_fluctuations['radius'] == selected_radius]
df_all_fluctuations = df_all_fluctuations[df_all_fluctuations['n'] >= min_occupation]

df_all_spheres = df_all_spheres[df_all_spheres['radius'] == selected_radius]
df_all_spheres = df_all_spheres[df_all_spheres['n'] >= min_occupation]




df_all_fluctuations['turbulence'] = df_all_fluctuations['thermal'].apply(lambda x: turbulence_lookup_mpers[int(x)])
df_all_spheres['turbulence'] = df_all_spheres['thermal'].apply(lambda x: turbulence_lookup_mpers[int(x)])


df_all_sigma_grouped = df_all_fluctuations.groupby(['thermal', 'turbulence', 'radius', 'datatype']
                                             ).agg(x_mean=('x', 'mean'),
                                                   y_mean=('y', 'mean'),
                                                   z_mean=('z', 'mean'),
                                                   xyz_mean=('xyz', 'mean'),
                                                   x_std=('x', 'std'),
                                                   y_std=('y', 'std'),
                                                   z_std=('z', 'std'),
                                                   xyz_std=('xyz', 'std'),
                                                   x_sem=('x', 'sem'),
                                                   y_sem=('y', 'sem'),
                                                   z_sem=('z', 'sem'),
                                                   xyz_sem=('xyz', 'sem'),
                                                   x_median=('x', 'median'),
                                                   y_median=('y', 'median'),
                                                   z_median=('z', 'median'),
                                                   xyz_median=('xyz', 'median'),
                                                   x_count=('x', 'count'),
                                                   y_count=('y', 'count'),
                                                   z_count=('z', 'count'),
                                                   xyz_count=('xyz', 'count'),
                                                   ).reset_index()



df_all_fluctuations.to_csv(os.path.join(save_folder, f'fluctuations.csv'), index=False)
df_all_spheres.to_csv(os.path.join(save_folder, f'stats_per_radius_per_sphere.csv'), index=False)
df_all_sigma_grouped.to_csv(os.path.join(save_folder, f'turbulence_sigmas_grouped.csv'), index=False)