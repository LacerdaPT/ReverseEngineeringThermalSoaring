import os.path
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

pd.options.mode.chained_assignment = None
root_path = '/home/pedro/PycharmProjects/ThermalModelling'
air_component = 'air'
save=True
selected_radius = 10
min_occupation = 5
path_to_csvs = os.path.join(root_path, f'results/turbulence/turbulence_noise/small_scale', air_component)


save_folder = os.path.join(path_to_csvs, f'min-occ_{min_occupation}_radius_{selected_radius}')

if save:
      os.makedirs(save_folder, exist_ok=True)

df_all_fluctuations = pd.read_csv(f'{path_to_csvs}/fluctuations.csv', index_col=False)
df_all_spheres = pd.read_csv(f'{path_to_csvs}/stats_per_radius_per_sphere.csv', index_col=False)
df_all_spheres.rename(columns={'count': 'n'}, inplace=True)

df_all_fluctuations = df_all_fluctuations[df_all_fluctuations['radius'] == selected_radius]
df_all_fluctuations = df_all_fluctuations[df_all_fluctuations['n'] >= min_occupation]

df_all_fluctuations['turbulence'] *= 0.632
df_all_fluctuations['noise']  *=  np.sqrt(3)

df_all_spheres = df_all_spheres[df_all_spheres['radius'] == selected_radius]
df_all_spheres = df_all_spheres[df_all_spheres['n'] >= min_occupation]

df_all_spheres['turbulence'] *= 0.632
df_all_spheres['noise']  *=  np.sqrt(3)


df_all_sigmas = df_all_fluctuations.groupby(['noise', 'turbulence', 'realization', 'datatype', 'radius']).std().reset_index()


df_all_sigma_grouped = df_all_sigmas.groupby(['noise', 'turbulence', 'radius', 'datatype']
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


if save:
      df_all_sigmas.to_csv(os.path.join(save_folder,f'turbulence_sigmas.csv'), index=False)
      df_all_fluctuations.to_csv(os.path.join(save_folder,f'fluctuations.csv'), index=False)
      df_all_spheres.to_csv(os.path.join(save_folder,f'stats_per_radius_per_sphere.csv'), index=False)
      df_all_sigma_grouped.to_csv(os.path.join(save_folder,f'turbulence_sigmas_grouped.csv'), index=False)


list_of_radii = df_all_sigma_grouped['radius'].unique()
list_of_noise = df_all_sigma_grouped['noise'].unique()
result_list = []

for i_radius, current_radius in enumerate(list_of_radii[:]):
    for n in list_of_noise:
        for coord in ['x', 'y', 'z', 'xyz']:
            current_noise_df = df_all_sigma_grouped[df_all_sigma_grouped['noise'] == n]
            current_noise_merge = pd.merge(current_noise_df[current_noise_df['datatype'] == 'dec'],
                                           current_noise_df[current_noise_df['datatype'] == 'gt'],
                                           on=['turbulence'], suffixes=('_dec','_gt'))
            popt, pcov, info, _, _ = curve_fit(lambda x, a,b: a* x + b,
                                               current_noise_merge[f'{coord}_median_gt'],
                                               current_noise_merge[f'{coord}_median_dec'],
                                               p0=(1,0),
                                               sigma=current_noise_merge[f'{coord}_sem_dec'],
                                               absolute_sigma=True,
                                               full_output=True)
            # info['fvec'] =  (f(x, *popt) - ydata)/sigma
            current_chi2 = np.sum(info['fvec']**2) / (len(info['fvec']) - 2 )
            result_list.append([n,current_radius, coord, ] + popt.tolist() + np.sqrt(np.diag(pcov)).tolist() + [current_chi2])
result_df = pd.DataFrame(result_list,
     columns=['noise', 'radius','comp', 'a', 'b' ,'e_a', 'e_b', 'chi2']
           )
if save:
    result_df.to_csv(os.path.join(save_folder,f'dec_vs_gt_linear_fits.csv'), index=False)



result_list = []

for i_radius, current_radius in enumerate(list_of_radii):
    for n in list_of_noise:
        for coord in ['x', 'y', 'z', 'xyz']:
            current_noise_df = df_all_sigma_grouped[(df_all_sigma_grouped['noise'] == n)
                                                    & (df_all_sigma_grouped['datatype'] == 'dec')]

            popt, pcov, info, _, _ = curve_fit(lambda x, a,b: a* x + b,
                                               current_noise_df[f'turbulence'],
                                               current_noise_df[f'{coord}_mean'],
                                               p0=(1,0),
                                               sigma=current_noise_df[f'{coord}_sem'],
                                               absolute_sigma=True,
                                               full_output=True)
            # info['fvec'] =  (f(x, *popt) - ydata)/sigma
            current_chi2 = np.sum(info['fvec']**2) / (len(info['fvec']) - 2 )
            result_list.append([n,current_radius, coord, ] + popt.tolist() + np.sqrt(np.diag(pcov)).tolist() + [current_chi2])
result_df = pd.DataFrame(result_list,
     columns=['noise', 'radius','comp', 'a', 'b' ,'e_a', 'e_b', 'chi2']
           )
if save:
    result_df.to_csv(os.path.join(save_folder,f'dec_vs_turbulence_linear_fits.csv'), index=False)