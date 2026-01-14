import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import PermutationMethod, ConstantInputWarning

from misc.config import stork_dataset_renaming_dict

root_path = '/home/pedro/PycharmProjects/ThermalModelling'
list_of_pearson = []
for air_component in ['air', 'thermal']:
    path_to_csvs = os.path.join(root_path, f'results/turbulence/same_flock_WL=6.0/small_scale', air_component)

    df_all_fluctuations = pd.read_csv(f'{path_to_csvs}/fluctuations.csv', index_col=False)

    df_all_fluctuations = df_all_fluctuations[df_all_fluctuations['n'] >= 5]
    df_all_sigma_grouped = df_all_fluctuations.groupby(['thermal', 'radius', 'datatype']
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


    df_all_sigma_grouped['thermal'] = df_all_sigma_grouped['thermal'].apply(lambda x: stork_dataset_renaming_dict[x])
    df_summary = pd.read_csv(f'results/air_velocity_field/same_flock_WL=6.0/thermals_summary.csv', index_col=False)
    df_summary = df_summary[df_summary['thermal'] != 'R1.1']




    df_summary = pd.merge(df_summary, df_all_sigma_grouped, on=['thermal', 'datatype'])
    my_cols = ['wind_norm_mean', 'wind_norm_std',
               'wind_angle_mean', 'wind_angle_std', 'thermal_mean_x', 'thermal_std_x',
               'thermal_mean_y', 'thermal_std_y', 'thermal_mean_z', 'thermal_std_z',
               'xyz_mean', 'x_std', 'y_std',
               'z_std', 'xyz_std',]
    for i_col1, col1 in enumerate(my_cols):
        for i_col2, col2 in enumerate(my_cols[i_col1+1:]):
            for s in np.sort(df_summary['radius'].unique()):

                pearson_result = stats.pearsonr(df_summary.loc[df_summary['radius'] == s, col1],
                           df_summary.loc[df_summary['radius'] == s, col2], method=PermutationMethod(100, batch=9999))
                list_of_pearson.append([air_component,col1,col2, s, pearson_result.statistic, pearson_result.pvalue, df_summary.loc[df_summary['radius'] == s, col2].size])



df_pearson = pd.DataFrame.from_records(list_of_pearson, columns=['air_component','col1', 'col2', 'radius', 'pearson_r', 'pvalue', 'n'])