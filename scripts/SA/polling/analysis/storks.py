import os.path
from functools import reduce
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, PermutationMethod


save = True
n_resamples = 9999
root_path = '/home/pedro/ThermalModelling'
path_to_modes = os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/storks/config/individual_bins/bin_z_size=10/n_resamples=1000')
path_to_save = os.path.join(root_path, 'results/wing_loadings/pooling', 'storks')

df_modes = pd.read_csv(os.path.join(path_to_modes, 'modes.csv'),
                       index_col=False)

if save:
    os.makedirs(path_to_save, exist_ok=True)

df_modes['thermal'].unique()

df_modes['thermal'] = df_modes['thermal'].replace('b010', '0')
df_modes['thermal'] = df_modes['thermal'].replace('b0230', '1')
df_modes['thermal'] = df_modes['thermal'].replace('b0230', '1.1')
df_modes['thermal'] = df_modes['thermal'].replace('b072', '2')
df_modes['thermal'] = df_modes['thermal'].replace('b077', '3')
df_modes['thermal'] = df_modes['thermal'].replace('b112', '4')
df_modes['thermal'] = df_modes['thermal'].replace('b121', '5')
df_modes['thermal'] = df_modes['thermal'].replace('b010_b023_b072_10', '012')
df_modes['thermal'] = df_modes['thermal'].replace('b077_b112_b121_10', '345')



my_thermals = ['0', '1', '2', '3', '4', '5']

my_dfs = {'singles': df_modes[df_modes['thermal'].astype(str).isin(my_thermals)],
          'triples': df_modes[df_modes['thermal'].astype(str).isin(['012', '345'])],
          'all_avg': df_modes.copy(), }
dict_rms = {k: [] for k in my_dfs.keys()}
dict_rmedians = {k: [] for k in my_dfs.keys()}
df_avg = pd.DataFrame()
my_dfs_avg ={}
for k, df in my_dfs.items():
    my_dfs_avg[k] = df.groupby(['bird_name', 'loss_percentile',
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    my_dfs_avg[k].rename(columns={0: 'WL_mode_avg'}, inplace=True)
    my_dfs_avg[k]['method'] = k
    df_avg = pd.concat([df_avg, my_dfs_avg[k]])


final_df = df_avg[(df_avg['loss_percentile'] == 2) & (df_avg['method'] == 'singles')]
if save:
    final_df.to_csv(os.path.join(path_to_save, 'best_set_of_wing_loadings.csv'), index=False)
    df_avg.to_csv(os.path.join(path_to_save, 'wingloadings_different_poolings.csv'), index=False)

df_pearson = pd.DataFrame(columns=['loss_percentile', 'c1','c2','pearson_r','p_value', 'n'])

for i_c, c in enumerate(combinations(my_thermals, 3)):

    print(c)
    c = sorted(c)
    c_not = [i for i in my_thermals if i not in c]
    c_not = sorted(c_not)
    c_str = reduce(lambda a,b: a+b, c, '')
    c_not_str = reduce(lambda a,b: a+b, c_not,'')
    if not df_pearson[df_pearson['c2'] == c_str].empty:
        print('passing')
        continue
    df_1 = df_modes[df_modes['thermal'].astype(str).isin(c)].groupby(['bird_name', 'loss_percentile',
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    df_1.rename(columns={0: 'WL_mode_avg'}, inplace=True)
    df_2 = df_modes[df_modes['thermal'].astype(str).isin(c_not)].groupby(['bird_name', 'loss_percentile',
                                                ]).apply(lambda row: np.average(row['WL_mode'],
                                                                        weights=row['WL_mode_std']),
                                                 include_groups=False
                                                 ).reset_index()
    df_2.rename(columns={0: 'WL_mode_avg'}, inplace=True)
    df_merge = pd.merge(df_1, df_2, on=['bird_name', 'loss_percentile'],
                          suffixes=('_1', '_2'))
    for lp in df_merge['loss_percentile'].unique():
        current_merge = df_merge[df_merge['loss_percentile'] == lp]


        p_result = pearsonr(current_merge['WL_mode_avg_1'].values, current_merge['WL_mode_avg_2'].values,
                            method=PermutationMethod(n_resamples=n_resamples, batch=n_resamples)
                            )
        df_pearson = pd.concat([df_pearson,
                                pd.DataFrame.from_records([[lp,
                                                            c_str,
                                                            c_not_str,
                                                            p_result.statistic,
                                                            p_result.pvalue,
                                                            current_merge['WL_mode_avg_1'].size]],
                                                          columns=df_pearson.columns)]
                               )

df_pearson = df_pearson.reset_index(drop=True)
if save:
    df_pearson.to_csv(os.path.join(path_to_save, 'all_3-combinations.csv'), index=False)

