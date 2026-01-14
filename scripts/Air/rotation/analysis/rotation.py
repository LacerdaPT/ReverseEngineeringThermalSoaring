import os.path
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
from misc.constants import root_path

# base_path = 'synthetic_data/from_atlasz/newdata/storks'
# save_folder = os.path.join(root_path, 'results/air_velocity_field/storks/rotation')
base_path = 'synthetic_data/from_atlasz/newdata/rotation'
save_folder = os.path.join(root_path, 'results/air_velocity_field/rotation/rotation')
pp=Path(os.path.join(root_path, base_path))

n_permutations = 1000

parameter_dict_lambda = lambda p: {'realization': p.parents[4].name,
                                   'parameter_name': p.parents[6].name.split('=')[0],
                                   'parameter_value': float(p.parents[6].name.split('=')[1])}
is_synthetic = pp.name != 'storks'
path_wildcard = f'rot_int=*/*/*/decomposition/average/0/final/reconstructed'
# for i in range(1):
#     path_wildcard = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'

df_concat = pd.DataFrame()
for i_ss, ss in enumerate( pp.glob(path_wildcard)):
    path_to_decomposition = str(ss)
    df_rotation = pd.read_csv(os.path.join(path_to_decomposition, 'rotation_binned.csv'), index_col=False)
    df_rotation_gt = pd.read_csv(os.path.join(path_to_decomposition, 'ground_truth_reconstructed', 'rotation_binned.csv'), index_col=False)
    df_rotation['datatype'] = 'DEC'
    df_rotation_gt['datatype'] = 'GT'
    for k,v in parameter_dict_lambda(ss).items():
        df_rotation[k] = v
        df_rotation_gt[k] = v
    df_concat = pd.concat([df_concat,df_rotation])
    df_concat = pd.concat([df_concat,df_rotation_gt])


df_dec = df_concat[df_concat['datatype'] == 'DEC']
df_gt = df_concat[df_concat['datatype'] == 'GT']
list_to_rotation_intensity = df_gt['parameter_value'].unique().tolist()
col_to_rms = 'V_phi_rotating_thermal_ground_mean'

dict_of_rms = {}
diff_of_null_rms = {}
for focal_rotation in list_to_rotation_intensity[:]:
    focal_dec = df_dec[df_dec['parameter_value'] == focal_rotation]
    focal_gt = df_gt[df_gt['parameter_value'] == focal_rotation]
    all_other_gt = df_gt[df_gt['parameter_value'] != focal_rotation]
    null_merge = pd.merge(focal_dec, focal_gt, on=['bin_index_rotation', 'realization'], suffixes=('_x', '_y'))
    null_diff = (null_merge[f'{col_to_rms}_x'] - null_merge[f'{col_to_rms}_y']).tolist()
    null_rms = np.array(null_diff)
    diff_of_null_rms[focal_rotation] = np.sqrt(np.nanmean(null_rms ** 2))
    dict_of_rms[focal_rotation] = []
    current_diff = []
    for p2, other_rotation in product(permutations(range(10)), all_other_gt['parameter_value'].unique()):
        if p2 == tuple(range(10)):
            continue
        for r1,r2 in  zip(range(10), p2):
            other_gt = all_other_gt[all_other_gt['parameter_value'] == other_rotation]
            current_focal = focal_dec[focal_dec['realization'] == str(r1)]
            current_other = other_gt[other_gt['realization'] == str(r2)]
            current_merge = pd.merge(current_focal, current_other, on='bin_index_rotation', suffixes=('_x', '_y'))
            current_diff += (current_merge[f'{col_to_rms}_x'] - current_merge[f'{col_to_rms}_y']).tolist()

        current_rms = np.array(current_diff)
        current_rms = np.sqrt(np.nanmean(current_rms ** 2))
        dict_of_rms[focal_rotation].append( current_rms)
        should_break = len(dict_of_rms[focal_rotation]) >= n_permutations
        if should_break:
            break


pd.DataFrame([[k, (np.array(current_list_of_rms) < diff_of_null_rms[k]).sum() / len(current_list_of_rms),  len(current_list_of_rms)]
              for k,current_list_of_rms in dict_of_rms.items()], columns=['rot_int', 'pvalue', 'n']
             ).sort_values(['rot_int']).to_csv(os.path.join(save_folder, f'permutation_test_{n_permutations}.csv'), index_label=False)

