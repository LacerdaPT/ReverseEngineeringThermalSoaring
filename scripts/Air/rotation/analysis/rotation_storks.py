import os.path
from itertools import permutations, product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb, rgb_to_hsv, hsv_to_rgb, to_hex
from scipy.spatial import KDTree
from scipy.stats import ttest_1samp

from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame
from calc.stats import get_all_permutated_rms
from data.get_data import load_decomposition_data
from misc.constants import root_path

base_path = 'synthetic_data/from_atlasz/newdata/storks'
save_folder = os.path.join(root_path, 'results/air_velocity_field/storks/rotation')
pp=Path(os.path.join(root_path, base_path))


parameter_dict_lambda = lambda p: {'thermal': p.parents[6].name,
                                   'bin_z_size': p.parents[3].name.split('=')[1],}
is_synthetic = pp.name != 'storks'

# path_wildcard = f'rot_int=*/*/*/decomposition/average/0/final/reconstructed'
# for i in range(1):
path_wildcard = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'
list_of_ts = []
col_to_rms = 'V_phi_rotating_thermal_ground_mean'
df_concat = pd.DataFrame()
for delta_R in range(5,20, 5):
    for i_ss, ss in enumerate( pp.glob(path_wildcard)):
        path_to_decomposition = str(ss)
        df_rotation = pd.read_csv(os.path.join(path_to_decomposition, 'thermal.csv'), index_col=False)
        df_rotation['datatype'] = 'DEC'
        current_params = parameter_dict_lambda(ss)
        for k,v in current_params.items():
            df_rotation[k] = v
        thermal=current_params['thermal']

        my_grid = np.meshgrid(np.arange(-50,50 + 5,delta_R),
                              np.arange(-50,50 + 5,delta_R),
                              np.arange(df_rotation['Z'].min(),df_rotation['Z'].max() + 5, delta_R))

        my_grid = np.stack(my_grid, axis=-1)

        my_tree = KDTree(df_rotation[['X_bird_TC', 'Y_bird_TC', 'Z']].values)
        dd, ii = my_tree.query(my_grid)
        my_points = pd.DataFrame(my_tree.data[ii].reshape(-1,3), columns=['X_bird_TC', 'Y_bird_TC', 'Z'])
        my_points.drop_duplicates(inplace=True)
        df_rotation = pd.merge(df_rotation, my_points, on=['X_bird_TC', 'Y_bird_TC', 'Z'], how='inner')
        t_result = ttest_1samp(df_rotation['V_phi_rotating_thermal_ground'], popmean=0,)

        list_of_ts.append([thermal, delta_R, df_rotation['V_phi_rotating_thermal_ground'].mean(), df_rotation['V_phi_rotating_thermal_ground'].std(ddof=1), df_rotation['V_phi_rotating_thermal_ground'].sem(ddof=1), t_result.statistic,t_result.pvalue, t_result.df])


df_ttest = pd.DataFrame(list_of_ts,
                        columns=['thermal', 'delta_R','mean', 'std', 'sem', 't', 'pvalue','df']
                        ).sort_values(['thermal', 'delta_R'],
                                      )
df_ttest.to_csv(os.path.join(save_folder, 'ttest.csv'), index=False)