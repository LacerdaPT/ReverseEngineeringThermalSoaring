import os.path
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, to_rgb, rgb_to_hsv, hsv_to_rgb, to_hex
from scipy.stats import ttest_1samp

from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame
from calc.stats import get_all_permutated_rms
from data.get_data import load_decomposition_data
from misc.config import science_matplotlib_config, stork_dataset_renaming_dict
from misc.constants import root_path

import scienceplots
import matplotlib

save = True

figsize_multiplier = 2
science_matplotlib_config(figsize_multiplier, save=save)
base_path = 'synthetic_data/from_atlasz/newdata/storks'
save_folder = os.path.join(root_path, 'results/air_velocity_field/storks/rotation')
pp=Path(os.path.join(root_path, base_path))

if save:
    os.makedirs(os.path.join(save_folder, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'svg'), exist_ok=True)



#ff0011
#ff6670
color_list = [['#ff0011', '#ff0051'],
              ['#6e2673','#b740bf'],
              ['#2b7326','#48bf40'],
              ['#005799', '#0091ff'],
              ['#ff6f00', '#ffae00'],
    ]
list_of_thermals = sorted(stork_dataset_renaming_dict.keys())
is_synthetic = pp.name != 'storks'
fig, ax = plt.subplots(2,3, sharex=True, sharey=True, layout='constrained', figsize=(figsize_multiplier * 4.75,figsize_multiplier *  3))
ax = ax.flatten()

path_wildcard = 'decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'
list_of_thermals = [e for e in list_of_thermals if e != 'b023_1.1']
list_of_bad = []
for i_ss, current_thermal in enumerate( list_of_thermals):
    path_to_decomposition = os.path.join(base_path, current_thermal, path_wildcard)
    df_rotation = pd.read_csv(os.path.join(path_to_decomposition, 'rotation_binned.csv'), index_col=False)


    df_rotation = df_rotation[df_rotation['rho_bird_TC_count'] >= 10]
    df_rotation = df_rotation[df_rotation['bin_index_rotation'].between(10,28)]
    (l,) = ax[i_ss].plot(df_rotation['rho_bird_TC_mean'],
             df_rotation['V_phi_rotating_thermal_ground_mean'], )
    ax[i_ss].fill_between(df_rotation['rho_bird_TC_mean'],
                     df_rotation['V_phi_rotating_thermal_ground_mean'] + df_rotation['V_phi_rotating_thermal_ground_sem'],
                     df_rotation['V_phi_rotating_thermal_ground_mean'] - df_rotation['V_phi_rotating_thermal_ground_sem'], alpha=0.3, fc=l.get_color())
    ax[i_ss].axhline(y=0,c='k', ls='-.')
    ax[i_ss].set_title(stork_dataset_renaming_dict[current_thermal])
    # ax[i_ss].legend()

[a.set_xlabel('$\\rho ~(\\mathrm{m})$') for a in ax[3:]]
[a.set_ylabel('$v_{\\phi} ~ (\\mathrm{m~s}' + '^{-1})$') for a in ax[::3]]
ax[i_ss].set_ylim((-1,1))



if save:
    fig.savefig(os.path.join(save_folder, 'png', f'rotation_binned.png'),transparent=True)
    fig.savefig(os.path.join(save_folder, 'svg', f'rotation_binned.svg'))


