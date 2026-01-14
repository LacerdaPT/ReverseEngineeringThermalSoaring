import os.path
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgb, rgb_to_hsv
from scipy.stats import ttest_1samp

from misc.config import science_matplotlib_config
from misc.constants import root_path


save = False
figsize_multiplier = 2
science_matplotlib_config(figsize_multiplier, save=save)
#
# base_path = 'synthetic_data/from_atlasz/newdata/storks'
# save_folder = os.path.join(root_path, 'results/air_velocity_field/storks/rotation')
base_path = 'synthetic_data/from_atlasz/newdata/rotation'
save_folder = os.path.join(root_path, 'results/air_velocity_field/rotation/rotation')
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

is_synthetic = pp.name != 'storks'
fig, ax = plt.subplots(2,5, sharex=True, sharey=True, layout='constrained', figsize=(figsize_multiplier * 7.25,figsize_multiplier *  3))
ax = ax.flatten()
for i in range(5):
    path_wildcard = f'rot_int={i}.0/*/*/decomposition/average/0/final/reconstructed'
# for i in range(1):
#     path_wildcard = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'

    list_of_bad = []
    for i_ss, ss in enumerate( pp.glob(path_wildcard)):
        path_to_decomposition = str(ss)
        df_rotation = pd.read_csv(os.path.join(path_to_decomposition, 'rotation_binned.csv'), index_col=False)


        df_rotation = df_rotation[df_rotation['rho_bird_TC_count'] >= 10]
        df_rotation = df_rotation[df_rotation['rho_bird_TC_mean'] >= 10]
        tt = ttest_1samp(df_rotation['V_phi_rotating_thermal_ground_mean'], popmean=0, alternative='two-sided', nan_policy='omit')
        (l,) = ax[i_ss].plot(df_rotation['rho_bird_TC_mean'],
                 df_rotation['V_phi_rotating_thermal_ground_mean'],c=color_list[i][0], label=f'$t={tt.statistic:.2g},~P={tt.pvalue:.3g}$')
        ax[i_ss].fill_between(df_rotation['rho_bird_TC_mean'],
                         df_rotation['V_phi_rotating_thermal_ground_mean'] + df_rotation['V_phi_rotating_thermal_ground_sem'],
                         df_rotation['V_phi_rotating_thermal_ground_mean'] - df_rotation['V_phi_rotating_thermal_ground_sem'], alpha=0.3, fc=l.get_color())
        # ax[i_ss].axhline(y=0,c='k', ls='-.')
        # ax[i_ss].legend()
        if is_synthetic:
            hsv_gt = rgb_to_hsv(to_rgb(l.get_color())) # GT = Decomposed
            # print(hsv_gt)
            hsv_gt[1] = 1 # hsv_gt[1] / 1.5 # saturation
            hsv_gt[2] = hsv_gt[2] - 0.3 # value

            df_rotation_GT = pd.read_csv(os.path.join(path_to_decomposition,'ground_truth_reconstructed', 'rotation_binned.csv'), index_col=False)
            df_rotation_GT = df_rotation_GT[df_rotation_GT['rho_bird_TC_count'] >= 10]
            df_rotation_GT = df_rotation_GT[df_rotation_GT['bin_index_rotation'].between(df_rotation['bin_index_rotation'].min(),
                                                                                       df_rotation['bin_index_rotation'].max(),)]
            (l_gt, ) = ax[i_ss].plot(df_rotation_GT['rho_bird_TC_mean'],
                     df_rotation_GT['V_phi_rotating_thermal_ground_mean'],
                                     alpha=0.5,
                          c='k' , ls='--' #hsv_to_rgb(hsv_gt)
                          )
            ax[i_ss].fill_between(df_rotation_GT['rho_bird_TC_mean'],
                             df_rotation_GT['V_phi_rotating_thermal_ground_mean'] + df_rotation_GT['V_phi_rotating_thermal_ground_sem'],
                             df_rotation_GT['V_phi_rotating_thermal_ground_mean'] - df_rotation_GT['V_phi_rotating_thermal_ground_sem'], alpha=0.3,
                                  fc=color_list[i][1] #$l_gt.get_color(),# hatch='/'
                                  )
[a.set_xlabel('$\\rho~~(\\mathrm{m})$') for a in ax[5:]]
[a.set_ylabel('$v_{\\phi}~~(\\mathrm{m~s}^{-1})$') for a in ax[::5]]
ax[i_ss].set_xlim((10, 28))



if save:
    fig.savefig(os.path.join(save_folder, 'png', f'rotation_binned.png'),transparent=True)
    fig.savefig(os.path.join(save_folder, 'svg', f'rotation_binned.svg'))