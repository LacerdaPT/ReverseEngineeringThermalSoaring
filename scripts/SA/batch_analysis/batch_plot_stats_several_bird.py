
import os
import numpy as np
import pandas as pd

from misc.config import science_matplotlib_config

save = True

figsize_multiplier = 2

science_matplotlib_config(figsize_multiplier=figsize_multiplier, save=save)
import matplotlib.pyplot as plt
from plotting.sim_annealing import plot_wing_loading_sweep_var
sweeps_vars_dict = {f'EER_sweep_avg': 'EER_avg',
                    f'EER_sweep_std': 'EER_std',
                    'sweep_offset_CD': 'CD',
                    'WL_sweep_avg_6_std_06': 'WL_avg',
                    'wing_loading_sweep_std': 'WL_std',
                    'sweep_offset_CL': 'CL',
}
#
# sweeps_vars_dict = {f'constant_wind': 'wind',
#                     f'random_walk': 'wind_avg',
#                     #'rotation': ['rotation_intensity','rotation_radius'],
#                     'turbulence_noise': ['turbulence','noise_level'],
#}
dict_labels = {'EER_avg': '$\mu_\mathrm{EER}$',
              'EER_std': '$\sigma_\mathrm{EER}$',
              'CD': '$C_D$',
              'WL_avg': '$ \mu_{W^L}\\ \\ (\\mathrm{kg~m}^{-2})$',
              'WL_std': '$\sigma_{W^L}\\ \\ (\\mathrm{kg~m}^{-2})$',
              'CL': '$C_L$'}

root_path = '/home/pedro/PycharmProjects/ThermalModelling'
path_to_save = os.path.join(root_path, 'results/SA/sweeps')
if save:
    os.makedirs(os.path.join(path_to_save, 'png'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, 'svg'), exist_ok=True)
zoomed = False
do_loss = True
do_twin = False
if do_loss:
    if not do_twin:
        print('askdjh')
        fig, axes = plt.subplots(len(sweeps_vars_dict),2,
                                  figsize=(figsize_multiplier * 4.75, figsize_multiplier * 1.25*4.75),
                                 layout='constrained',
                                # wspace=0.07,
                                # hspace=0.07
                                 )
        axes = axes.T.flatten()
        ax_dict = {sweep_type: axes[2 * i_sweep_type:2 * i_sweep_type + 2]
                   for i_sweep_type, sweep_type in enumerate(sweeps_vars_dict.keys())}
    else:
        fig, axes = plt.subplots(len(sweeps_vars_dict) // 2, 2, layout='constrained', figsize=(7, 7),
                                 # wspace=0.07,
                                 # hspace=0.07
                                 )
        axes = axes.T.flatten()
        ax_dict = {sweep_type: [axes[i_sweep_type], axes[i_sweep_type].twinx()]
                   for i_sweep_type, sweep_type in enumerate(sweeps_vars_dict.keys())}
else:
    fig, axes = plt.subplots(len(sweeps_vars_dict) // 2,2,layout='constrained', figsize=(7, 7),
                            # wspace=0.07,
                            # hspace=0.07
                             )
    axes = axes.T.flatten()
    ax_dict = {sweep_type: axes[i_sweep_type]
               for i_sweep_type, sweep_type in enumerate(sweeps_vars_dict.keys())}


# list_of_letters = 'ABCDEFGHIJKLMNOPQRSTUVXWYZ'

for i_stat, (sweep_type, ax_arr) in enumerate(ax_dict.items()) :

    sweep_var = sweeps_vars_dict[sweep_type]

    base_path = os.path.join('synthetic_data/from_atlasz/newdata/', sweep_type)
    base_path = os.path.join(root_path, base_path)
    if not os.path.exists(os.path.join(base_path, 'results', 'sweep_annealing_result_post.csv'),):
        continue
    df_results = pd.read_csv(os.path.join(base_path, 'results', 'sweep_annealing_result_post.csv'), index_col=0, comment='#')

    if 'outlier' in df_results.columns:
        df_results = df_results[~df_results['outlier']]
    df_results[sweep_var] = np.round(df_results['parameter_value'], 3)
    plot_wing_loading_sweep_var(ax_arr, df_results, sweep_var)
    if isinstance(ax_arr, (list, np.ndarray, tuple)):
        ax_arr[0].set_xlabel(dict_labels[sweep_var])
        ax_arr[1].set_xlabel(dict_labels[sweep_var])
        ax_arr[0].set_ylabel('$W^L\\ \\ (\\mathrm{kg~m}^{-2})$')
        ax_arr[1].set_ylabel('loss  $(\\mathrm{m~s}^{-1})$')
        if not zoomed:
            ax_arr[0].set_ylim((0, ax_arr[0].get_ylim()[-1] * 1.1))
            ax_arr[1].set_ylim((0, ax_arr[1].get_ylim()[-1] * 1.1))
    else:
        ax_arr.set_xlabel(dict_labels[sweep_var])
        ax_arr.set_ylabel('$W^L\\ \\ (\\mathrm{kg~m}^{-2})$')
        if not zoomed:
            ax_arr.set_ylim((0, ax_arr.get_ylim()[-1] * 1.1))

    #ax_arr[0].annotate(list_of_letters[2 * i_stat], (-0.05, 1.15), xycoords='axes fraction', weight="bold")
    #ax_arr[1].annotate(list_of_letters[2 * i_stat + 1], (-0.05, 1.15), xycoords='axes fraction', fontweight="bold" )   #subfigs[i_stat].suptitle(sweep_type)
#fig.tight_layout()


if save:
    plt.draw()
    if zoomed:
        fig.savefig(os.path.join(path_to_save, 'png', f'SA_sweep_bird_zoomed.png'),transparent=True, dpi=300)
        fig.savefig(os.path.join(path_to_save, 'svg', f'SA_sweep_bird_zoomed.svg'))
    else:
        fig.savefig(os.path.join(path_to_save, 'png', f'SA_sweep_bird.png'), transparent=True, dpi=300)
        fig.savefig(os.path.join(path_to_save, 'svg', f'SA_sweep_bird.svg'))
    plt.close(fig)
else:
    plt.show(block=True)