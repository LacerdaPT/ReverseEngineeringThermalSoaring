
import os
import numpy as np
import pandas as pd
from misc.config import science_matplotlib_config

save = False
figsize_multiplier = 2
science_matplotlib_config(figsize_multiplier=figsize_multiplier, save=save)
from matplotlib import pyplot as plt, gridspec

from plotting.sim_annealing import plot_wing_loading_sweep_var, plot_wing_loading_sweep_var2d




sigma_turbulence = np.sqrt(0.37331674 ** 2 + 0.43644384 ** 2 + 0.3794553 ** 2)
sweeps_vars_dict = {f'constant_wind': 'wind',
                    f'random_walk': 'wind_avg',
                    'rotation': 'rot_int',# ['rot_int','rot_radius'],
                    'turbulence_noise': ['turbulence','noise_level'],
}
dict_labels = {'wind': '$v_\\mathrm{wind}\\ \\ (\\mathrm{m~s}^{-1})$',
              'wind_avg': '$<v_\\mathrm{wind}>\\ \\ (\\mathrm{m~s}^{-1})$',
               'rot_int': '$v_\\mathrm{max}\\ \\ (\\mathrm{m~s}^{-1})$',
              'turbulence': '$\sigma_\\mathrm{Turb}\\ \\ (\\mathrm{m~s}^{-1})$',
              'noise_level': '$\sigma_\\mathrm{Noise}\\ \\ (\\mathrm{m})$'}

root_path = '/home/pedro/PycharmProjects/ThermalModelling'
path_to_save = os.path.join(root_path, 'results/SA/sweeps')
zoomed = False
do_loss = True
do_twin = False
share_axes =True

if save:
    os.makedirs(os.path.join(path_to_save, 'png'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, 'svg'), exist_ok=True)
if do_loss:
    if not do_twin:
        fig = plt.figure(layout='constrained', figsize=(figsize_multiplier * 4.75, figsize_multiplier * 1.25*4.75))
        gs = gridspec.GridSpec(3, 2, figure=fig, wspace=0.5, hspace=0.5, width_ratios=(1,1.2))
        gs_dict = {'constant_wind': gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], wspace=0.1, hspace=0.06 if not share_axes else 0.01),
                   'random_walk':   gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], wspace=0.1, hspace=0.06 if not share_axes else 0.01),
                   'rotation':      gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2, 0], wspace=0.1, hspace=0.06 if not share_axes else 0.01),
                   'turbulence_noise':    gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 1], wspace=0.1, hspace=0.05), }
        ax_dict = {k: [fig.add_subplot(v[0]), fig.add_subplot(v[1])] for k,v in gs_dict.items()}
        if share_axes:
            [a1.sharex(a2) for k, (a1, a2) in ax_dict.items() if k != 'turbulence_noise']
    else:
        fig, ax_arr = plt.subplots(32,2, layout='constrained', figsize=(7, 7))
        ax_dict = {
            'constant_wind':    [ax_arr[0, 0]],
            'random_walk':      [ax_arr[1, 0]],
            'rotation':      [ax_arr[2, 0]],
            'turbulence_noise': [ax_arr[0, 1], ax_arr[1, 1]],
        }
        for k in ax_dict.items():
            if k != 'turbulence_noise':
                ax_dict[k].append(ax_dict[k][0].twinx())
else:
        fig, ax_dict = plt.subplot_mosaic([['constant_wind', 'turbulence_noise'],
                                           ['random_walk', 'turbulence_noise'],
                                           ['rotation', 'turbulence_noise']], layout='constrained',
                                          figsize=(7, 4))





for i_stat, (sweep_type, current_ax_arr) in enumerate(ax_dict.items()) :

    sweep_var = sweeps_vars_dict[sweep_type]
    base_path = os.path.join('synthetic_data/from_atlasz/newdata/', sweep_type)
    base_path = os.path.join(root_path, base_path)

    if not os.path.exists(os.path.join(base_path, 'results', 'sweep_annealing_result_post.csv')):
        continue
    df_results = pd.read_csv(os.path.join(base_path, 'results', 'sweep_annealing_result_post.csv'), index_col=0)

    if 'outlier' in df_results.columns:
        df_results = df_results[~df_results['outlier']]
    if sweep_type == 'turbulence_noise':
        df_results = df_results.sort_values(['noise_level', 'turbulence', 'realization'])
        df_results['turbulence'] *= sigma_turbulence
        df_results['noise_level'] *= np.sqrt(3)
        cbar = plot_wing_loading_sweep_var2d(current_ax_arr, df_results, sweep_var)
        if do_loss:
            current_ax_arr[0].set_xlabel(dict_labels[sweep_var[0]])
            current_ax_arr[0].set_ylabel(dict_labels[sweep_var[1]])
            current_ax_arr[1].set_xlabel(dict_labels[sweep_var[0]])
            current_ax_arr[1].set_ylabel(dict_labels[sweep_var[1]])
            cbar[0].set_label('$\Delta W^L ~~(\\mathrm{kg~m} ^{-2})$')
            cbar[1].set_label('$\\mathrm{Loss}~~(\\mathrm{m~s}^{-1})$')
            # current_ax_arr[0].set_title(sweep_type)
        else:
            current_ax_arr.set_xlabel(dict_labels[sweep_var[0]])
            current_ax_arr.set_ylabel(dict_labels[sweep_var[1]])
            cbar[0].set_label('$\Delta W^L\\ \\ (\\mathrm{kg m} ^{-2})$')
            current_ax_arr.set_title(sweep_type)

    else:
        plot_wing_loading_sweep_var(current_ax_arr, df_results, sweep_var)
        if do_loss:

            current_ax_arr[1].set_xlabel(dict_labels[sweep_var])
            current_ax_arr[0].set_ylabel('$W^L\\ \\ (\\mathrm{kg~m} ^{-2})$')
            current_ax_arr[1].set_ylabel('$\\mathrm{Loss}~~(\\mathrm{m~s}^{-1})$')
            if share_axes:

                current_ax_arr[0].tick_params(labelbottom=False)
                # current_ax_arr[0].set_xticklabels(current_ax_arr[0].get_xticks(), [])
                current_ax_arr[0].set_xlabel('')
            else:
                current_ax_arr[0].set_xlabel(dict_labels[sweep_var])

            if not zoomed:
                current_ax_arr[0].set_ylim((0, 6.5))
                current_ax_arr[1].set_ylim((0, None))
        else:
            current_ax_arr.set_xlabel(dict_labels[sweep_var])
            current_ax_arr.set_ylabel('$W^L\\ \\ (\\mathrm{kg~m} ^{-2})$')
            if not zoomed:
                current_ax_arr.set_ylim((0, 6.5))
        # if do_loss:
        #     current_ax_arr[0].set_title(sweep_type)
        # else:
        #     current_ax_arr.set_title(sweep_type)

if save:
    plt.draw()
    if zoomed:
        fig.savefig(os.path.join(path_to_save, 'png', f'SA_sweep_air_zoomed.png'),transparent=True, dpi=300)
        fig.savefig(os.path.join(path_to_save, 'svg', f'SA_sweep_air_zoomed.svg'))
    else:
        fig.savefig(os.path.join(path_to_save, 'png', f'SA_sweep_air.png'), transparent=True, dpi=300)
        fig.savefig(os.path.join(path_to_save, 'svg', f'SA_sweep_air.svg'))
    plt.close(fig)
else:
    plt.show(block=True)

