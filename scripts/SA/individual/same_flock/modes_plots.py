import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from data.get_data import load_synthetic_data
from plotting.sim_annealing import plot_kde_modes_histograms

save = True
figsize_multiplier = 1
from misc.config import science_matplotlib_config
science_matplotlib_config(figsize_multiplier, save=save)
from matplotlib import pyplot as plt


list_of_birds = None
ordered_list_of_birds = None
list_of_percentiles = [2]

candidate=True
do_correlations=True
do_deviation=True
kernel_bandwidth=None
modes_folder = 'synthetic_data/from_atlasz/last/same_flock_WL=6.0/config/individual_bins/bin_z_size=10/permutation_n_resamples=1000'
save_folder = os.path.join('results/wing_loadings/single_thermals/same_flock_WL=6.0', 'figures')

p = Path(modes_folder)

if save:
    os.makedirs(os.path.join(save_folder, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'svg'), exist_ok=True)
synthetic_data_dict = load_synthetic_data('synthetic_data/from_atlasz/last/same_flock_WL=6.0/0')
bird_parameters = synthetic_data_dict['bird_parameters']
del synthetic_data_dict
bird_parameters = [[d['bird_name'], d['physical_parameters']['mass'] / d['physical_parameters']['wing_area']] for d in
                   bird_parameters]
if list_of_birds is None:
    list_of_birds = list(map(lambda x: x[0], bird_parameters))

df_bird_parameters = pd.DataFrame(bird_parameters, columns=['bird_name', 'WL_GT'])
ordered_list_of_birds = df_bird_parameters.sort_values('WL_GT')['bird_name'].values.tolist()
df_modes_all_percentiles = pd.read_csv(os.path.join(modes_folder, 'modes_all_percentiles.csv'), index_col=False)
df_modes_all_percentiles.rename(columns={'WL_real': 'WL_GT'}, inplace=True)
df_modes_avg_all_percentiles = pd.read_csv(os.path.join(modes_folder, 'modes_average_all_percentiles.csv'), index_col=False)
df_modes_avg_all_percentiles = pd.read_csv('results/wing_loadings/pooling/same_flock_WL=6.0/wingloadings_different_poolings.csv', index_col=False)
df_modes_avg_all_percentiles = df_modes_avg_all_percentiles[df_modes_avg_all_percentiles['method'] == 'singles']
df_corr_all_percentiles = pd.read_csv(os.path.join(modes_folder, 'pearson_correlations_all_percentiles.csv'), index_col=False)
df_corr_GT_all_percentiles = pd.read_csv(os.path.join(modes_folder, 'pearson_correlations_real_all_percentiles.csv'), index_col=False)

list_of_thermals = ['0', '1', '2', '3', '4', '5']
df_modes_all_percentiles = df_modes_all_percentiles[df_modes_all_percentiles['thermal'].isin(list_of_thermals)]
list_of_SA = df_modes_all_percentiles.groupby(['config_file','thermal']).count().reset_index()
list_of_SA = list_of_SA[['config_file','thermal']]

if list_of_percentiles is None:
    list_of_percentiles = df_modes_all_percentiles['loss_percentile'].unique()
for loss_percentile in list_of_percentiles:
    current_mode_percentile = df_modes_all_percentiles[df_modes_all_percentiles['loss_percentile'] == loss_percentile].copy()
    current_mode_avg_percentile = df_modes_avg_all_percentiles[df_modes_avg_all_percentiles['loss_percentile'] == loss_percentile].copy()
    current_corr_percentile = df_corr_all_percentiles[df_corr_all_percentiles['loss_percentile'] == loss_percentile].copy()
    current_corr_GT_percentile = df_corr_GT_all_percentiles[df_corr_GT_all_percentiles['loss_percentile'] == loss_percentile].copy()

    ################################################################################################
    ################################         HISTOGRAMS        #####################################
    ################################################################################################
    for i_thermal, (_,(config_file, current_thermal)) in enumerate(list_of_SA.iterrows()):
        with open(config_file, 'r') as f:
            sa_config = yaml.safe_load(f)

        path_to_annealing = sa_config['run_parameters']['output_folder']
        is_individual = sa_config['search_parameters']['individual_search']

        df_history = pd.read_csv(os.path.join(path_to_annealing, 'annealing_history.csv'))
        current_thermal_mode = current_mode_percentile[current_mode_percentile['thermal'] == current_thermal]
        list_of_col = df_history.columns[:-4]
        if candidate:
            candidate_cols = list_of_col[len(list_of_col) // 2:]
            df_history = df_history[candidate_cols]
        else:
            accepted_cols = list_of_col[:len(list_of_col) // 2]
            df_history = df_history[accepted_cols]
        parameter_cols = df_history.columns[:-1]
        loss_col = df_history.columns[-1]

        loss_threshold = np.percentile(df_history[loss_col], loss_percentile)
        df_threshold = df_history[df_history[loss_col] < loss_threshold]
        df_mean = df_threshold.T.mean(axis=1)
        df_mean.index = [k.replace('wing_loading_','').replace('_candidate', '') for k in df_mean.index]
        #list_of_birds = df_mean.index.values.tolist()
        n_rows = 5 # round(np.sqrt(len(list_of_birds)))
        n_cols = 5 # len(list_of_birds) // n_rows + 1


        fig_histograms, ax_arr_histograms = plt.subplots(n_rows, n_cols, sharex='all',
                                                         layout='constrained',
                                                         figsize=(figsize_multiplier * 4.75, figsize_multiplier * 4.75))


        plot_kde_modes_histograms(ax_arr_histograms.flatten(), df_threshold, current_thermal_mode, list_of_birds=list_of_birds,
                                  kde_kwargs={'bw_method': 'silverman'})

        for _, (_, current_row) in enumerate(current_thermal_mode.iterrows()):
            current_col = current_row['bird_name']
            i = list_of_birds.index(current_col)
            ax_arr_histograms.flatten()[i].axvline(x=current_row['WL_GT'], c='g', label='GT')

        # fig_histograms.suptitle(
        #     f'{current_thermal} - {loss_percentile / 100 * df_history[loss_col].size} /{df_history[loss_col].size}')
        #plt.show(block=True)
        [a.set_xlabel('$W^{L\\ast}$  (kg m$^{-2}$)') for a in ax_arr_histograms[-1, :]]
        ax_arr_histograms[-1, -1].legend()
        if save:
            plt.draw()
            plt.draw_all()
            fig_histograms.savefig(os.path.join(save_folder, 'png', f'bird_histograms_{current_thermal}_{loss_percentile}.png'))
            fig_histograms.savefig(os.path.join(save_folder, 'svg', f'bird_histograms_{current_thermal}_{loss_percentile}.svg'))
            plt.close(fig_histograms)
        else:
            plt.show(block=True)

    if ordered_list_of_birds is None:
        current_ordered_list_of_birds = current_mode_avg_percentile.sort_values('WL_mode_mean')['bird_name'].values.tolist()
    else:
        current_ordered_list_of_birds = copy.deepcopy(ordered_list_of_birds)



    ################################################################################################
    ################################         RANKING           #####################################
    ################################################################################################
    current_mode_percentile.loc[:, 'ranking'] = current_mode_percentile['bird_name'].apply(lambda x: current_ordered_list_of_birds.index(x))
    current_mode_avg_percentile.loc[:, 'ranking'] = current_mode_avg_percentile['bird_name'].apply(lambda x: current_ordered_list_of_birds.index(x))
    df_bird_parameters.loc[:, 'ranking'] = df_bird_parameters['bird_name'].apply(lambda x: current_ordered_list_of_birds.index(x))

    current_mode_avg_percentile.sort_values('ranking', inplace=True)

    fig_ranking, ax_ranking = plt.subplots(1, 1, layout='constrained', figsize=(figsize_multiplier * 4.75,
                                                                                figsize_multiplier * 2.75))
    # fig_ranking.suptitle(f'{loss_percentile=}')
    # Average
    # ax_ranking.errorbar(current_mode_avg_percentile['ranking'], current_mode_avg_percentile['WL_mode_mean'], current_mode_avg_percentile['WL_mode_sem'],
    #                    marker='s', markersize=4, alpha=0.5)
    # Per thermal
    list_of_thermal = current_mode_percentile['thermal'].unique()
    for i_t,t in enumerate(list_of_thermal):
        df_thermal = current_mode_percentile[current_mode_percentile['thermal'] == t]
        marker_style = 'o' if len(t) == 1  else 's'
        #ax_ranking.errorbar(df_thermal['ranking'], df_thermal['WL_mode'], df_thermal['WL_mode_std'], fmt='none', label=t, alpha=0.4)
        ax_ranking.scatter(df_thermal['ranking'], df_thermal['WL_mode'], label=t, zorder=10, s=20, c=f'C{i_t}', marker=marker_style, alpha=0.6)

    # GROUND TRUTH DATA

    ax_ranking.scatter(df_bird_parameters['ranking'], df_bird_parameters['WL_GT'], label='Ground Truth', zorder=5, s=60, c=f'C{i_t+1}', marker='X')
    ax_ranking.scatter(current_mode_avg_percentile['ranking'], current_mode_avg_percentile['WL_mode_avg'], s=60, marker='d',c=f'C{i_t+2}', zorder=10,
                       label='pooled')
    ax_ranking.set_xticks(range(len(current_ordered_list_of_birds)), current_ordered_list_of_birds, rotation=90)
    ax_ranking.set_xticks([], minor=True)
    ax_ranking.set_ylabel('$W^{L*}$  (kg m$^{-2}$)')
    ax_ranking.legend(loc='lower right', ncol=2, frameon=True, columnspacing=0.8)

    if save:
        plt.draw()
        plt.draw_all()
        fig_ranking.savefig(os.path.join(save_folder, 'png', f'bird_wingloadings_ranking_{loss_percentile}.png'))
        fig_ranking.savefig(os.path.join(save_folder, 'svg', f'bird_wingloadings_ranking_{loss_percentile}.svg'))
        plt.close(fig_ranking)
    else:
        plt.show(block=True)

    ################################################################################################
    ################################      CORRELATIONS         #####################################
    ################################################################################################
    n_plots = len(list_of_thermal) * (len(list_of_thermal) + 1) / 2 - len(list_of_thermal) # Arithmetic sum
    n_rows = 3
    n_cols = 5
    fig_corr, ax_corr = plt.subplots(n_rows, n_cols, layout='constrained',
                                     figsize=(figsize_multiplier * 4.75, 3 / 5 * figsize_multiplier * 4.75),

                                     sharex='all', sharey='all')
    # fig_corr.suptitle(f'{loss_percentile=}')
    previous_shape = ax_corr.shape
    ax_corr = ax_corr.flatten()
    i_ax = 0
    for (i_t1, t1) in enumerate(list_of_thermal):
        for (i_t2, t2) in enumerate(list_of_thermal[i_t1+1:]):

            df_t1 = current_mode_percentile[(current_mode_percentile['thermal'] == t1) ]
            df_t2 = current_mode_percentile[(current_mode_percentile['thermal'] == t2) ]
            df_merge = pd.merge(df_t1, df_t2, on='bird_name', how='inner')

            if df_t2.empty or df_t1.empty:
                continue
            current_ax = ax_corr[i_ax]
            current_ax.set_aspect('equal')

            if df_merge.empty:
                continue

            cc = current_corr_percentile[(current_corr_percentile['thermal_1'] == t1)
                                         & (current_corr_percentile['thermal_2'] == t2)].to_dict(orient='records')[0]

            current_ax.scatter(df_merge['WL_mode_x'],
                               df_merge['WL_mode_y'], c='b')


            #current_ax.set_xlabel("$WL \ \ (kg m^{{-2}})$")
            pearson_r = cc['pearson_r']
            p_value = cc['pvalue']
            n = df_merge['WL_mode_x'].size
            # current_ax.set_title(f'{t1} vs {t2}\n' + f'$r = {pearson_r: .3g},\\ p={p_value: .3g}$') #, \\ n={n}
            current_ax.set_xlabel(f'$W^{{L}}_{t1}$  (kg m$^{-2}$)')
            current_ax.set_ylabel(f'$W^{{L}}_{t2}$  (kg m$^{-2}$)')
            current_ax.spines[['right', 'top']].set_visible(False)
            visible_ticks = {
                "top": False,
                "right": False
            }
            current_ax.tick_params(axis="x", which="both", **visible_ticks)
            current_ax.tick_params(axis="y", which="both", **visible_ticks)

            i_ax += 1
    ax_corr = ax_corr.reshape(previous_shape)
    # [a.set_ylabel('$W^{L}$  (kg m$^{-2}$)') for a in ax_corr[:,0]]
    # [a.set_xlabel('$W^{L}$  (kg m$^{-2}$)') for a in ax_corr[-1,:]]
    [a.plot([np.min(current_mode_percentile['WL_mode'].values), np.max(current_mode_percentile['WL_mode'].values)],
            [np.min(current_mode_percentile['WL_mode'].values), np.max(current_mode_percentile['WL_mode'].values)],
            'r--') for a in ax_corr.flatten()]
    if save:
        plt.draw()
        plt.draw_all()
        fig_corr.savefig(os.path.join(save_folder, 'png', f'WL_correlations_{loss_percentile}.png'))
        fig_corr.savefig(os.path.join(save_folder, 'svg', f'WL_correlations_{loss_percentile}.svg'))
        plt.close(fig_corr)
    else:
        plt.show(block=True)

    fig_corr_GT, ax_corr_GT = plt.subplots(2,3 , layout='constrained',
                                     figsize=(figsize_multiplier * 4.75,figsize_multiplier *3), sharex='all', sharey='all')
    previous_shape = ax_corr_GT.shape
    ax_corr_GT = ax_corr_GT.flatten()
    i_ax = 0
    for (i_t1, t1) in enumerate(list_of_thermal):

        df_t1 = current_mode_percentile[(current_mode_percentile['thermal'] == t1)].copy()

        if df_t1.empty:
            continue
        current_ax = ax_corr_GT[i_ax]
        current_ax.set_aspect('equal')

        cc = current_corr_GT_percentile[(current_corr_GT_percentile['thermal'] == t1)].to_dict(orient='records')[0]
        pearson_r = cc['pearson_r']
        p_value = cc['pvalue']
        n = df_t1['WL_mode'].size


        current_ax.scatter(df_t1['WL_mode'],
                           df_t1['WL_GT'], c='b')
        current_ax.spines[['right', 'top']].set_visible(False)
        visible_ticks = {
            "top": False,
            "right": False
        }
        current_ax.tick_params(axis="x", which="both", **visible_ticks)
        current_ax.tick_params(axis="y", which="both", **visible_ticks)

        # current_ax.plot([np.min(df_t1[['WL_mode', 'WL_GT']].values),
        #                  np.max(df_t1[['WL_mode', 'WL_GT']].values)],
        #                 [np.min(df_t1[['WL_mode', 'WL_GT']].values),
        #                  np.max(df_t1[['WL_mode', 'WL_GT']].values)],
        #                 'r--')

        # current_ax.set_title(f'{t1}\n' + f'$r = {pearson_r: .3g},\\ p={p_value: .3g}$') # , \\ n={n}
        current_ax.set_xlabel(f"$W^L_{{{t1}}}$  (kg m$^{{-2}}$)")
        i_ax += 1

    ax_corr_GT.reshape(previous_shape)[0,0].set_ylabel("$ W^L_{GT}$  (kg m$^{-2}$)")
    ax_corr_GT.reshape(previous_shape)[1,0].set_ylabel("$ W^L_{GT}$  (kg m$^{-2}$)")


    [a.plot([np.min(current_mode_percentile[['WL_mode', 'WL_GT']].values),
                     np.max(current_mode_percentile[['WL_mode', 'WL_GT']].values)],
                    [np.min(current_mode_percentile[['WL_mode', 'WL_GT']].values),
                     np.max(current_mode_percentile[['WL_mode', 'WL_GT']].values)],
                    'r--') for a in ax_corr_GT.flatten()]
    if save:
        plt.draw()
        plt.draw_all()
        fig_corr_GT.savefig(os.path.join(save_folder, 'png', f'WL_correlations_to_GT_{loss_percentile}.png'))
        fig_corr_GT.savefig(os.path.join(save_folder, 'svg', f'WL_correlations_to_GT_{loss_percentile}.svg'))
        plt.close(fig_corr_GT)
    else:
        plt.show(block=True)

    fig_violin, ax_violin = plt.subplots(2,1, layout='constrained', sharex='all',
                                         figsize=(figsize_multiplier * 2.25,figsize_multiplier * 2.25))
    fig_violin.suptitle(f'{loss_percentile=}')
    for (i_t1, t1) in enumerate(list_of_thermal):
        current_thermal_mode = current_mode_percentile[(current_mode_percentile['thermal'] == t1)]
        if current_thermal_mode.empty:
            continue
        ax_violin[0].violinplot(current_thermal_mode['delta_WL'], [i_t1], showmeans=True,
                                #showmedians=True
                                )
        ax_violin[1].violinplot(np.abs(current_thermal_mode['delta_WL']), [i_t1], showmeans=True,
                                #showmedians=True
                                )

    ax_violin[0].set_xlabel('thermal')
    ax_violin[0].set_ylabel('$\\Delta W^L$  (kg m$^{{-2}}$)') # \\Delta = W^L_{{mode}} - W^L_{{GT}}
    ax_violin[1].set_xlabel('thermal')
    ax_violin[1].set_ylabel('$|\\Delta W^L|$  (kg m$^{{-2}}$)')
    ax_violin[0].grid()
    ax_violin[1].grid()
    ax_violin[0].set_xticks(np.arange(len(list_of_thermal)), labels=list_of_thermal)
    ax_violin[1].set_xticks(np.arange(len(list_of_thermal)), labels=list_of_thermal)

    if save:
        plt.draw()
        plt.draw_all()
        fig_violin.savefig(os.path.join(save_folder,'png', f'delta_WL_violin_{loss_percentile}.png'))
        fig_violin.savefig(os.path.join(save_folder,'svg', f'delta_WL_violin_{loss_percentile}.svg'))
        plt.close(fig_violin)
    else:
        plt.show(block=True)