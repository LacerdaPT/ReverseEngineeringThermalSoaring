import os
import numpy as np
import pandas as pd
import yaml

from plotting.sim_annealing import plot_kde_modes_histograms

save = True
figsize_multiplier = 2
from misc.config import science_matplotlib_config
science_matplotlib_config(figsize_multiplier, save=save)

from matplotlib import pyplot as plt

renaming_dict = {'b010':              '0',
              'b0230':             '1',
              'b0231':             '1.1',
              'b072':              '2',
              'b077':              '3',
              'b112':              '4',
              'b121':              '5',
              'b010_b023_b072_10': '012',
              'b077_b112_b121_10': '345', }
list_of_birds = ['Balou', 'Betty', 'Bubbel', 'Conchito', 'Cookie', 'Crisps', 'Ekky',
       'Ella', 'Fanny', 'Fifi', 'Flummy', 'Frank', 'Hannibal', 'Kim',
       'Kristall', 'Mirabell', 'Muffine', 'Niclas', 'Ohnezahn', 'Peaches',
       'Redrunner', 'Ronja', 'Snowy', 'Tobi']

#loss_percentile = 2
candidate = True
config_base = ('/home/pedro/PycharmProjects/ThermalModelling/synthetic_data/from_atlasz/newdata/'
               'storks/config/individual_bins/bin_z_size=10/n_resamples=1000')
save_folder = os.path.join('results/wing_loadings/single_thermals/storks', 'figures')
if save:

    os.makedirs(os.path.join(save_folder, 'png'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'svg'), exist_ok=True)
#save_folder = 'data_pedro/decomposition_ready/subthermals/config_extended/figures_scipy'

# config_file_dict = {(a.parent.name, a.parent.parent.name): os.path.join(config_root, a.parent.parent.name, a.parent.name, 'sa.yaml')
#                     for a in p.glob('0.9/b*/sa.yaml')}


df_modes_all_percentiles = pd.read_csv(os.path.join(config_base, 'modes.csv'), index_col=False)

df_modes_all_percentiles['thermal'] = df_modes_all_percentiles['thermal'].apply(lambda x: renaming_dict[x])

df_modes_avg_all_percentiles = pd.read_csv(os.path.join(config_base, 'modes_average.csv'), index_col=False)
df_modes_avg_all_percentiles = pd.read_csv('results/wing_loadings/pooling/storks/wingloadings_different_poolings.csv', index_col=False)
df_modes_avg_all_percentiles = df_modes_avg_all_percentiles[df_modes_avg_all_percentiles['method'] == 'singles']
df_corr_all_percentiles = pd.read_csv(os.path.join(config_base, 'pearson_correlations_all_percentiles.csv'), index_col=False)
df_corr_all_percentiles['thermal_1'] = df_corr_all_percentiles['thermal_1'].apply(lambda x: renaming_dict[x])
df_corr_all_percentiles['thermal_2'] = df_corr_all_percentiles['thermal_2'].apply(lambda x: renaming_dict[x])
list_of_SA = df_modes_all_percentiles.groupby(['config_file','thermal','thermal_subindex']).count().reset_index()

list_of_SA = list_of_SA[['config_file','thermal','thermal_subindex']]
list_of_thermals =  ['0', '1', '2', '3', '4', '5']
list_of_SA = list_of_SA[list_of_SA['thermal'].isin(list_of_thermals)]
#list_of_percentiles = df_modes_all_percentiles['loss_percentile'].unique()
list_of_percentiles = [2]
for loss_percentile in list_of_percentiles:

    current_mode_percentile = df_modes_all_percentiles[df_modes_all_percentiles['loss_percentile'] == loss_percentile].copy()
    current_mode_avg_percentile = df_modes_avg_all_percentiles[df_modes_avg_all_percentiles['loss_percentile'] == loss_percentile].copy()
    current_corr_percentile = df_corr_all_percentiles[df_corr_all_percentiles['loss_percentile'] == loss_percentile].copy()

    ################################################################################################
    ################################         HISTOGRAMS        #####################################
    ################################################################################################
    for i_thermal, (_,(config_file, current_thermal, subindex)) in enumerate(list_of_SA.iterrows()):
        with open(os.path.join(config_file), 'r') as f:
            sa_config = yaml.safe_load(f)

        if current_thermal.count('b') > 1:
            continue
        current_mode_thermal = current_mode_percentile[(current_mode_percentile['thermal'] == current_thermal)
                                                       & (current_mode_percentile['thermal_subindex'] == subindex)].copy()

        path_to_annealing = sa_config['run_parameters']['output_folder']
        is_individual = sa_config['search_parameters']['individual_search']

        df_history = pd.read_csv(os.path.join(path_to_annealing, 'annealing_history.csv'))

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

        n_rows = round(np.sqrt(len(list_of_birds)))
        n_cols = len(list_of_birds) // n_rows + 1


        fig_hist, ax_arr_hist = plt.subplots(n_rows, n_cols,
                                             layout='constrained', sharex='all', figsize=(figsize_multiplier * 7.25,
                                                                                    figsize_multiplier * 7.25))
        ax_arr_hist = ax_arr_hist.flatten()
        plot_kde_modes_histograms(ax_arr_hist, df_threshold, current_mode_thermal, list_of_birds=list_of_birds,
                                  kde_kwargs={'bw_method': 'silverman',
                                              #'weights': 1 / df_threshold['loss_candidate']
                                              })

        # fig_hist.suptitle(
        #     f'{current_thermal} - {loss_percentile / 100 * df_history[loss_col].size} /{df_history[loss_col].size}')
        if save:
            plt.draw()
            plt.draw_all()
            fig_hist.savefig(os.path.join(save_folder, 'png', f'bird_histograms_{current_thermal}_{loss_percentile}.png'))
            fig_hist.savefig(os.path.join(save_folder, 'svg', f'bird_histograms_{current_thermal}_{loss_percentile}.svg'))
            plt.close(fig_hist)
        else:
            plt.show(block=True)

    current_mode_avg_percentile = current_mode_avg_percentile.sort_values('WL_mode_avg')

    ################################################################################################
    ################################         RANKING           #####################################
    ################################################################################################
    current_ordered_list_of_birds = current_mode_avg_percentile.sort_values('WL_mode_avg')['bird_name'].values.tolist()


    fig_ranking, ax_ranking = plt.subplots(1, 1, layout='constrained', figsize=(figsize_multiplier * 4.75,
                                                                                figsize_multiplier * 2.75))
    current_mode_avg_percentile.loc[:, 'ranking'] = current_mode_avg_percentile['bird_name'].apply(
        lambda x: current_ordered_list_of_birds.index(x))
    current_mode_percentile.loc[:, 'ranking'] = current_mode_percentile['bird_name'].apply(lambda x: current_ordered_list_of_birds.index(x))
    for i_t, t in enumerate(list_of_thermals):
        if t.count('b') > 1:
            continue
        df_thermal = current_mode_percentile[current_mode_percentile['thermal'] == t]

        #ax_ranking.errorbar(df_thermal['ranking'], df_thermal['WL_mode'], df_thermal['WL_mode_std'], fmt='none', label=t, alpha=0.4)
        ax_ranking.scatter(df_thermal['ranking'], df_thermal['WL_mode'], label=t, s=20, c=f'C{i_t}')
    ax_ranking.scatter(current_mode_avg_percentile['ranking'], current_mode_avg_percentile['WL_mode_avg'], s=60, marker='d',c=f'C{i_t +1}',
                       label='pooled', zorder=10)
    ax_ranking.set_xticks(range(len(current_mode_avg_percentile)), current_mode_avg_percentile['bird_name'], rotation=90)
    ax_ranking.set_xticks([], minor=True)
    ax_ranking.set_ylabel('$W^{L*}$  (kg m$^{-2}$)')
    ax_ranking.legend(loc='lower right', ncol=2, frameon=True, columnspacing=0.8)

    if save:
        plt.draw()
        plt.draw_all()
        fig_ranking.savefig(os.path.join(save_folder, 'png', f'bird_wingloadings_{loss_percentile}.png'))
        fig_ranking.savefig(os.path.join(save_folder, 'svg', f'bird_wingloadings_{loss_percentile}.svg'))
        plt.close(fig_ranking)
    else:
        plt.show(block=True)

    ################################################################################################
    ################################      CORRELATIONS         #####################################
    ################################################################################################

    fig_corr, ax_corr = plt.subplots(3, 5, layout='constrained',
                                     figsize=(figsize_multiplier * 4.75, 3 / 5 * figsize_multiplier * 4.75),
                                     sharex='all', sharey='all')
    ax_corr = ax_corr.flatten()
    i_ax = 0

    for (i_t1, t1) in enumerate(list_of_thermals):
        for (i_t2, t2) in enumerate(list_of_thermals[i_t1+1:]):
            ax_corr = ax_corr.flatten()
            cut_off = np.percentile(current_mode_percentile[current_mode_percentile['thermal'].isin([t1,t2])]['WL_mode_std'].values, 100)

            df_t1 = current_mode_percentile[current_mode_percentile['thermal'] == t1]
            df_t2 = current_mode_percentile[current_mode_percentile['thermal'] == t2]
            df_merge = pd.merge(df_t1, df_t2, on='bird_name', how='inner')
            df_merge['is_good'] = True
            #df_merge.loc[:, 'is_good'] = (df_merge['WL_mode_std_x'] < cut_off) & (df_merge['WL_mode_std_y'] < cut_off)
            if df_t2.empty or df_t1.empty:
                continue
            current_ax = ax_corr[i_ax]
            current_ax.set_aspect('equal')

            if df_merge.empty:
                continue

            cc = current_corr_percentile[current_corr_percentile['thermal_1'].isin([t1,t2])
                                         & current_corr_percentile['thermal_2'].isin([t1,t2])
                                         ].to_dict(orient='records')[0]
            pearson_r = cc['pearson_r']
            p_value = cc['pvalue']
            yerr = df_merge.loc[df_merge['is_good'], 'WL_mode_std_x'] #.apply(lambda x: eval(x)[1] - eval(x)[0])
            xerr = df_merge.loc[df_merge['is_good'], 'WL_mode_std_y'] #.apply(lambda x: eval(x)[1] - eval(x)[0])

            current_ax.scatter(df_merge.loc[df_merge['is_good'], 'WL_mode_x'],
                               df_merge.loc[df_merge['is_good'], 'WL_mode_y'],
                                # yerr=yerr,
                                # xerr=xerr,
                               c='b'
                               )
            current_ax.plot([np.min(current_mode_percentile['WL_mode'].values),
                             np.max(current_mode_percentile['WL_mode'].values)],
                            [np.min(current_mode_percentile['WL_mode'].values),
                             np.max(current_mode_percentile['WL_mode'].values)],
                            'r--')
            current_ax.set_xlabel(f'$W^{{L}}_{t1}$  (kg m$^{{-2}}$)')
            current_ax.set_ylabel(f'$W^{{L}}_{t2}$  (kg m$^{{-2}}$)')
            current_ax.spines[['right', 'top']].set_visible(False)
            visible_ticks = {
                "top": False,
                "right": False
            }
            current_ax.tick_params(axis="x", which="both", **visible_ticks)
            current_ax.tick_params(axis="y", which="both", **visible_ticks)

            i_ax += 1


    if save:
        plt.draw()
        plt.draw_all()
        fig_corr.savefig(os.path.join(save_folder, 'png', f'WL_correlations_{loss_percentile}.png'))
        fig_corr.savefig(os.path.join(save_folder, 'svg', f'WL_correlations_{loss_percentile}.svg'))
        plt.close(fig_corr)
    else:
        plt.show(block=True)
