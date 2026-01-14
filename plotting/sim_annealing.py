import os

import numpy as np
import matplotlib as mpl
import pandas as pd
import yaml
from matplotlib import pyplot as plt, cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
import logging

from scipy.stats import gaussian_kde

logging.getLogger('matplotlib').setLevel(logging.WARNING)


def plot_sim_annealing_scatter_hist_1d(ax_arr, parameters_array, loss, steps, label=None):
    guideline_color = 'g'
    padding = 0.02
    ax_scatter, ax_hist = ax_arr
    if label is None:
        label = 'x1'
    parameters_array = np.array(parameters_array)
    n_points = parameters_array.size
    loss = np.array(loss)

    index_min = np.argmin(loss)
    x_min = parameters_array[index_min]
    loss_min = loss[index_min]
    step_min = steps[index_min]

    sorting_indices = np.argsort(parameters_array)

    m = ax_scatter.scatter(parameters_array[sorting_indices],
                           loss[sorting_indices],
                           c=steps[sorting_indices],
                           alpha=0.6)
    cb = plt.colorbar(m, ax=ax_scatter, label='SA step', location='top')
    cb.solids.set(alpha=1)

    ax_scatter.set_xlabel(label)
    ax_scatter.set_ylabel('loss')

    m = ax_hist.hist(parameters_array,
                     bins=int(np.sqrt(n_points)))

    ax_hist.set_xlabel(label)
    ax_hist.set_ylabel('counts')

    # =============================================
    # =========       ANNOTATIONS      ============
    # =============================================
    # SCATTER
    x_lims = ax_scatter.get_xlim()
    y_lims = ax_scatter.get_ylim()
    ax_scatter.text(x_min,
                    y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                    f'{x_min:.4g}')

    ax_scatter.text(x_lims[1] + padding * (x_lims[1] - x_lims[0]),
                    loss_min,
                    f'{loss_min:.4g}')
    ax_scatter.axvline(x=x_min, c='r', ls='--', alpha=0.4)
    ax_scatter.axhline(y=loss_min, c='r', ls='--', alpha=0.4)

    return ax_arr, m


def plot_sim_annealing_scatter_hist_2d(ax_arr, parameter_array, loss, labels=None, with_arrows=False):
    guideline_color = 'g'
    padding = 0.02
    ax_scatter, ax_hist = ax_arr
    if labels is None:
        labels = ['x1', 'x2']
    parameter_array = np.array(parameter_array)
    loss = np.array(loss)
    index_min = np.argmin(loss)
    effective_n_iterations = len(loss)
    x_min = parameter_array[index_min]
    loss_min = loss[index_min]

    n_points = parameter_array.shape[0]
    # =============================================
    # =========     SCATTER PLOT    ===============
    # =============================================

    m = ax_scatter.scatter(parameter_array[:, 0], parameter_array[:, 1],
                           c=loss,
                           alpha=0.6,
                           norm=LogNorm())
    cb = plt.colorbar(m, ax=ax_scatter, label='loss', location='top')
    cb.set_alpha(1)
    cb.draw_all()
    ax_scatter.set_xlabel(labels[0])
    ax_scatter.set_ylabel(labels[1])

    # =============================================
    # =========      ARROWS      ==================
    # =============================================
    if with_arrows:
        pace_x = np.diff(parameter_array[:, 0])
        pace_y = np.diff(parameter_array[:, 1])
        arrows_kwargs = {'ls': '--', 'head_width': 0.01, }
        for n in range(len(pace_x)):
            arrows_kwargs['color'] = mpl.colormaps['viridis'](Normalize(0, len(pace_x))(n))
            ax_scatter.arrow(parameter_array[n, 0],
                             parameter_array[n, 1],
                             pace_x[n],
                             pace_y[n], **arrows_kwargs)

    # =============================================
    # =========     HISTOGRAM    ==================
    # =============================================

    m = ax_hist.hist2d(parameter_array[:, 0], parameter_array[:, 1],
                       bins=int(np.sqrt(n_points)),
                       norm=LogNorm())
    cbar = plt.colorbar(m[-1], ax=ax_hist, label='counts', location='top')

    ax_hist.set_xlabel(labels[0])
    ax_hist.set_ylabel(labels[1])

    # =============================================
    # =========       GUIDELINES      =============
    # =============================================

    # SCATTER

    ax_scatter.axvline(x=x_min[0], c=guideline_color, ls='--', lw=2, alpha=0.4)
    ax_scatter.axhline(y=x_min[1], c=guideline_color, ls='--', lw=2, alpha=0.4)

    # =============================================
    # =========       ANNOTATIONS      ============
    # =============================================
    # SCATTER
    x_lims = ax_scatter.get_xlim()
    y_lims = ax_scatter.get_ylim()
    ax_scatter.text(x_min[0],
                    y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                    f'{x_min[0]:.4g}')
    ax_scatter.text(x_lims[1] + padding * (x_lims[1] - x_lims[0]),
                    x_min[1],
                    f'{x_min[1]:.4g}')

    return ax_arr, m, cbar


def plot_sim_annealing_over_time(ax_arr, parameters_array, loss_array, steps, t_array=None, labels=None):
    padding = 0.02

    guideline_color = 'g'

    parameters_array = np.array(parameters_array)
    loss = np.array(loss_array)
    n_dim = parameters_array.shape[-1]
    if labels is None:
        labels = [f'$X_{{{i}}}$' for i in range(n_dim)]
    ax_loss, ax_parameters = ax_arr

    index_min = np.argmin(loss)

    x_min = parameters_array[index_min]
    loss_min = loss_array[index_min]
    step_min = steps[index_min]

    # =============================================
    # =========     LOSS OVER TIME    =============
    # =============================================
    artist_loss = ax_loss.plot(steps, loss, label='loss')

    ax_loss.set_yscale("log", nonpositive='clip')
    if t_array is not None:
        ax_temp = ax_loss.twinx()
        artist_temperature = ax_temp.plot(steps, t_array, 'r', label='temp.', alpha=0.6)

        all_artists = artist_temperature + artist_loss
        list_of_labels = [l.get_label() for l in all_artists]
        ax_loss.legend(all_artists, list_of_labels)
        ax_temp.set_ylabel('temperature')
        ax_temp.set_ylim((0, None))

    ax_loss.set_xlabel('SA step')
    ax_loss.set_ylabel('loss')

    # =============================================
    # =========     PARAMETERS OVER TIME    =============
    # =============================================
    for i in range(n_dim):
        ax_parameters.plot(parameters_array[:, i], label=labels[i], alpha=0.4)

    ax_parameters.set_xlabel('SA step')
    ax_parameters.set_ylabel('$X_i$')
    ax_parameters.legend()

    # =============================================
    # =========       GUIDELINES      =============
    # =============================================

    # LOSS OVER TIME

    ax_loss.axhline(y=loss_min, c=guideline_color, ls='--', alpha=0.4)
    ax_loss.axvline(x=step_min, c=guideline_color, ls='--', alpha=0.4)

    # PARAMETERS OVER TIME

    for i in range(n_dim):
        ax_parameters.axhline(y=x_min[i], c=guideline_color, ls='--', alpha=0.4)
    ax_parameters.axvline(x=step_min, c=guideline_color, ls='--', alpha=0.4)

    # =============================================
    # =========       ANNOTATIONS      ============
    # =============================================

    # LOSS OVER TIME

    x_lims = ax_loss.get_xlim()
    y_lims = ax_loss.get_ylim()
    ax_loss.text(x_lims[0]
                 - padding * (x_lims[1] - x_lims[0]),
                 loss_min,
                 f'{loss_min:.4g}',
                 horizontalalignment='right',
                 verticalalignment='center',
                 )
    ax_loss.text(step_min, y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                 f'iter=\n{step_min}',
                 horizontalalignment='center',
                 verticalalignment='bottom', )

    # PARAMETERS OVER TIME
    x_lims = ax_parameters.get_xlim()
    y_lims = ax_parameters.get_ylim()
    for i in range(n_dim):
        ax_parameters.text(x_lims[1] + padding * (x_lims[1] - x_lims[0]),
                           x_min[i],
                           labels[i] + '\n' + f'={x_min[i]:.4g}')
    ax_parameters.text(step_min, y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                       f'iter=\n{step_min}',
                       horizontalalignment='center',
                       verticalalignment='bottom', )
    return ax_arr


def sim_annealling_summary_from_sa_file(config_file, do_candidates=True):

    with open(config_file, 'r') as f:
        sa_yaml = yaml.safe_load(f)
    if isinstance(sa_yaml['decomposition_parameters']['file'], str):
        with open(sa_yaml['decomposition_parameters']['file'], 'r') as f:
            decomposition_yaml = yaml.safe_load(f)
    else:
        with open(sa_yaml['decomposition_parameters']['file'][0], 'r') as f:
            decomposition_yaml = yaml.safe_load(f)

    path_to_data = decomposition_yaml['run_parameters']['input_folder']
    path_to_annealing = sa_yaml['run_parameters']['output_folder']


    path_to_history = os.path.join(path_to_annealing, 'annealing_history.csv')
    path_to_annealing_results = os.path.join(path_to_annealing, 'sim_annealing_results.yaml')
    history = pd.read_csv(path_to_history, delimiter=',', )
    with open(path_to_annealing_results, 'r') as f:
        annealing_results = yaml.safe_load(f)
    last_iteration = len(history) - 1
    annealing_parameters = {'search_parameters':{ 'parameters_to_search': ['wing_loading']}}
    # GPS - wind - thermal - bird
    history = history[annealing_parameters['search_parameters']['parameters_to_search']
                      + ['loss']
                      + [ f'{p}_candidate' for p in annealing_parameters['search_parameters']['parameters_to_search'] ]
                      + ['loss_candidate',
                         'acceptance_probability',
                         'accept',
                         'new_global_minimum',
                         'temperature']]
    #labels = history.columns[:2].values
    labels = annealing_parameters['search_parameters']['parameters_to_search']
    labels_candidate = [l.replace('# ', '') + '_candidate' for l in labels]
    return plot_sim_annealing_summary_from_history(history, labels=labels_candidate if do_candidates else labels,
                                            skip=0,
                                                          plot_candidates=True,
                                            fig_kwargs={'layout': 'constrained',
                                                        'figsize':            (19, 16),
                                                                         'sharex': 'col'} ), history

def plot_sim_annealing_summary_from_history(history, annealing_result=None, labels=None, plot_candidates=False,
                                            skip=10, fig_kwargs=None):
    # 4 corresponds to accept_probability, accepted, new_global_minimum, temperature columns
    # 2 corresponds to loss loss_candidate columns
    # the remaining columns are 2 columns per parameter

    if fig_kwargs is None:
        fig_kwargs = {'constrained_layout': True,
                      'figsize':            (19, 16)}
    n_parameters = int((history.shape[-1] - 4 - 2) / 2 )
    list_of_parameters = history.columns.values[:n_parameters]

    parameter_array = history[list_of_parameters].values
    loss_array = history['loss'].values
    parameter_candidate_array = history[[p + '_candidate' for p in list_of_parameters]].values
    loss_candidate_array = history['loss_candidate'].values
    temperature = history['temperature'].values

    if plot_candidates:
        plotting_array = np.array(parameter_candidate_array)
        loss = np.array(loss_candidate_array)
    else:
        plotting_array = np.array(parameter_array)
        loss = np.array(loss_array)

    if annealing_result is not None:
        index_min = annealing_result['i']
    else:
        index_min = np.argmin(loss)
    effective_n_iterations = len(loss)

    skip_from = max(index_min - 1, 0) if index_min <= skip else skip

    plotting_array = plotting_array[skip_from:]
    loss = loss[skip_from:]
    temperature = temperature[skip_from:]
    step_array = np.arange(skip_from, effective_n_iterations)

    if labels is None:
        labels = [f'$X_{{{i}}}$' for i in range(n_parameters)]

    n_2d_plots = n_parameters // 2
    n_1d_plots = n_parameters % 2
    n_plots = 2 * n_2d_plots + 2 * n_1d_plots + 2
    fig, ax_arr = plt.subplots(2, n_plots // 2, **fig_kwargs)
    for j in range(n_2d_plots):
        plot_sim_annealing_scatter_hist_2d(ax_arr[:, j], plotting_array[:, 2 * j: 2 * (j + 1) + 1],
                                           loss, labels=labels[2 * j: 2 * (j + 1) + 1])
    if n_1d_plots:
        plot_sim_annealing_scatter_hist_1d(ax_arr[:, n_2d_plots],
                                           plotting_array[:, 2 * n_2d_plots],
                                           loss, label=labels[2 * n_2d_plots],
                                           steps=step_array)

    plot_sim_annealing_over_time(ax_arr[:, -1], plotting_array, loss, steps=step_array,
                                 t_array=temperature, labels=labels)
    return fig, ax_arr


def plot_sim_annealing_summary(ax_arr, parameters_array, loss_array, t_array=None, labels=None):
    padding = 0.02
    character_length = padding
    n_significant_digits = 4
    guideline_color = 'g'
    ax_scatter, ax_loss, ax_hist, ax_parameters = ax_arr.flatten()
    parameters_array = np.array(parameters_array)
    _, n_dim = parameters_array.shape

    if labels is None:
        labels = [f'X_{i}' for i in range(n_dim)]
    x_array = np.array(parameters_array)
    loss = np.array(loss_array)
    cbar = None
    index_min = np.argmin(loss)
    effective_n_iterations = len(loss_array)
    x_min = x_array[index_min]
    loss_min = loss[index_min]
    fig = ax_scatter.get_figure()
    # =============================================
    # =========     SCATTER PLOT    ===============
    # =============================================
    if n_dim == 2:
        m = ax_scatter.scatter(parameters_array[:, 0], parameters_array[:, 1],  # x_array, loss / x_array,
                                 c=loss_array,
                                 #s=,
                                 alpha=0.6,
                                 norm=LogNorm())
        plt.colorbar(m, ax=ax_scatter, label='loss', location='left')
        ax_scatter.set_xlabel(labels[0])
        ax_scatter.set_ylabel(labels[1])

    elif n_dim == 1:

        m = ax_scatter.scatter(parameters_array, loss_array,
                               c=np.arange(effective_n_iterations),
                               s=5,
                               alpha=0.6,
                               norm=LogNorm())
        cbar = plt.colorbar(m, ax=ax_scatter, label='SA step', location='left')
        ax_scatter.set_xlabel(labels[0])
        ax_scatter.set_ylabel('loss')
    else:
        raise NotImplementedError

    # =============================================
    # =========     LOSS OVER TIME    =============
    # =============================================
    artist_loss = ax_loss.plot(loss_array, alpha=0.6, label='loss')

    ax_loss.set_yscale("log", nonpositive='clip')
    if t_array is not None:
        ax_temp = ax_loss.twinx()
        artist_temperature = ax_temp.plot(t_array, 'r', label='temp.')

        all_artists = artist_temperature + artist_loss
        list_of_labels = [l.get_label() for l in all_artists]
        ax_loss.legend(all_artists, list_of_labels)
        ax_temp.set_ylabel('temperature')
        ax_temp.set_ylim((0, None))

    #ax_loss.set_ylim((0, None))

    ax_loss.set_xlabel('SA step')
    ax_loss.set_ylabel('loss')

    # =============================================
    # =========     HISTOGRAM    ==================
    # =============================================
    if n_dim == 2:
        m = ax_hist.hist2d(x_array[:, 0], x_array[:, 1],
                                bins=int(np.sqrt(effective_n_iterations)),
                                norm=LogNorm())
        cbar = plt.colorbar(m[-1], ax=ax_hist, label='counts', location='left')

        ax_hist.set_xlabel(labels[0])
        ax_hist.set_ylabel(labels[1])
    elif n_dim == 1:

        m = ax_hist.hist(parameters_array[:, 0],
                              bins=int(np.sqrt(effective_n_iterations)))

        ax_hist.set_xlabel(labels[0])
        ax_hist.set_ylabel('counts')
    else:
        raise NotImplementedError

    # =============================================
    # =========     PARAMETERS OVER TIME    =============
    # =============================================
    for i in range(n_dim):
        ax_parameters.plot(parameters_array[:, i], label=labels[i], alpha=0.4)

    ax_parameters.set_xlabel('SA step')
    ax_parameters.set_ylabel('$X_i$')
    ax_parameters.legend()

    # =============================================
    # =========       GUIDELINES      =============
    # =============================================

    # SCATTER

    ax_scatter.axvline(x=x_min[0], c=guideline_color, ls='--', alpha=0.4)
    ax_scatter.axhline(y=x_min[1], c=guideline_color, ls='--', alpha=0.4)

    # LOSS OVER TIME

    ax_loss.axhline(y=loss_min, c=guideline_color, ls='--', alpha=0.4)
    ax_loss.axvline(x=index_min, c=guideline_color, ls='--', alpha=0.4)

    # PARAMETERS OVER TIME

    for i in range(n_dim):
        ax_parameters.axhline(y=x_min[i], c=guideline_color, ls='--', alpha=0.4)
    ax_parameters.axvline(x=index_min, c=guideline_color, ls='--', alpha=0.4)

    # =============================================
    # =========       ANNOTATIONS      ============
    # =============================================
    # SCATTER
    x_lims = ax_scatter.get_xlim()
    y_lims = ax_scatter.get_ylim()
    ax_scatter.text(x_min[0],
                    y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                    f'{x_min[0]:.4g}')
    ax_scatter.text(x_lims[1] + padding* (x_lims[1] - x_lims[0]),
                    x_min[1],
                    f'{x_min[1]:.4g}')

    # LOSS OVER TIME

    x_lims = ax_loss.get_xlim()
    y_lims = ax_loss.get_ylim()
    ax_loss.text(x_lims[0]
                 - padding * (x_lims[1] - x_lims[0]),
                 loss_min,
                 f'{loss_min:.4g}',
                 horizontalalignment='right',
                 verticalalignment='center',
                 )
    ax_loss.text(index_min, y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                 f'iter=\n{index_min}',
                 horizontalalignment='center',
                 verticalalignment='bottom',)

    # PARAMETERS OVER TIME
    x_lims = ax_parameters.get_xlim()
    y_lims = ax_parameters.get_ylim()
    for i in range(n_dim):
        ax_parameters.text(x_lims[1] + padding * (x_lims[1] - x_lims[0]),
                           x_min[i],
                           labels[i] + '\n' + f'={x_min[i]:.4g}')
    ax_parameters.text(index_min, y_lims[1] + padding * (y_lims[1] - y_lims[0]),
                       f'iter=\n{index_min}',
                       horizontalalignment='center',
                       verticalalignment='bottom',)
    return ax_arr, m, cbar


def plot_wing_loading_sweep_var(ax, df_results, sweep_var):
    df_agg = df_results.groupby(sweep_var).agg(WL_SA_avg=('WL_SA', np.nanmean),
                                               WL_SA_median=('WL_SA', np.nanmedian),
                                               WL_SA_std=('WL_SA', np.nanstd),
                                               WL_avg_avg=('WL_avg', np.nanmean),
                                               WL_avg_median=('WL_avg', np.nanmedian),
                                               WL_avg_std=('WL_avg', np.nanstd),
                                               WL_std_avg=('WL_std', np.nanmean),
                                               WL_std_median=('WL_std', np.nanmedian),
                                               WL_std_std=('WL_std', np.nanstd),
                                               loss_avg=('loss', np.nanmean),
                                               loss_median=('loss', np.nanmedian),
                                               loss_std=('loss', np.nanstd),
                                               ).reset_index()

    ax[0].scatter(df_results[sweep_var], df_results['WL_SA'])
    ax[1].scatter(df_results[sweep_var], df_results['loss'])

    ax[0].errorbar(df_agg[sweep_var], df_agg['WL_SA_avg'], df_agg['WL_SA_std'], c='r', fmt='o')
    ax[0].errorbar(df_agg[sweep_var], df_agg['WL_avg_avg'], df_agg['WL_avg_std'],
               c='g',
               fmt='o-'
               )
    ax[1].errorbar(df_agg[sweep_var], df_agg['loss_avg'], df_agg['loss_std'], c='r', fmt='o')

    ax[0].set_xlabel(sweep_var)
    ax[1].set_xlabel(sweep_var)
    ax[0].set_ylabel('wing_loading')
    ax[1].set_ylabel('loss (m/s)')

def plot_wing_loading_sweep_var2d(ax, df_results, sweep_var):
    df_agg = df_results.groupby(sweep_var).agg(WL_SA_avg=('WL_SA', 'mean'),
                                               WL_SA_median=('WL_SA', 'median'),
                                               WL_SA_std=('WL_SA', 'std'),
                                               WL_avg_avg=('WL_avg', 'mean'),
                                               WL_avg_median=('WL_avg', 'median'),
                                               WL_avg_std=('WL_avg', 'std'),
                                               WL_std_avg=('WL_std', 'mean'),
                                               WL_std_median=('WL_std', 'median'),
                                               WL_std_std=('WL_std', 'std'),
                                               loss_avg=('loss', 'mean'),
                                               loss_median=('loss', 'median'),
                                               loss_std=('loss', 'std'),
                                               ).reset_index()
    unique_0 = df_agg[sweep_var[0]].unique()
    unique_1 = df_agg[sweep_var[1]].unique()
    sweep_var_mgs = np.meshgrid(unique_0, unique_1)
    df_agg = df_agg.sort_values(sweep_var)
    my_norm = CenteredNorm(vcenter=0,)
    my_cmap = cm.get_cmap('seismic')

    if isinstance(ax, (np.ndarray, list, tuple)):
        m_wl = ax[0].tricontourf(df_agg[sweep_var[0]],
                                 df_agg[sweep_var[1]],
                                 df_agg['WL_SA_median'] - df_agg['WL_avg_avg'].mean(), # .values.reshape(sweep_var_mgs[0].shape)
                             cmap=my_cmap, norm=my_norm, #edgecolors='k', linewidth=1.0,
                                levels=30
                             )
        wl_sizes = (df_agg['WL_SA_std'] / df_agg['WL_SA_std'].max()) * 0.1
        wl_avg_size = (np.median(df_agg['WL_SA_std']) / df_agg['WL_SA_std'].max()) * 0.1

        (_, _, _) = ax[0].errorbar(df_agg[sweep_var[0]], df_agg[sweep_var[1]], xerr=wl_sizes, yerr=wl_sizes,
                                   fmt='none', ecolor='k', #mfc=my_cmap(my_norm(df_agg['WL_SA_median'] - df_agg['WL_avg_avg'].mean()))
                              )
        m_loss = ax[1].tricontourf(df_agg[sweep_var[0]],
                                   df_agg[sweep_var[1]],
                                   df_agg['loss_median'],
                                   cmap='gnuplot',
                                levels=30
                             )
        wl_sizes = (df_agg['loss_std'] / df_agg['loss_std'].max()) * 0.1

        (_, _, _) = ax[1].errorbar(df_agg[sweep_var[0]], df_agg[sweep_var[1]],
                                   xerr=wl_sizes,
                                   yerr=wl_sizes,
                                   fmt='none', ecolor='k',
                              )

        ax[0].set_xlabel(sweep_var[0])
        ax[0].set_ylabel(sweep_var[1])
        ax[1].set_xlabel(sweep_var[0])
        ax[1].set_ylabel(sweep_var[1])
        cbar_wl = plt.colorbar(ScalarMappable(cmap=my_cmap, norm=my_norm), ax=ax[0])
        cbar_loss = plt.colorbar(ScalarMappable(cmap=m_loss.cmap, norm=m_loss.norm), ax=ax[1])
    else:
        m_wl = ax.tricontourf(df_agg[sweep_var[0]],
                              df_agg[sweep_var[1]],
                              df_agg['WL_SA_median'] - df_agg['WL_avg_avg'].mean(),
                              cmap=my_cmap, norm=my_norm,
                              levels=30
                             )
        wl_sizes = (df_agg['WL_SA_std'] / df_agg['WL_SA_std'].max()) * 0.1
        wl_avg_size = (np.median(df_agg['WL_SA_std']) / df_agg['WL_SA_std'].max()) * 0.1

        (_, _, _) = ax.errorbar(df_agg[sweep_var[0]], df_agg[sweep_var[1]], xerr=wl_sizes, yerr=wl_sizes,
                                   fmt='none', ecolor='k',
                              )

        ax.set_xlabel(sweep_var[0])
        ax.set_ylabel(sweep_var[1])
        cbar_wl = plt.colorbar(m_wl, ax=ax)
        cbar_loss = None

    return cbar_wl, cbar_loss


def plot_wing_loading_sweep_var(ax, df_results, sweep_var):
    df_agg = df_results.groupby(sweep_var).agg(WL_SA_avg=('WL_SA', 'mean'),
                                               WL_SA_median=('WL_SA', 'median'),
                                               WL_SA_std=('WL_SA', 'std'),
                                               WL_avg_avg=('WL_avg', 'mean'),
                                               WL_avg_median=('WL_avg', 'median'),
                                               WL_avg_std=('WL_avg', 'std'),
                                               WL_std_avg=('WL_std', 'mean'),
                                               WL_std_median=('WL_std', 'median'),
                                               WL_std_std=('WL_std', 'std'),
                                               loss_avg=('loss', 'mean'),
                                               loss_median=('loss', 'median'),
                                               loss_std=('loss', 'std'),
                                               ).reset_index()
    marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                        markersize=15, markerfacecoloralt='g')
    if isinstance(ax, (np.ndarray, list, tuple)):
        ax[0].scatter(df_results[sweep_var], #+ (df_results[sweep_var].max() - df_results[sweep_var].min()) * 0.051,
                      df_results['WL_SA'], c='b', s=10)
        ax[1].scatter(df_results[sweep_var], #+ (df_results[sweep_var].max() - df_results[sweep_var].min()) * 0.051,
                      df_results['loss'], c='b', s=10)

        ax[0].errorbar(df_agg[sweep_var], df_agg['WL_SA_median'], df_agg['WL_SA_std'], c='r', fmt='o')
        ax[0].errorbar(df_agg[sweep_var], df_agg['WL_avg_avg'], df_agg['WL_avg_std'],
                       #mfc='none', mec='g',
                       c='g', marker='*'
                       )
        ax[1].errorbar(df_agg[sweep_var], df_agg['loss_avg'], df_agg['loss_std'], c='r', fmt='o')

        ax[0].set_xlabel(sweep_var.replace('_nominal', ''))
        ax[1].set_xlabel(sweep_var)
        ax[0].set_ylabel('WL ' + '$(kg m^{-1})$')
        ax[1].set_ylabel('loss ' + '$(m s^{-1})$')
    else:
        ax.scatter(df_results[sweep_var], #+ (df_results[sweep_var].max() - df_results[sweep_var].min()) * 0.051,
                      df_results['WL_SA'], c='b', s=10)

        ax.errorbar(df_agg[sweep_var], df_agg['WL_SA_median'], df_agg['WL_SA_std'], c='r', fmt='o')
        ax.errorbar(df_agg[sweep_var], df_agg['WL_avg_avg'], df_agg['WL_avg_std'],
                       #mfc='none', mec='g',
                       c='g', marker='*'
                       )
        ax.set_xlabel(sweep_var.replace('_nominal', ''))
        ax.set_ylabel('WL ' + '$(kg m^{-1})$')

def plot_kde_modes_histograms(ax_arr, df, df_modes, list_of_birds=None, kde_kwargs=None):
    if list_of_birds is None:
        list_of_birds = df_modes['bird_name'].unique().tolist()
    if kde_kwargs is not None:
        kde_kwargs = {}
    for _, (_, current_row) in enumerate(df_modes.iterrows()):
        current_col = current_row['bird_name']
        parameter_col = f'wing_loading_{current_col}_candidate'
        i = list_of_birds.index(current_col)
        h, bin_edges, artist = ax_arr[i].hist(df[parameter_col], bins=round(np.sqrt(len(df))),
                                              density=True, alpha=0.75)
        current_wl_array = np.linspace(bin_edges[0], bin_edges[-1], endpoint=True, num=100)

        current_kde = gaussian_kde(df[parameter_col], **kde_kwargs)
        current_kde_values = current_kde(current_wl_array)

        ax_arr[i].plot(current_wl_array, current_kde_values, 'r')
#        ax_arr[i].axvline(x=current_row['WL_SA'], c='y', label='SA result')
        ax_arr[i].axvline(x=current_row['WL_mode'], c='k', label='KDE mode')

        ax_arr[i].set_title(current_col)
    ax_arr[i].legend(loc='lower right')