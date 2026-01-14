import os
from copy import deepcopy
from functools import reduce
from itertools import product


import shelve
from pathlib import Path

from dill import Pickler, Unpickler

from calc.post_processing.air_velocity_field import get_stats_on_wind_and_thermal

shelve.Pickler = Pickler
shelve.Unpickler = Unpickler

import numpy as np
import pandas as pd

import matplotlib
import scienceplots
# import yaml
# from scipy.spatial import ConvexHull
#
# from calc.post_processing.air_velocity_field import get_thermal_and_wind_from_air_velocity_points
# from misc.auxiliar import sanitize_dict_for_yaml

save = False
matplotlib.style.use(['science'])
if save:
    matplotlib.use('Cairo')
    matplotlib.interactive(False)

from matplotlib import pyplot as plt




#plt.rcParams.update({'figure.dpi': '320'})
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
output_dpi = 320

from scipy.stats import cauchy, norm, laplace, PermutationMethod

from calc.auxiliar import UnivariateSplineWrapper
from calc.stats import get_rms_and_pearson
from data.get_data import load_decomposition_data, load_synthetic_data
from object.air import ReconstructedAirVelocityField


root_path = '/home/pedro/ThermalModelling'
base_path = os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/same_flock_WL=6.0' )
glob_string = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'


path_to_save = os.path.join(root_path,'results/metrics/same_flock_WL=6.0' )

#*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed
df_stats = pd.read_csv(os.path.join(path_to_save, 'pearson_rms.csv'))
if save:
    os.makedirs(os.path.join(path_to_save, 'png'), exist_ok=True)
    os.makedirs(os.path.join(path_to_save, 'svg'), exist_ok=True)
pp = Path(os.path.join(root_path, base_path))


for i_thermal, ss in enumerate( pp.glob(glob_string)):
    # try:
    path_to_decomposition = str(ss)

    current_dataset = ss.parents[6].name
    current_df_stats = df_stats[df_stats['thermal'].astype(str) == current_dataset]
    dec = load_decomposition_data(path_to_decomposition, iteration='best',
                                  list_of_files=['aggregated.csv',
                                                 'iterations.csv',
                                                 'splines.yaml',
                                                 'decomposition_args.yaml',
                                                 'inter_args.yaml'
                                                 #'bins_final_post.csv',
                                                 ])
    path_to_synthetic = dec['decomposition_args']['run_parameters']['input_folder']




    syn = load_synthetic_data(path_to_synthetic, list_of_object=['air.csv'])
    df_dec = dec['iterations']
    df_air = syn['air']
    'thermal.csv'
    # df_thermal_gt = df_thermal_gt[df_thermal_gt['Z'].between( np.min(wind_spline_gt['wind_X']['tck'][0]), np.max(wind_spline_gt['wind_X']['tck'][0]))]

    df_thermal_gt = pd.read_csv(os.path.join(path_to_decomposition, 'ground_truth_reconstructed','thermal.csv'), index_col=False)
    df_thermal = pd.read_csv(os.path.join(path_to_decomposition,'thermal.csv'), index_col=False)

    df_thermal_merge = pd.merge(df_thermal_gt, df_thermal, on=['bird_name','time'], suffixes=('_GT', '_dec'))
    df_compare = df_thermal_merge[['bird_name','time',
                                   'dXdT_air_ground_GT',
                                   'dXdT_air_ground_dec',
                                   'dYdT_air_ground_GT',
                                   'dYdT_air_ground_dec',
                                   'dZdT_air_ground_GT',
                                   'dZdT_air_ground_dec',
                                   'dXdT_thermal_ground_GT',
                                   'dXdT_thermal_ground_dec',
                                   'dYdT_thermal_ground_GT',
                                   'dYdT_thermal_ground_dec',
                                   'dZdT_thermal_ground_GT',
                                   'dZdT_thermal_ground_dec',
                                   'wind_X_GT', 'wind_X_dec',
                                   'wind_Y_GT','wind_Y_dec']]



    label_dict = {f'd{col}dT_air_ground_dec': f'\\hat{{V}}^{{\\ dec}}_{{{col}}}'           for i_col, col in enumerate(['X', 'Y', 'Z'])}
    label_dict = label_dict | {f'd{col}dT_air_ground_GT': f'\\hat{{V}}^{{\\ GT}}_{{{col}}}'             for i_col, col in enumerate(['X', 'Y', 'Z'])}

    label_dict = label_dict | {f'wind_{col}_GT':          f'\\hat{{V}}^{{\\ GT}}_{{{col}}}'     for i_col, col in enumerate(['X', 'Y'])}
    label_dict = label_dict | {f'wind_{col}_dec':         f'\\hat{{V}}^{{\\ dec}}_{{{col}}}'    for i_col, col in enumerate(['X', 'Y'])}

    fig_thermal = plt.figure(layout='constrained', figsize=(18, 9))
    fig_thermal_scatter, fig_thermal_hist = fig_thermal.subfigures(1, 2,width_ratios=(1,1))

    ax_thermal_hist = fig_thermal_hist.subplots(3,6)
    ax_thermal_scatter = fig_thermal_scatter.subplots(3,6, sharex='row', sharey='row')
    for i_coord, coord in enumerate(['X', 'Y', 'Z']):
        for i_cols, col in enumerate([f'd{coord}dT_air_ground',]):
            col1, col2 = f'{col}_GT', f'{col}_dec'
            current_hist_ax = ax_thermal_hist[i_coord, i_thermal]
            current_scatter_ax = ax_thermal_scatter[i_coord, i_thermal]
            (u, s,
             u_norm, s_norm,
             u_cauchy, s_cauchy,
             u_laplace, s_laplace,) = current_df_stats.loc[current_df_stats['col'] == col, ['mean', 'sigma',
                                                                                  'mean_norm', 'sigma_norm',
                                                                                  'mean_cauchy', 'sigma_cauchy',
                                                                                  'mean_laplace', 'sigma_laplace']].values[0]
            diff = df_compare[col1].values - df_compare[col2].values
            good_mask = (~np.isnan(diff)) & (diff != np.inf) & (np.abs((diff - u ) / s) < 4)
            diff = diff[good_mask]
            counts,bins,patches = current_hist_ax.hist(diff, bins=int(np.sqrt(len(diff))), density=True)
            #current_hist_ax.plot(bins, norm.pdf(bins, loc=u_norm, scale=s_norm), 'r--')
            #current_hist_ax.plot(bins, cauchy.pdf(bins, loc=u_cauchy, scale=s_cauchy), 'g--')
            #current_hist_ax.plot(bins, laplace.pdf(bins, loc=u_laplace, scale=s_laplace), 'k--')
            current_hist_ax.axvline(x=u, color='r')
            current_hist_ax.axvline(x=u + s, color='r', ls='--')
            current_hist_ax.axvline(x=u - s, color='r', ls='--')
            current_hist_ax.text(x=u, y = current_hist_ax.get_ylim()[-1], s=f'{u:.2f}', rotation=45)
            current_hist_ax.text(x=u + s, y = current_hist_ax.get_ylim()[-1], s=f'{s:.2f}', rotation=45)
            current_hist_ax.set_xlabel(f'$' + label_dict[col1] + ' - ' + label_dict[col2] + '$')

            current_scatter_ax.scatter(df_compare[col1].values[good_mask],
                                       df_compare[col2].values[good_mask], alpha=0.05, s=3)
            current_scatter_ax.plot([np.min(df_compare[[col1, col2]].values[good_mask]),np.max(df_compare[[col1, col2]].values[good_mask])],
                                    [np.min(df_compare[[col1, col2]].values[good_mask]),np.max(df_compare[[col1, col2]].values[good_mask])],
                                    'r--')
            (current_rms,
             current_correlation,
             current_p_value,
             current_n) = current_df_stats.loc[current_df_stats['col'] == col, ['rms',
                                                                                'correlation',
                                                                                'p_value',
                                                                                'n']].values[0]
            if current_p_value < 10**-3:
                current_title = f'$RMSE={current_rms:.3f}$\n $\\rho = {current_correlation:.3f}$\n $p<0.001$'
            else:
                current_title = f'$RMSE={current_rms:.3f}$\n $\\rho = {current_correlation:.3f}$\n $p={current_p_value:.4f}$'
            current_scatter_ax.set_title(current_title
                                         # f'$n={current_n}$\n'
                                         )

            current_scatter_ax.set_xlabel(f'$' + label_dict[col1] + '$')
            current_scatter_ax.set_ylabel(f'$' + label_dict[col2] + '$')

    [a.set_aspect('equal') for a in ax_thermal_scatter.ravel()]


    # [a.set_aspect('equal') for a in ax.ravel()]
    fig_thermal.suptitle('thermal')

    fig_wind = plt.figure(layout='constrained', figsize=(18, 7))
    fig_wind_scatter, fig_wind_hist = fig_wind.subfigures(1, 2, wspace=0.1)

    ax_wind_hist = fig_wind_hist.subplots(2,3)
    ax_wind_scatter = fig_wind_scatter.subplots(2,3, sharex='row', sharey='row')
    for i_coord, coord in enumerate(['X', 'Y']):
        for i_cols, col in enumerate([f'wind_{coord}',]):
            col1, col2 = f'{col}_GT', f'{col}_dec'
            (current_rms,
             current_correlation,
             current_p_value,
             current_n) = current_df_stats.loc[current_df_stats['col'] == col, ['rms',
                                                                                 'correlation',
                                                                                 'p_value',
                                                                                 'n']].values[0]
            current_hist_ax = ax_wind_hist[i_coord, i_cols]
            current_scatter_ax = ax_wind_scatter[i_coord, i_cols]
            diff = df_compare[col1].values - df_compare[col2].values
            diff = diff[(~np.isnan(diff)) & (diff != np.inf)]
            current_hist_ax.hist(diff, bins=int(np.sqrt(len(diff))), density=True)
            current_hist_ax.set_xlabel(f'$' + label_dict[col1] + ' - ' + label_dict[col2] + '$')

            current_scatter_ax.scatter(df_compare[col1], df_compare[col2], alpha=0.05, s=3)
            current_scatter_ax.plot([df_compare[[col1, col2]].min(),df_compare[[col1, col2]].max()],
                                    [df_compare[[col1, col2]].min(),df_compare[[col1, col2]].max()],
                                    'r--')
            current_scatter_ax.set_xlabel(f'$' + label_dict[col1] + '$')
            current_scatter_ax.set_ylabel(f'$' + label_dict[col2] + '$')
            if current_p_value < 10**-3:
                current_title = f'$RMSE={current_rms:.3f}$\n $\\rho = {current_correlation:.3f}$\n $p<0.001$'
            else:
                current_title = f'$RMSE={current_rms:.3f}$\n $\\rho = {current_correlation:.3f}$\n $p={current_p_value:.4f}$'
            current_scatter_ax.set_title(current_title
                                         )
    [a.set_aspect('equal') for a in ax_wind_scatter.ravel()]


    fig_wind.suptitle('wind')
    if save:
        fig_wind.canvas.draw()
        fig_thermal.canvas.draw()
        fig_wind.savefig(os.path.join(path_to_save, 'png', f'wind_correlations_{current_dataset}.png'))
        fig_wind.savefig(os.path.join(path_to_save, 'svg', f'wind_correlations_{current_dataset}.svg'))
        fig_thermal.savefig(os.path.join(path_to_save, 'png', f'thermal_correlations_{current_dataset}.png'))
        fig_thermal.savefig(os.path.join(path_to_save, 'svg', f'thermal_correlations_{current_dataset}.svg'))
        plt.close(fig_wind)
    else:
        plt.show(block=True)

