import os.path
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import scienceplots
import matplotlib



import yaml
from scipy import signal
from scipy.optimize import curve_fit

from calc.analysis.turbulence import get_single_sweep_autocorrelation
from object.air import ReconstructedAirVelocityField


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
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
output_dpi = 320

root_path = '/home/pedro/PycharmProjects/ThermalModelling'

color_list = [f'C{i + 1}' for i in range(10)]

p = Path(os.path.join(root_path, 'results/turbulence/same_flock_WL=6.0/mesoscale'))
path_to_csv = str(p)
save_folder = os.path.join(path_to_csv, 'results', 'turbulence', 'mesoscale')

os.makedirs(save_folder, exist_ok=True)

list_of_datatypes = ['dec','gt','data', 'turb']

df_all_correlations_dec_avg = pd.read_csv(os.path.join(path_to_csv, 'turbulence_correlation_grid_size=2_avg.csv'), index_col=False)
df_all_correlations_gt_avg = pd.read_csv(os.path.join(path_to_csv, 'turbulence_correlation_gt_grid_size=2_avg.csv'), index_col=False)

df_all_correlations_dec_avg.rename(columns={ f'{col}_{stat}': f'{col}_{stat}_dec' for col, stat in product(['inner', 'x', 'y', 'z'], ['mean', 'median', 'std', 'count', 'sem'])},
                           inplace=True)
df_all_correlations_gt_avg.rename(columns={ f'{col}_{stat}': f'{col}_{stat}_gt' for col, stat in product(['inner', 'x', 'y', 'z'], ['mean', 'median', 'std', 'count', 'sem'])},
                           inplace=True)
df_all_correlations_data_avg.rename(columns={ f'{col}_{stat}': f'{col}_{stat}_data' for col, stat in product(['inner', 'x', 'y', 'z'], ['mean', 'median', 'std', 'count', 'sem'])},
                           inplace=True)
df_all_correlations_data_turb_avg.rename(columns={ f'{col}_{stat}': f'{col}_{stat}_turb' for col, stat in product(['inner', 'x', 'y', 'z'], ['mean', 'median', 'std', 'count', 'sem'])},
                           inplace=True)

turbulence_lookup_mpers = {0: 0.22,
                           1: 0.30,
                           2: 0.28,
                           3: 0.42,
                           4: 0.41,
                           5: 0.61, }

df_all_correlations_avg = pd.merge(df_all_correlations_dec_avg, df_all_correlations_gt_avg, on=['thermal', 'bin_z_size', 'size', 'delta_R'], how='left')
df_all_correlations_avg = pd.merge(df_all_correlations_avg, df_all_correlations_data_avg, on=['thermal', 'bin_z_size', 'size', 'delta_R'], how='left')
df_all_correlations_avg = pd.merge(df_all_correlations_avg, df_all_correlations_data_turb_avg, on=['thermal', 'bin_z_size', 'size', 'delta_R'], how='left')


df_all_correlations_avg['turbulence_level'] = df_all_correlations_avg['thermal'].apply(lambda x: turbulence_lookup_mpers[x])

list_of_thermals = df_all_correlations_avg.sort_values('turbulence_level')['thermal'].unique()
# list_of_thermals = df_all_correlations_dec_avg['thermal'].unique()

list_of_sizes = df_all_correlations_dec_avg['size'].unique()


dict_of_popt = []
for i_thermal, thermal in enumerate(list_of_thermals):
    for i_size, size in enumerate(list_of_sizes[1:]):
        for i_comp, comp in enumerate(['inner',
                                       'x',
                                       'y',
                                       'z']):
            for data_type in list_of_datatypes: #
                current_corr = df_all_correlations_avg[np.all(df_all_correlations_avg[['thermal','size']] == (thermal, size), axis=1)]
                current_corr = current_corr[(current_corr['delta_R'] >= 10)
                #                            & (current_corr[f'{comp}_mean'] > 0)
                ]

                fit_result = curve_fit(lambda x, l, b: b * x ** l,
                                       current_corr['delta_R'].values,
                                       current_corr[f'{comp}_mean_{data_type}'].values,
                                       p0=(0.01,1),
                                       sigma=current_corr[f'{comp}_sem_{data_type}'] if data_type not in ['data', 'turb'] else None)
                popt, pcov = fit_result
                dict_of_popt.append({'thermal': thermal,'size': size, 'comp': comp, 'data_type': data_type,
                                     'l': popt[0], 'b': popt[1],
                                     'sigma_l': np.sqrt(np.diag(pcov))[0],
                                     'sigma_b': np.sqrt(np.diag(pcov))[1]})

df_popt = pd.DataFrame.from_records(dict_of_popt)



fig, ax = plt.subplots(2,len(list_of_thermals),
                       sharex='row',
                       sharey='row'
                       )

for i_thermal, thermal in enumerate(list_of_thermals):
    for i_size, size in enumerate(list_of_sizes[-1:]):
        for i_comp, comp in enumerate([#'inner',
                                       'x',
                                       'y',
                                       #'z'
                                       ]):
            for data_type in list_of_datatypes: #
                current_corr = df_all_correlations_avg[np.all(df_all_correlations_avg[['thermal','size']] == (thermal, size), axis=1)]
                current_corr = current_corr[current_corr['delta_R'] > 2]
                # current_corr = current_corr[current_corr[f'{comp}_mean'] > 0]
                current_fits = df_popt.loc[np.all(df_popt[['thermal','size', 'comp', 'data_type']] == (thermal, size, comp, data_type),
                                                                axis=1), ['l', 'sigma_l']].values
                if current_fits.size != 0:
                    current_popt, current_pcov = current_fits[0]

                ax[i_comp, i_thermal].errorbar(current_corr['delta_R'].values,
                                               current_corr[f'{comp}_mean_{data_type}'].values,
                                               current_corr[f'{comp}_std_{data_type}'].values / np.sqrt(current_corr[f'{comp}_count_{data_type}'].values),
                                               label=f'{data_type}\n{current_popt:.3g} +- {current_pcov:.3g}' if current_pcov else data_type)
                ax[i_comp, i_thermal].axhline(y=0, c='k', ls=':', alpha=0.3 )

                ax[i_comp, i_thermal].legend()
                if i_thermal == 0:
                    ax[i_comp, i_thermal].set_ylabel(comp)
    ax[0, i_thermal].set_title(f'{thermal} - $\\sigma_{{Turb}} = {turbulence_lookup_mpers[thermal]}$')
    # ax[0, i_thermal].set_title(f'')
    ax[-1, i_thermal].set_xlabel('$\\Delta R\\ (m)$')


