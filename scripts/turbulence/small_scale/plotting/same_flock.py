import os.path
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import scienceplots
import matplotlib

import yaml
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, CenteredNorm, TwoSlopeNorm
from scipy import signal

from misc.constants import root_path, turbulence_lookup_mpers
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
marker_list = 'o*+xd^v'

from data.get_data import load_decomposition_data, load_synthetic_and_decomposed, load_synthetic_data
pd.options.mode.chained_assignment = None
base_path = os.path.join(root_path, f'results/turbulence/same_flock_WL=6.0/small_scale')


# ==================================================================================================================== #
# ==========================================          PLOTTING         =============================================== #
# ==================================================================================================================== #

list_of_mappable = []

my_radius = 40
df_all_sigmas = pd.read_csv(os.path.join(base_path, f'turbulence_sigmas.csv'), index_col=False)
df_all_sigmas_GT = pd.read_csv(os.path.join(base_path, f'turbulence_sigmas_GT.csv'), index_col=False)

with open(os.path.join(base_path, 'fluctuations_GT.yaml'), 'r') as f:
    dict_of_fluctuations_GT = yaml.safe_load(f)
with open(os.path.join(base_path, 'fluctuations.yaml'), 'r') as f:
    dict_of_fluctuations = yaml.safe_load(f)

list_of_datasets = df_all_sigmas['dataset'].unique().astype(str)
list_of_datasets = sorted(list_of_datasets, key=lambda x: turbulence_lookup_mpers[int(x)])
list_of_radii =  df_all_sigmas['radius'].unique()
list_of_comp =  ['x', 'y', 'z', 'xyz']
df_all_sigmas_merge = pd.merge(df_all_sigmas, df_all_sigmas_GT, on=['dataset','radius'], suffixes=('_DEC', '_GT'))

my_norm = TwoSlopeNorm(0, vmin=-6, vmax=6)
my_cmap = matplotlib.colormaps['seismic']
my_norm = Normalize(vmin=0, vmax=1.5)
my_cmap = matplotlib.colormaps['gnuplot']
# df_all_sigmas = df_all_sigmas[df_all_sigmas['noise'] < 0.6]

#
# with open(os.path.join('results/turbulence/same_flock_WL=6.0/small_scle/individual_sigmas_GT.yaml'), 'r') as f:
#     individual_sigmas_GT = yaml.safe_load(f)
# with open(os.path.join('results/turbulence/same_flock_WL=6.0/small_scle/individual_sigmas.yaml'), 'r') as f:
#     individual_sigmas = yaml.safe_load(f)


fig, ax = plt.subplots(4,len(list_of_datasets), sharex='row')
for i_dataset, dataset in enumerate( list_of_datasets):
    for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz']):
        for datatype, current_dict_of_fluctuations in zip(['GT', 'DEC'], [dict_of_fluctuations_GT, dict_of_fluctuations]):
            current_fluctations = np.array( current_dict_of_fluctuations[dataset][my_radius][comp])
            current_fluctations = current_fluctations[np.abs(current_fluctations)< np.percentile(current_fluctations,99.9)]
            ax[i_comp,i_dataset].hist(current_fluctations,
                                     bins=int(np.sqrt(len(current_fluctations))), density=True,
                                     alpha=0.5, label=f'{datatype} proc')
    ax[0, i_dataset].set_title(f'{dataset} - $\\sigma_{{turb}} = {turbulence_lookup_mpers[int(dataset)]} \\ m s^{{-1}}$')

#
# fig, ax = plt.subplots(4,4, sharex='row')
#
#
#
# for i_dataset, dataset in enumerate( dict_of_fluctuations_GT.keys()):
#     for i_radius, radius in enumerate( dict_of_fluctuations_GT[dataset].keys()):
#         for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz']):
#             ax[i_comp, i_radius].scatter(individual_sigmas[dataset][radius]['average'][comp], individual_sigmas[dataset][radius]['fluctuations'][comp], c=f'C{i_radius}', marker='*')
#             ax[i_comp, i_radius].scatter(individual_sigmas_GT[dataset][radius]['average'][comp], individual_sigmas_GT[dataset][radius]['fluctuations'][comp], c=f'C{i_radius}', marker='o')
#             # ax[i_radius].plot(df_all_sigmas_GT['radius'], df_all_sigmas_GT[f'{comp}'], c=f'C{i_comp}', label=f'{comp} GT')
#             # ax[i_radius].plot(df_all_sigmas['radius'], df_all_sigmas[f'{comp}'], '--',c=f'C{i_comp}', label=comp )
#     break

agg_dict = {f'{comp}{suffix}_{stat}': (f'{comp}{suffix}', stat)
            for stat in ['mean', 'std', 'median', 'count']
            for i_comp, comp in enumerate(['x', 'y', 'z', 'xyz'])
            for suffix in ['_DEC', '_GT']}

df_all_sigma_grouped = df_all_sigmas_merge.groupby(['radius']).agg(**agg_dict).reset_index()

fig, ax_all = plt.subplots(4,len(list_of_datasets),figsize=(16,12),
                                 sharey='row',
                                 sharex='row',
                                 layout='constrained')

for i_dataset, dataset in enumerate(list_of_datasets):
    current_dataset_sigmas = df_all_sigmas_merge[df_all_sigmas_merge['dataset'].astype(str) == dataset]
    for i_coord, coord in enumerate(list_of_comp):
        current_ax = ax_all[i_coord, i_dataset]
        current_ax.scatter(current_dataset_sigmas['radius'], current_dataset_sigmas[f'{coord}_GT'], label='GT proc.', marker=marker_list[0])
        current_ax.scatter(current_dataset_sigmas['radius'], current_dataset_sigmas[f'{coord}_DEC'], label='DEC proc.', marker=marker_list[1])
        ax_all[i_coord, 0].set_ylabel(f'$\\delta v_{{ {coord} }} (m s ^{{-1}})$ ')
        current_ax.set_xlim((0.0, None))
        current_ax.set_ylim((0.0, None))

    ax_all[-1, i_dataset].set_xlabel('radius, R (m)')
    ax_all[0, i_dataset].set_title(f'{dataset} - $\\sigma_{{turb}} = {turbulence_lookup_mpers[int(dataset)]} $')

ax_all[0, i_dataset].legend()
for i_dataset, dataset in enumerate(list_of_datasets):
    current_dataset_sigmas = df_all_sigmas_merge[df_all_sigmas_merge['dataset'].astype(str) == dataset]
    for coord, current_ax in  ax_corr.items():
        current_ax.scatter( current_dataset_sigmas[f'{coord}_GT'], current_dataset_sigmas[f'{coord}_DEC'],label='GT proc.')

        current_ax.set_xlabel(f'{coord}, $\\delta v_{{ {coord} }} ^{{GT}} (m s ^{{-1}})$ ')
        current_ax.set_ylabel(f'{coord}, $\\delta v_{{ {coord} }} ^{{DEC}} (m s ^{{-1}})$ ')
        current_ax.set_aspect('equal')
