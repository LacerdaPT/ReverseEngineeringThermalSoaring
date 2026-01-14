import os.path
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import scienceplots
import matplotlib


import yaml
from scipy import signal
from scipy.optimize import curve_fit

from calc.analysis.turbulence import get_single_sweep_autocorrelation
from misc.constants import root_path
from object.air import ReconstructedAirVelocityField


p = Path(os.path.join(root_path, 'results/turbulence/storks/mesoscale'))
path_to_csv = str(p)
save_folder = os.path.join(path_to_csv, 'figures')


# df_all_correlations = pd.read_csv(os.path.join(path_to_csv, 'turbulence_correlation_no_noise_grid_size=2.csv'), index_col=False)
df_all_correlations_avg = pd.read_csv(os.path.join(path_to_csv, 'turbulence_correlation_no_noise_grid_size=2_avg.csv'), index_col=False)

df_all_correlations_avg['xy_mean'] = df_all_correlations_avg['x_mean'] + df_all_correlations_avg['y_mean']
df_all_correlations_avg['xy_std'] = np.sqrt(df_all_correlations_avg['x_std'] ** 2 + df_all_correlations_avg['y_std'] ** 2)
df_all_correlations_avg['xy_sem'] = df_all_correlations_avg['xy_std'] / df_all_correlations_avg['x_count']
list_of_thermals = df_all_correlations_avg['thermal'].sort_values().unique()

list_of_sizes = df_all_correlations_avg['size'].unique()[:1]

list_of_components = ['inner', 'x', 'y', 'xy', 'z']
