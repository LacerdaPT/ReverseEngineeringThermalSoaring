import copy
import os.path
from itertools import permutations, product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb, rgb_to_hsv, hsv_to_rgb, to_hex, TwoSlopeNorm
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from scipy.stats import ttest_1samp

from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame
from calc.stats import get_all_permutated_rms
from data.get_data import load_decomposition_data
from misc.constants import root_path

base_path = 'synthetic_data/from_atlasz/newdata/storks'
save_folder = os.path.join(root_path, 'results/air_velocity_field/storks/rotation')
pp=Path(os.path.join(root_path, base_path))
def mvg(X, a, mu_x, mu_y, s_x, s_y, s_xy):
    X = np.array(X)
     # = p
    mu = np.array([mu_x, mu_y])
    sigma = [[s_x, s_xy], [s_xy, s_y]]
    sigma = np.array(sigma)
    return a * np.exp(-0.5 * (X - mu).T @ (np.linalg.inv(sigma) @ ( X - mu)))


def my_quadratic(X, a,b,c,d,e,f,):
    x,y = X

    return a * x ** 2 + b * y ** 2 + c * x * y + d * x + e * y  + f
parameter_dict_lambda = lambda p: {'thermal': p.parents[6].name,
                                   'bin_z_size': p.parents[3].name.split('=')[1],}
is_synthetic = pp.name != 'storks'

# path_wildcard = f'rot_int=*/*/*/decomposition/average/0/final/reconstructed'
# for i in range(1):
path_wildcard = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'
list_of_ts = []

df_concat = pd.DataFrame()
my_norm = Normalize(vmin=0,vmax=5)
my_cmap = 'gnuplot'
list_stats = []
bin_size_rho = 5
bin_size_z = 20
list_of_thermals = ['b010_0.1',
# 'b023_0.1',
'b072_0.1',
'b077_0.1',
'b112_0.2',
'b121_0.1',]
fig, ax = plt.subplots(3,5, #sharex='all', sharey='all'
                       layout='constrained'
                       )

for i_thermal, current_thermal in enumerate(list_of_thermals):

    path_to_decomposition = os.path.join(base_path, current_thermal,
                                         'decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed')

    dec = load_decomposition_data(path_to_decomposition, list_of_files=['thermal.csv', 'splines.yaml'])
    splines = dec['splines'][0]
    df_thermal = dec['thermal']
    df_thermal = df_thermal[df_thermal['in_hull'] & (~df_thermal['interpolated_thermal_Z'])]
    df_thermal = df_thermal[np.abs(df_thermal['curvature']) < 0.1]

    df_thermal = df_thermal[df_thermal['Z'].between(np.min(splines['wind']['wind_X']['tck'][0]),
                                                    np.max(splines['wind']['wind_X']['tck'][0])) ]
    z_splits = [np.nanquantile(df_thermal['Z'].values, i * 1/3) for i in range(4)]
    for i_z in range(3):
        z_min, z_max = z_splits[i_z], z_splits[i_z+1]
        current_axis = ax[i_z, i_thermal]
        current_df_thermal = df_thermal[df_thermal['Z'].between(z_min, z_max )].copy()
        current_df_thermal['wind_norm'] = np.linalg.norm(current_df_thermal[['wind_X', 'wind_Y']], axis=1)
        current_df_thermal['wind_angle'] = np.arctan2(current_df_thermal['wind_Y'], current_df_thermal['wind_X'])# + np.pi
        for col in ['wind_norm','dZdT_thermal_ground']:
            list_stats.append([current_thermal, col,
                               current_df_thermal[col].mean(),
                               current_df_thermal[col].median(),
                               current_df_thermal[col].std(),
                               current_df_thermal[col].count(),
                               np.nanpercentile(current_df_thermal[col].values, 5),
                               np.nanpercentile(current_df_thermal[col].values, 95), ])
        current_df_thermal = current_df_thermal[current_df_thermal['dZdT_thermal_ground'] >= 0]
        m = current_axis.scatter(current_df_thermal['X_bird_TC'], current_df_thermal['Y_bird_TC'], c=current_df_thermal['dZdT_thermal_ground'], s=2, norm=my_norm, cmap=my_cmap)

        my_cov = np.cov(current_df_thermal[['X_bird_TC','Y_bird_TC']].T,# aweights=current_df_thermal['dZdT_thermal_ground'] + np.abs(current_df_thermal['dZdT_thermal_ground'].min())
                        )

        u,s,v = np.linalg.svd(my_cov)
        for i_eig in range(2):
            if np.isclose(u[0, i_eig], 0,atol=0.01):
                current_axis.axvline(x=0, ymin=-np.sqrt(s[i_eig]), ymax=np.sqrt(s[i_eig]))
            x_array = np.linspace(-np.sqrt(s[i_eig]), np.sqrt(s[i_eig]), 20, endpoint=True)
            my_line = u[1,i_eig] / u[0,i_eig] * x_array
            good_mask = (my_line > -150) & (my_line < 150)
            x_array = x_array[good_mask]
            my_line = my_line[good_mask]
            current_axis.plot(x_array, my_line)
        current_axis.plot(x_array, np.tan(current_df_thermal["wind_angle"].mean()) * x_array, 'g--')
        np.tan(current_df_thermal["wind_angle"].mean())

        current_axis.set_xlim((-150, 150))
        current_axis.set_aspect('equal')

        current_axis.set_title(current_thermal + f'\n {current_df_thermal["wind_angle"].mean() * 180 / np.pi:.1f}' + f'\n'
                                           + f'{np.arctan2(u[1,0] , u[0,0]) * 180 / np.pi:.1f} '
                                           + f'{np.arctan2(u[1,1] , u[0,1]) * 180 / np.pi:.1f} ')

plt.colorbar(m, ax=current_axis)


dict_of_norms = {'b010_0.1':Normalize(0,4),
                 #'b023_0.1':Normalize(0,4),
                 'b072_0.1':Normalize(0,2),
                 'b077_0.1':Normalize(0,3),
                 'b112_0.2':Normalize(0,4),
                 'b121_0.1':Normalize(0,4)}
fig_radial, ax_radial = plt.subplots(2,3, #sharex='col', sharey='col'
                                     )
ax_radial = ax_radial.flatten()
for i_thermal, current_thermal in enumerate(list_of_thermals):
    path_to_decomposition = os.path.join(base_path, current_thermal,
                                         'decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed')

    dec = load_decomposition_data(path_to_decomposition, list_of_files=['thermal.csv', 'splines.yaml'])
    splines = dec['splines'][0]
    df_thermal = dec['thermal']
    df_thermal = df_thermal[df_thermal['in_hull'] & (~df_thermal['interpolated_thermal_Z'])].copy()
    # df_thermal = df_thermal[np.abs(df_thermal['curvature']) < 0.1]

    rho_bins = np.arange(0, np.nanpercentile(df_thermal['rho_bird_TC'], 90) + bin_size_rho, bin_size_rho)
    Z_bins = np.arange(np.nanpercentile(df_thermal['Z'], 10), np.nanpercentile(df_thermal['Z'], 90) + bin_size_z, bin_size_z)
    df_thermal['bin_index_rho'] = np.digitize(df_thermal['rho_bird_TC'],rho_bins) -1
    df_thermal['bin_index_Z'] = np.digitize(df_thermal['Z'],Z_bins) -1

    df_thermal = df_thermal[df_thermal['bin_index_rho'].between(0, df_thermal['bin_index_rho'].max() - 1)]
    df_thermal = df_thermal[df_thermal['bin_index_Z'].between(0, df_thermal['bin_index_Z'].max() - 1)]
    current_norm = copy.deepcopy(dict_of_norms[current_thermal])

    df_thermal_avg = df_thermal.groupby(['bin_index_rho','bin_index_Z']).agg(rho_bird_TC_avg=('rho_bird_TC', 'mean'),
                                                                             Z_avg=('Z', 'mean'),
                                                                             dZdT_thermal_ground_avg=('dZdT_thermal_ground', 'mean'),
                                                                             ).reset_index()

    # ax_radial[i_thermal].scatter(df_thermal_avg['rho_bird_TC_avg'], df_thermal_avg['Z_avg'], c=df_thermal_avg['dZdT_thermal_ground_avg'], alpha=1, )
    # z_bins = np.arange(df_thermal['Z'].min() + np.ptp(df_thermal['Z']) * 0.1, df_thermal['Z'].max() - np.ptp(df_thermal['Z']) * 0.1, )

    # ax_radial[i_thermal].hexbin(df_thermal['rho_bird_TC'], df_thermal['Z'], C=df_thermal['dZdT_thermal_ground'], alpha=1, gridsize=12)

    m_radial = ax_radial[i_thermal].tricontourf(df_thermal_avg['rho_bird_TC_avg'], df_thermal_avg['Z_avg'],
                                                df_thermal_avg['dZdT_thermal_ground_avg'],
                                                levels=30, norm=current_norm,
                                                cmap=my_cmap
                                                )
    ax_radial[i_thermal].set_title(current_thermal)
    plt.colorbar(ScalarMappable(norm=current_norm, cmap=my_cmap), ax=ax_radial[i_thermal])
    # print(popt)
# ax[i_ss + i_offset].set_ylim((-2,5))
# ax[i_ss + i_offset].set_ylim((-150,150))






#
#
# df_stats = pd.DataFrame(list_stats, columns=['thermal', 'col', 'mean', 'median', 'std', 'count', '5percentile', '95percentile'])
# df_stats.to_latex()
#
# df_vz_fluct = pd.DataFrame([['b010_0.1', 3.8, 2.21,  0.55 , 0.95, 0.88, 0.55],
#                             ['b023_0.1', 3.24, 2.88, 0.90 , 0.76, 0.72, 0.90],
#                             # ['b023_1.1', 3.20, 2.89, 0.36 , 0.61, 0.62, 0.36],
#                             ['b072_0.1', 2.30, 1.24, 0.44 , 0.89, 0.89, 0.44],
#                             ['b077_0.1', 1.14, 1.74, 0.35 , 0.69, 0.74, 0.35],
#                             ['b112_0.2', 5.73, 1.78, 0.42 , 0.88, 0.73, 0.42],
#                             ['b121_0.1', 4.76, 2.00, 0.47 , 0.78, 0.81, 0.47],],
#                            columns=['thermal', 'wind', 'vz', 'sigma_xyz','sigma_x', 'sigma_y', 'sigma_z'])
# df_vz_fluct['sigma_xy'] = np.sqrt(df_vz_fluct['sigma_x'] ** 2 + df_vz_fluct['sigma_y'] ** 2 )
# fig, ax = plt.subplots(5,2, sharex='col', sharey='row')
# for i_col, col in enumerate( ['sigma_xyz','sigma_x', 'sigma_y', 'sigma_xy', 'sigma_z']):
#     ax[i_col, 0].scatter(df_vz_fluct['vz'], df_vz_fluct[col])
#     ax[i_col, 1].scatter(df_vz_fluct['wind'], df_vz_fluct[col])
#     ax[i_col, 0].set_ylabel(col)
# ax[-1, 0].set_xlabel('vz')
# ax[-1, 1].set_xlabel('wind')