
import os
import numpy as np
import pandas as pd
save = True
figsize_multiplier = 2
from misc.config import science_matplotlib_config
science_matplotlib_config(figsize_multiplier, save=save)
from matplotlib import pyplot as plt

column_renaming_dict= {'X_bird_TC':           '$X_\\mathrm{Glider,TC}$',
                       'Y_bird_TC':           '$Y_\\mathrm{Glider,TC}$',
                       'dXdT_bird_ground':    '$v_\\mathrm{Glider;x}$',
                       'dYdT_bird_ground':    '$v_\\mathrm{Glider;y}$',
                       'dZdT_bird_ground':    '$v_\\mathrm{Glider;z}$',
                       'dXdT_bird_air':       '$v_\\mathrm{Glider,Air;x}$',
                       'dYdT_bird_air':       '$v_\\mathrm{Glider,Air;y}$',
                       'dZdT_bird_air':       '$v_\\mathrm{Glider,Air;z}$',
                       'curvature_bird_air':  '$\\mathrm{Curvature}, \\kappa$',
                       'bank_angle_bird_air': '$\\mathrm{Bank\\ Angle}, \\phi$',
                       'X_TC_ground':         '$X_\\mathrm{TC,Ground}$',
                       'Y_TC_ground':         '$Y_\\mathrm{TC,Ground}$',
                       'wind_X':              '$v_{\\mathrm{Wind};x}$',
                       'wind_Y':              '$v_{\\mathrm{Wind};y}$',
                       'dXdT_thermal_ground': '$v_{\\mathrm{Thermal};x}$',
                       'dYdT_thermal_ground': '$v_{\\mathrm{Thermal};y}$',
                       'dZdT_thermal_ground': '$v_{\\mathrm{Thermal};z}$',
                       'dXdT_air_ground':     '$v_{\\mathrm{Air};x}$',
                       'dYdT_air_ground':     '$v_{\\mathrm{Air};y}$',
                       'dZdT_air_ground':     '$v_{\\mathrm{Air};z}$',
 }
sweep_var_renaming_dict = {'WL_avg': '$<W^L>\\ \\ (\\mathrm{kg~m}^{-2})$'}

metric_renaming_dict = {'pearson_r': '$\\mathrm{Pearson\'s}~r$',
                        'RMS': '$\\mathrm{RMSE}$'}

root = '/home/pedro/PycharmProjects/ThermalModelling'
base_path = 'synthetic_data/from_atlasz/newdata/WL_sweep_avg_6_std_06'
fig_base_title = '_'.join(base_path.split('/')[2:])
base_path = os.path.join(root, base_path)
save_path = os.path.join(base_path,'results')
metric = 'RMS' # , 'pearson_r', 'pearson_p_value'
sweep_var = 'WL_avg'
df_results = pd.read_csv(os.path.join(base_path, 'results', 'sweep_annealing_result_post.csv'), index_col=False)
df_correlations = pd.read_csv(os.path.join(base_path, 'results', 'sweep_correlations_post.csv'), index_col=False)
#df_correlations = df_correlations[df_correlations['turbulence'] == 0.3]


mosaic_matrix = [['X_bird_TC', 'Y_bird_TC', 'bank_angle_bird_air'],
                 ['X_TC_ground', 'Y_TC_ground', 'curvature_bird_air'],
                 ['wind_X', 'wind_Y', 'dZdT_thermal_ground'],
                 ['dXdT_bird_air', 'dYdT_bird_air', 'dZdT_bird_air']]



df_correlations[sweep_var] = round(df_correlations[sweep_var], 2)
df_correlations = df_correlations[df_correlations['dec_col'] != 'radius']
df_correlations_avg = df_correlations.groupby([sweep_var, 'dec_col']).agg(RMS_avg=('RMS', 'mean'),
                                                                          pearson_r_avg=('pearson_r', 'mean'),
                                                                          pearson_p_value_avg=('pearson_p_value', 'mean'),
                                                                          RMS_std=('RMS', 'std'),
                                                                          pearson_r_std=('pearson_r', 'std'),
                                                                          pearson_p_value_std=('pearson_p_value', 'std')).reset_index()

list_of_cols = df_correlations['dec_col'].unique()
list_of_cols = ['X_bird_TC', 'Y_bird_TC', 'dXdT_bird_ground', 'dYdT_bird_ground', 'dZdT_bird_ground',
                'bank_angle', 'curvature', 'dXdT_bird_air', 'dYdT_bird_air', 'dZdT_bird_air',
                'X_TC_ground', 'Y_TC_ground', #'dXdT_thermal_ground_avg', 'dYdT_thermal_ground_avg', 'dZdT_thermal_ground_avg',
                'wind_X', 'wind_Y', #'dXdT_thermal_TC_res', 'dYdT_thermal_TC_res', 'dZdT_thermal_TC_res',
                'dXdT_air_ground', 'dYdT_air_ground', 'dZdT_air_ground']

if save:
    os.makedirs(os.path.join(base_path, 'results', 'figures', 'png'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'results', 'figures', 'svg'), exist_ok=True)
# fig_rms, ax_rms = plt.subplots(4, 5,figsize=(15, 12), constrained_layout=True)
# ax_rms = ax_rms.flatten()
# for i_row, dec_col in enumerate(list_of_cols[:len(ax_rms)]):
#     if dec_col == 'radius':
#         continue
#     current_col = df_correlations[df_correlations['dec_col'] == f'{dec_col}_dec']
#     current_col_avg = df_correlations_avg[df_correlations_avg['dec_col'] == f'{dec_col}_dec']
#     current_axis_correlation = ax_rms[i_row]
#
#
#     current_axis_correlation.scatter(current_col[sweep_var], current_col[metric], s=5)
#     current_axis_correlation.errorbar(current_col_avg[sweep_var], current_col_avg[f'{metric}_avg'], current_col_avg[f'{metric}_std'], fmt='o--', c='r')
#     current_axis_correlation.set_xlabel(sweep_var)
#     current_axis_correlation.set_ylabel(metric)
#     current_axis_correlation.set_title(f'{column_renaming_dict[dec_col]}')
# if save:
#    fig_rms.suptitle(f'{fig_base_title}         RMS')
#    fig_rms.savefig(os.path.join(save_path, 'figures', 'png', f'{fig_base_title}_RMS.png'))
#    fig_rms.savefig(os.path.join(save_path, 'figures', 'svg', f'{fig_base_title}_RMS.svg'))
#    plt.close(fig_rms)
metric = 'pearson_r'


fig_correlation, ax_correlation = plt.subplot_mosaic(mosaic_matrix, layout='constrained',
                                                     sharex=True,
                                                     # sharey=True,
                                                     figsize=(figsize_multiplier*4.75, figsize_multiplier*4.75), )
# ax_correlation = ax_correlation.flatten()
for i_row, dec_col in enumerate(np.array(mosaic_matrix).flatten()):
    current_col = df_correlations[df_correlations['dec_col'] == f'{dec_col}_dec']
    current_col_avg = df_correlations_avg[df_correlations_avg['dec_col'] == f'{dec_col}_dec']
    current_axis_correlation = ax_correlation[dec_col]


    current_axis_correlation.scatter(current_col[sweep_var], current_col[metric], s=figsize_multiplier*5)
    current_axis_correlation.errorbar(current_col_avg[sweep_var], current_col_avg[f'{metric}_avg'], current_col_avg[f'{metric}_std'], fmt='o--', c='r')
    current_axis_correlation.set_ylabel(metric_renaming_dict[metric])
    current_axis_correlation.set_title(f'{column_renaming_dict[dec_col]}')

[ax_correlation[my_col].set_xlabel(sweep_var_renaming_dict[sweep_var]) for my_col in mosaic_matrix[-1]]
# fig_correlation.suptitle(f'{fig_base_title}         pearson_r')
if save:

    fig_correlation.savefig(os.path.join(save_path, 'figures', 'png',  f'{fig_base_title}_pearson_r.png'))
    fig_correlation.savefig(os.path.join(save_path, 'figures', 'svg',  f'{fig_base_title}_pearson_r.svg'))

    plt.close(fig_correlation)
else:
    plt.show(block=True)
