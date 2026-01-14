import os

save=False
import matplotlib
if save:
    matplotlib.use('Cairo')
    matplotlib.interactive(False)

from simulated_annealing.post_process import post_process_annealing_results, get_modes_from_multiple_sim_annealing_history, \
    get_correlations_between_sim_annealing_runs

#

# list_of_thermals = ['b010_0.1', 'b023_0.1', 'b023_1.1',
#                     'b072_0.1', 'b077_0.1',
#     'b112_0.2', 'b121_0.1'
# ]
list_of_birds = ['Balou', 'Betty', 'Bubbel', 'Conchito', 'Cookie', 'Crisps', 'Ekky',
       'Ella', 'Fanny', 'Fifi', 'Flummy', 'Frank', 'Hannibal', 'Kim',
       'Kristall', 'Mirabell', 'Muffine', 'Niclas', 'Ohnezahn', 'Peaches',
       'Redrunner', 'Ronja', 'Snowy', 'Tobi']

#loss_percentile = 2
list_of_percentiles = [2]
append=False
do_deviation=True
bootstrap_n_resamples=1000
permutation_resamples=1000
kernel_bandwidth=None
config_base = '/home/pedro/PycharmProjects/ThermalModelling/synthetic_data/from_atlasz/last/storks/config/individual_bins/bin_z_size=10'
#save_folder = os.path.join(config_base, 'resamples=1000')
save_folder = os.path.join('results/wing_loadings/storks', f'n_resamples={bootstrap_n_resamples}')

if save:

    os.makedirs(save_folder, exist_ok=True)
#save_folder = 'data_pedro/decomposition_ready/subthermals/config_extended/figures_scipy'

# config_file_dict = {(a.parent.name, a.parent.parent.name): os.path.join(config_root, a.parent.parent.name, a.parent.name, 'sa.yaml')
#                     for a in p.glob('0.9/b*/sa.yaml')}
config_file_dict = {}
config_file_dict[('b010', '0')] = os.path.join(config_base, 'b010_0.1')
config_file_dict[('b0230', '0')] = os.path.join(config_base, 'b023_0.1')
config_file_dict[('b0231', '1')] = os.path.join(config_base, 'b023_1.1')
config_file_dict[('b072', '0')] = os.path.join(config_base, 'b072_0.1')
config_file_dict[('b077', '0')] = os.path.join(config_base, 'b077_0.1')
config_file_dict[('b112', '0')] = os.path.join(config_base, 'b112_0.2')
config_file_dict[('b121', '0')] = os.path.join(config_base, 'b121_0.1')
config_file_dict[('b010_b023_b072_10', '0')] = os.path.join(config_base, 'b010_b023_b072')
config_file_dict[('b077_b112_b121_10', '0')] = os.path.join(config_base, 'b077_b112_b121')
#config_file_dict[('b010_b023_b072_20', '0')] = os.path.join(config_base,  'b010_b023_b072', 'sa.yaml')
#config_file_dict[('b077_b112_b121_20', '0')] = os.path.join(config_base,  'b077_b112_b121', 'sa.yaml')
#config_file_dict[('b010_b023_b072_30', '0')] = os.path.join(config_base,  'b010_b023_b072', 'sa.yaml')
#config_file_dict[('b077_b112_b121_30', '0')] = os.path.join(config_base,  'b077_b112_b121', 'sa.yaml')


(df_modes_all_percentiles,
 b) = get_modes_from_multiple_sim_annealing_history(config_file_dict, list_of_percentiles, list_of_birds=list_of_birds,
                                                    aggregated_save_folder=save_folder, save=save, append=append,
                                                    do_deviation=do_deviation, n_resamples=bootstrap_n_resamples,
                                                    kernel_bandwidth=kernel_bandwidth)
# ==================================================================================================================== #
# ==========================================         CORRELATIONS         ============================================ #
# ==================================================================================================================== #
df_corr_all_percentiles = get_correlations_between_sim_annealing_runs(df_modes_all_percentiles, permutation_resamples, save_folder,
                                                                      save=save)