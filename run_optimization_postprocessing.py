import os
from argparse import ArgumentParser

import pandas as pd
import yaml
from scipy.stats import PermutationMethod, pearsonr

from data.get_data import load_synthetic_data
from simulated_annealing.post_process import post_process_annealing_results



parser = ArgumentParser()
parser.add_argument('--config-file', dest='config_file', type=str)
parser.add_argument('--output-folder', dest='save_folder', type=str, default=None)
parser.add_argument('--loss-percentile', dest='loss_percentile', type=float, nargs='+', default=2)
parser.add_argument('--n-resamples', dest='n_resamples', type=int, default=9999)
parser.add_argument('--kernel-bandwidth', dest='kernel_bandwidth', type=str, default=None)
parser.add_argument('--not-synthetic', dest='is_synthetic', action='store_false')

def main():
    args = parser.parse_args()
    config_file = args.config_file
    list_of_loss_percentile = args.loss_percentile
    n_resamples = args.n_resamples
    kernel_bandwidth = args.kernel_bandwidth
    save_folder = args.save_folder
    is_synthetic=args.is_synthetic
    if save_folder is None:
        save_folder = os.path.dirname(config_file)
    os.makedirs(save_folder, exist_ok=True)

    if kernel_bandwidth is not None and kernel_bandwidth.isdigit():
        kernel_bandwidth = float(kernel_bandwidth)

    df_modes_all_percentiles = pd.DataFrame()
    for loss_percentile in list_of_loss_percentile:
        df_history, (current_df_modes,
                 kde_dict) = post_process_annealing_results(config_file,
                                                            loss_percentile,
                                                            candidate=True,
                                                            do_deviation=True,
                                                            n_resamples=n_resamples,
                                                            kernel_bandwidth=kernel_bandwidth)

        current_df_modes['config_file'] = config_file
        current_df_modes['loss_percentile'] = loss_percentile

        df_modes_all_percentiles = pd.concat([df_modes_all_percentiles, current_df_modes])

    if is_synthetic:
        with open(config_file, 'r') as f:
            sa_yaml = yaml.safe_load(f)
        with open(sa_yaml['decomposition_parameters']['file'][0], 'r') as f:
            path_to_synthetic = yaml.safe_load(f)['run_parameters']['input_folder']
        synthetic_data_dict = load_synthetic_data(path_to_synthetic)
        bird_parameters = synthetic_data_dict['bird_parameters']
        del synthetic_data_dict
        bird_parameters = [[d['bird_name'], d['physical_parameters']['mass'] / d['physical_parameters']['wing_area']] for d
                           in
                           bird_parameters]
        df_bird_parameters = pd.DataFrame(bird_parameters, columns=['bird_name', 'WL_GT'])
        df_modes_all_percentiles = pd.merge(df_modes_all_percentiles, df_bird_parameters, on='bird_name')
        df_corr_real_all_percentiles = []
        for loss_percentile in list_of_loss_percentile:
            current_percentile_df = df_modes_all_percentiles[df_modes_all_percentiles['loss_percentile'] == loss_percentile]

            pearson_result = pearsonr(current_percentile_df['WL_mode'],
                                      current_percentile_df['WL_GT'],
                                      method=PermutationMethod(n_resamples=n_resamples, batch=n_resamples))
            df_corr_real_all_percentiles.append({'loss_percentile': loss_percentile,
                                                 'pearson_r':       pearson_result.statistic,
                                                 'pvalue':          pearson_result.pvalue,
                                                 'CI_low':          pearson_result.confidence_interval(0.95).low,
                                                 'CI_high':         pearson_result.confidence_interval(0.95).high
                                                 })
        df_corr_real_all_percentiles = pd.DataFrame.from_records(df_corr_real_all_percentiles)

        df_corr_real_all_percentiles.to_csv(os.path.join(save_folder, f'pearson_correlations_ground_truth.csv'))

    df_modes_all_percentiles.to_csv(os.path.join(save_folder, 'wing_loading_modes.csv'), index=False)

if __name__ == '__main__':
    main()