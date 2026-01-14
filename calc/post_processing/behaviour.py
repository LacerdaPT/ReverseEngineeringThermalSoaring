import pandas as pd
import numpy as np
from scipy.special import rel_entr
from scipy.stats import gaussian_kde, kstest


def compare_curvature_histograms(df_concat, df_all_histograms, df_template_point, df_template_histograms,
                       list_of_cols, list_of_birds, list_of_EER, list_of_scales, bin_avgs):

    list_ks = []
    for i_col, col in enumerate(list_of_cols):
        for current_eer in list_of_EER:
            current_eer_gt = df_template_point.loc[df_template_point['EER_index'] == current_eer, col].values.astype(float)
            for scale in list_of_scales:
                current_scaled_eer_gt = scale * current_eer_gt
                current_kde_gt = gaussian_kde(current_scaled_eer_gt[current_scaled_eer_gt < 0.1])

                def my_cdf(high):
                    return current_kde_gt.integrate_box_1d(0, high)

                my_cdf = np.vectorize(my_cdf)

                for i_bird, current_bird in enumerate(list_of_birds):
                    print(col, current_eer, scale, current_bird)
                    current_df_bird_col = df_concat.loc[df_concat['bird_name'] == current_bird, col].values.astype(
                        float)
                    current_ks = kstest(current_df_bird_col[current_df_bird_col < 0.1], my_cdf)

                    list_ks.append([current_eer,
                                    current_bird, col, scale,
                                    current_ks.statistic,
                                    current_ks.pvalue,
                                    current_ks.statistic_location,
                                    current_ks.statistic_sign,
                                    ])

    df_ks = pd.DataFrame(list_ks, columns=['EER_index',  'bird_name', 'col', 'scale',
                                           'ks',
                                           'ks_pvalue',
                                           'ks_location',
                                           'ks_sign'])
    list_histgrams_metrics = []
    for i_col, col in enumerate(list_of_cols):
        for current_eer in list_of_EER:
            for scale in list_of_scales:
                current_resampled_hist_gt = df_template_histograms.loc[
                    (df_template_histograms['EER_index'] == current_eer)
                    & (df_template_histograms['col'] == col)
                    & (df_template_histograms['scale'] == scale), 'hist'].values.copy()
                for i_bird, current_bird in enumerate(list_of_birds):
                    print(col, current_eer, scale, current_bird)
                    current_bird_hist = df_all_histograms.loc[(df_all_histograms['bird_name'] == current_bird)
                                                              & (df_all_histograms['col'] == col), 'hist'].values.copy()
                    zerozero_mask = (current_bird_hist == 0) & (current_resampled_hist_gt == 0)
                    zerozero_mask = ~zerozero_mask

                    good_mask = zerozero_mask & (bin_avgs < 0.1)
                    chi2 = ((current_bird_hist[good_mask] - current_resampled_hist_gt[good_mask]) ** 2
                            / (current_bird_hist[good_mask] + current_resampled_hist_gt[good_mask])).sum()
                    relative_entropy_eer_bird = rel_entr(current_resampled_hist_gt[good_mask],
                                                         current_bird_hist[good_mask]).sum()
                    relative_entropy_bird_eer = rel_entr(current_bird_hist[good_mask],
                                                         current_resampled_hist_gt[good_mask]).sum()
                    list_histgrams_metrics.append([current_eer,
                                                   current_bird, col, scale,
                                                   chi2,
                                                   relative_entropy_eer_bird,
                                                   relative_entropy_bird_eer,
                                                   ])

    df_hist_metrics = pd.DataFrame(list_histgrams_metrics, columns=['EER_index', 'bird_name', 'col', 'scale',
                                                                    'chi2',
                                                                    'relative_entropy_eer_bird',
                                                                    'relative_entropy_bird_eer',
                                                                    ])

    df_ks = pd.merge(df_ks, df_hist_metrics, on=['EER_index', 'bird_name', 'col', 'scale'])

    return df_ks
