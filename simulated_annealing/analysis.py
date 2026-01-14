from itertools import product

import pandas as pd
from scipy import stats


def get_pearson_correlations(df, merge_var, correlation_var, partition_key, partition_list=None, **kwargs ):
    if partition_list is None:
        partition_list = df[partition_key].unique()
    df_corr = []

    for (i_t1, p1) in enumerate(partition_list):
        for (i_t2, p2) in enumerate(partition_list[i_t1 + 1:]):
        #cut_off = np.percentile(df_modes[df_modes['thermal'].isin([t1,t2])]['WL_mode_std'].values, 100)

            df_p1 = df[df[partition_key] == p1]
            df_p2 = df[df[partition_key] == p2]
            df_merge = pd.merge(df_p1, df_p2, on=merge_var, how='inner')

            #df_merge.loc[:, 'is_good'] = (df_merge['WL_mode_std_x'] < cut_off) & (df_merge['WL_mode_std_y'] < cut_off)
            if df_merge.empty:
                continue
            if isinstance(correlation_var, (list)):
                current_corr = stats.pearsonr(df_merge[correlation_var[0]],
                                              df_merge[correlation_var[1]],
                                              **kwargs)
            else:
                current_corr = stats.pearsonr(df_merge[f'{correlation_var}_x'],
                                              df_merge[f'{correlation_var}_y'],
                                              **kwargs)

            df_corr.append({f'{partition_key}_1': p1,
                            f'{partition_key}_2': p2,
                            'pearson_r': current_corr.statistic,
                            'pvalue':current_corr.pvalue,
                            'CI_low':current_corr.confidence_interval(0.95).low,
                            'CI_high':current_corr.confidence_interval(0.95).high
            })
    df_corr = pd.DataFrame.from_dict(df_corr, orient='columns')
    return df_corr