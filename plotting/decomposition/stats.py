import numpy as np
import pandas as pd

from decomposition.auxiliar import get_relative_change


def plot_convergence(df, ax, list_of_cols=None, merge_on=None, plot_kwargs=None):
    if merge_on is None:
        merge_on = ['bird_name', 'time']
    if plot_kwargs is None:
        plot_kwargs = {}
    if list_of_cols is None:
        list_of_cols = ['dXdT_bird_4',
                       'dYdT_bird_4',
                       'dZdT_bird_4',
                       'dXdT_air_4',
                       'dYdT_air_4',
                       'dZdT_air_4',
                       'wind_X',
                       'wind_Y']

    df_diff = pd.DataFrame(columns=list_of_cols + ['iteration'])
    for i in range(2, 50):
        current_diff = {}
        current_iteration = df[df['iteration'] == i]
        if current_iteration.empty:
            break
        previous_iteration = df[df['iteration'] == i - 1]
        current_iteration = current_iteration[np.abs(current_iteration['curvature']) < 0.1]
        previous_iteration = previous_iteration[np.abs(previous_iteration['curvature']) < 0.1]
        print(i)
        for col in list_of_cols:
            current_diff[col] = np.mean(np.linalg.norm(get_relative_change(current_iteration, previous_iteration,
                                           list_of_cols=[col], merge_on=merge_on), axis=1))

        current_diff['average'] = np.mean(np.linalg.norm(get_relative_change(current_iteration, previous_iteration,
                                                                             list_of_cols=list_of_cols,
                                                                             merge_on=merge_on),
                                                         axis=1)
                                          )
        current_diff['iteration'] = i
        df_diff = pd.concat([df_diff, pd.DataFrame.from_dict(current_diff, orient='index').T], ignore_index=True)

    for i_col, col in enumerate(list_of_cols + ['average']):
        ax[i_col].plot(df_diff['iteration'], df_diff[col], **plot_kwargs)
        ax[i_col].semilogy()
        ax[i_col].set_title(col)

    return df_diff, ax

def plot_convergence_comparative(df, df_real, ax, col_mapping=None, merge_on=None, plot_kwargs=None):
    if merge_on is None:
        merge_on = ['bird_name', 'time']
    if plot_kwargs is None:
        plot_kwargs = {}

    if col_mapping is None:
        col_mapping = {'dXdT_bird_real': 'dXdT_bird_4',
                       'dYdT_bird_real': 'dYdT_bird_4',
                       'dZdT_bird_real': 'dZdT_bird_4',
                       'dXdT_air_real':  'dXdT_air_4',
                       'dYdT_air_real':  'dYdT_air_4',
                       'dZdT_air_real':  'dZdT_air_4',
                       'wind_X_real':    'wind_X',
                       'wind_Y_real':    'wind_Y', }
    df = pd.merge(df, df_real[merge_on + list(col_mapping.keys())], on=merge_on, how='left')

    for syn_col, col in col_mapping.items():
        df[f'diff_{col}'] = df[syn_col] - df[col]

    df_diff = pd.DataFrame(columns=list(col_mapping.values())  + ['iteration'])

    for i in range(2, 50):
        current_diff = {}
        current_iteration = df[df['iteration'] == i]
        if current_iteration.empty:
            break
        for col in col_mapping.values():
            current_diff[col] = np.nanmedian(np.abs(current_iteration[f'diff_{col}'].values))

        current_diff['average'] = np.nanmedian(np.abs(current_iteration[[f'diff_{col}' for col in col_mapping.values()]].values))
        current_diff['iteration'] = i
        df_diff = pd.concat([df_diff, pd.DataFrame.from_dict(current_diff, orient='index').T], ignore_index=True)

    for i_col, col in enumerate(list(col_mapping.values()) + ['average']):
        ax[i_col].plot(df_diff['iteration'], df_diff[col], **plot_kwargs)
        ax[i_col].semilogy()
        ax[i_col].set_title(col)

    return df_diff, ax
