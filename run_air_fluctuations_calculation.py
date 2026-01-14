import os.path
from argparse import ArgumentParser
import numpy as np
import pandas as pd


from calc.analysis.turbulence import get_local_fluctuations

parser = ArgumentParser()
parser.add_argument('-i', '--input-folder', dest='path_to_decomposition', type=str)
parser.add_argument('-o', '--output-folder', dest='save_folder', type=str, default=None)
parser.add_argument('-r', '--radii', dest='list_of_radii', nargs="+", type=float, default=[15])
parser.add_argument('-ac', '--air-component', dest='air_component', choices = ['air', 'thermal'], default='air')
parser.add_argument('-mo', '--min-occupation', dest='min_occupation', type=int, default=3)

parser.add_argument('-ns', '--not-synthetic', action='store_false', dest='is_synthetic')
parser.add_argument('-dr', '--dry-run', action='store_false', dest='save')


def main():
    args=parser.parse_args()
    path_to_decomposition = args.path_to_decomposition
    save_folder = args.save_folder
    list_of_radii = args.list_of_radii
    is_synthetic = args.is_synthetic
    air_component = args.air_component
    min_occupation = args.min_occupation
    save = args.save
    if save_folder is None:
        save_folder = os.path.join(path_to_decomposition, 'results', 'fluctuations', f'{air_component}')
    if save:
        os.makedirs(save_folder, exist_ok=True)

    list_of_datatypes = ['dec', 'gt'] if is_synthetic else ['dec']
    df_fluctuations = pd.DataFrame()
    df_stats = pd.DataFrame()
    v_cols = [f'dXdT_{air_component}_ground', f'dYdT_{air_component}_ground', f'dZdT_{air_component}_ground']
    for datatype in list_of_datatypes:

        if datatype == 'gt':
            path_to_decomposition = os.path.join(path_to_decomposition, 'ground_truth_reconstructed')

        print(path_to_decomposition)

        df = pd.read_csv(os.path.join(path_to_decomposition, 'thermal.csv'), index_col=False)
        df = df[(~np.any(df[['interpolated_thermal_X', 'interpolated_thermal_Y', 'interpolated_thermal_Z']], axis=1))]
        if 'in_hull' in df.columns:
            df = df[df['in_hull']]
        my_data = df[
            ['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC'] + v_cols]

        (current_individual_stats,
         current_fluctuations) = get_local_fluctuations(my_data, list_of_radii=list_of_radii, v_cols=v_cols,
                                                        min_occupation_number=min_occupation)
        current_individual_stats = pd.DataFrame(current_individual_stats,
                                                 columns=['radius', 'center_x', 'center_y', 'center_z',
                                                          'avg_x', 'avg_y', 'avg_z', 'avg_xyz',
                                                          'sigma_x', 'sigma_y', 'sigma_z', 'sigma_xyz', 'count'])
        current_individual_stats['datatype'] = datatype
        current_fluctuations['datatype'] = datatype
        df_stats = pd.concat([df_stats, current_individual_stats])
        df_fluctuations= pd.concat([df_fluctuations, current_fluctuations])

    df_sigmas = df_fluctuations.groupby(['datatype', 'radius']).std().reset_index()

    if save:
        df_fluctuations.to_csv(os.path.join(save_folder, f'fluctuations.csv'), index=False)
        df_stats.to_csv(os.path.join(save_folder, f'stats_per_radius_per_sphere.csv'), index=False)

        df_sigmas.to_csv(os.path.join(save_folder, f'turbulence_sigmas.csv'), index=False)

if __name__ == '__main__':
    main()