import os.path
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd



from calc.analysis.turbulence import get_single_sweep_autocorrelation
from object.air import ReconstructedAirVelocityField

save = True

pd.options.mode.chained_assignment = None
root_path = '/home/pedro/PycharmProjects/ThermalModelling'

 #* 0.6314318

grid_size = 2

list_of_sizes = [30]



air_component = 'air'
average_removed = False
base_path = Path(os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/same_flock_WL=6.0'))
glob_string = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'
save_folder = os.path.join(root_path, 'results/turbulence/same_flock_WL=6.0/mesoscale_newcenters', air_component)
save_filename = f'turbulence_correlation_{grid_size=}.csv'

if not average_removed:
    save_filename = save_filename.replace('.csv', '_with_average.csv')
if save:
    os.makedirs(save_folder, exist_ok=True)

if os.path.exists(os.path.join(save_folder, save_filename)):

    df_all_thermals = pd.read_csv(os.path.join(save_folder, save_filename), index_col=False)
    df_all_thermals_avg = pd.read_csv(os.path.join(save_folder, save_filename.replace('.csv', '_avg.csv')),
                                  index_col=False)
else:
    df_all_thermals = pd.DataFrame()
    df_all_thermals_avg = pd.DataFrame()
label_function = lambda pt: {'bin_z_size': pt.parents[3].name.split('=')[-1],
                             'thermal':    pt.parents[6].name}

for i_t, current_path in enumerate(base_path.glob(glob_string)):
    for datatype in ['dec', 'gt']:
        path_to_decomposition = str(current_path)

        label_dict = label_function(current_path)
        if ((not df_all_thermals_avg.empty)
                and (not df_all_thermals_avg[np.all(df_all_thermals_avg[sorted(label_dict.keys())].astype(str) == [str(label_dict[k])
                                                                                                                   for k in sorted(label_dict.keys())],
                                                    axis=1)
                                                    & (df_all_thermals_avg['datatype'] == 'gt')].empty)):
            print('SKIPPING - ', str(path_to_decomposition))
            continue

        if datatype == 'gt':
            path_to_decomposition = os.path.join(path_to_decomposition, 'ground_truth_reconstructed')

        print(path_to_decomposition)
        decomposed_avf = ReconstructedAirVelocityField.from_path(path_to_files=path_to_decomposition, max_extrapolated_distance=0)
        z_min = np.min(decomposed_avf.wind_spline['X'].x_min)
        z_max = np.max(decomposed_avf.wind_spline['X'].x_max)

        current_thermal_df = pd.DataFrame()

        for i_size, size in enumerate(list_of_sizes[:]):
            current_size_df = pd.DataFrame()
            lx, ly = 50, 50
            lz =  size

            for center_z in np.arange(z_min + 0.5 * lz, z_max - 1*lz, 2 * lz)[:]:
                center = np.array([0,0,center_z])

                c_r = get_single_sweep_autocorrelation(center, (lx, ly, lz), (lx, ly, 3 * lz), grid_size,
                                                       decomposed_avf.get_velocity,
                                                       average_removed=average_removed,
                                                       include=['thermal'] if air_component == 'thermal' else None,
                                                       relative_to_ground=False)
                if c_r is None:
                    continue
                current_center_df = pd.DataFrame.from_records(c_r)
                current_center_df['center_z'] = center_z
                current_size_df = pd.concat([ current_size_df, current_center_df])
            current_size_df['size'] = size
            current_size_df['datatype'] = datatype
            current_thermal_df = pd.concat([current_thermal_df, current_size_df])
        for k,v in label_dict.items():
            current_thermal_df[k] = v
        df_all_thermals = pd.concat([df_all_thermals, current_thermal_df])

        if save:
            df_all_thermals.to_csv(os.path.join(save_folder, save_filename), index=False)
        del decomposed_avf


    df_all_thermals_avg = df_all_thermals.groupby(list(sorted(label_dict.keys()))
                                                  + ['datatype', 'size', 'delta_R']
                                                  ).agg(**{f'{col}_{stat}': (col, stat)
                                                           for col in ['inner' ,'x' ,'y' ,'z']
                                                           for stat in ['mean', 'median', 'std', 'count']
                                                           }).reset_index()


    for i_comp, comp in enumerate(['inner', 'x', 'y', 'z']):
        df_all_thermals_avg[f'{comp}_sem'] = df_all_thermals_avg[f'{comp}_std'].values / np.sqrt(df_all_thermals_avg[f'{comp}_count'].values)
    if save:
        df_all_thermals_avg.to_csv(os.path.join(save_folder, save_filename.replace('.csv', '_avg.csv')),
                                        index=False)
