import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from misc.config import stork_dataset_renaming_dict, digital_twin_renaming_dict
from misc.constants import root_path
from object.air import ReconstructedAirVelocityField

save = True

base_path = Path(os.path.join(root_path, 'synthetic_data/from_atlasz/newdata/same_flock_WL=6.0'))
glob_string = '*/decomposition/individual_bins/bin_z_size=10/optimized/n_resamples=1000/final/reconstructed'


save_folder = os.path.join(root_path, f'results/air_velocity_field/same_flock_WL=6.0')
if save:
    os.makedirs(save_folder, exist_ok=True)


label_function = lambda pt: {'bin_z_size': pt.parents[3].name.split('=')[-1],
                             'thermal':    pt.parents[6].name}
list_of_stats = []
for i_t, current_path in enumerate(base_path.glob(glob_string)):

    label_dict = label_function(current_path)
    label_keys = sorted(label_dict.keys())
    label_values = [label_dict[l] for l in label_keys]

    for datatype in ['dec', 'gt']:
        path_to_decomposition = str(current_path)
        if datatype == 'gt':
            path_to_decomposition = os.path.join(path_to_decomposition, 'ground_truth_reconstructed')

        print(path_to_decomposition)
        avf = ReconstructedAirVelocityField.from_path(path_to_decomposition, max_extrapolated_distance=0)
        z_min, z_max = avf.wind_spline['X'].x_min, avf.wind_spline['X'].x_max
        z_array = np.arange(z_min, z_max + 10, 10)
        Z3 = np.zeros(shape=(z_array.size, 3))
        Z3[:,-1] = z_array
        current_wind = avf.get_velocity(Z3, include='wind')
        current_wind_norm = np.linalg.norm(current_wind, axis=-1)
        current_wind_angle = np.arctan2( current_wind[:, 1], current_wind[:,0],) * 180 / np.pi

        XYZ_meshgrid = np.meshgrid(np.arange(-40,40 + 9,10),
                                   np.arange(-40, 40 + 9, 10),
                                   z_array,
                                   indexing='ij')

        XYZ_meshgrid = np.stack(XYZ_meshgrid, axis=-1)

        current_thermal = avf.get_velocity(XYZ_meshgrid, include='thermal', relative_to_ground=False)
        current_stats = {'datatype':datatype,
                         'wind_norm_mean':  np.nanmean(current_wind_norm),
                         'wind_norm_std':   np.nanstd(current_wind_norm),
                         'wind_angle_mean': np.nanmean(current_wind_angle),
                         'wind_angle_std':  np.nanstd(current_wind_angle),
                         'thermal_mean_x':  np.nanmean(current_thermal[..., 0]),
                         'thermal_std_x':   np.nanstd(current_thermal[..., 0]),
                         'thermal_mean_y':  np.nanmean(current_thermal[..., 1]),
                         'thermal_std_y':   np.nanstd(current_thermal[..., 1]),
                         'thermal_mean_z':  np.nanmean(current_thermal[..., 2]),
                         'thermal_std_z':   np.nanstd(current_thermal[..., 2])}
        current_stats = label_dict | current_stats

        list_of_stats.append(current_stats)



df_stats = pd.DataFrame.from_records(list_of_stats, columns=['thermal', 'bin_z_size', 'datatype',
                                                             'wind_norm_mean', 'wind_norm_std',
                                                             'wind_angle_mean', 'wind_angle_std',
                                                             'thermal_mean_x', 'thermal_std_x',
                                                             'thermal_mean_y', 'thermal_std_y',
                                                             'thermal_mean_z', 'thermal_std_z']
                          )
df_stats['thermal'] = df_stats['thermal'].apply(lambda x: digital_twin_renaming_dict[int(x)])

df_stats.sort_values('thermal', inplace=True)
if save:
    df_stats.to_csv(os.path.join(save_folder, 'thermals_summary.csv'), index=False)