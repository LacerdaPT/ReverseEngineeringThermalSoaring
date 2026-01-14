from copy import deepcopy
from typing import Iterable, List

import numpy as np
import pandas as pd
from numba import njit
from scipy import signal


def my_correlation(sample, volume, average_removed):
    n_points = np.sum((~np.any(np.isnan(sample),
                                       axis=-1))
                              & (~np.any(np.isnan(volume),
                                         axis=-1))
                              )
    if n_points == 0:
        return np.array( [np.nan] * 3)

    return_value = np.zeros(shape=(3,))
    for i_coord in range(3):
        if average_removed:
            current_volume = volume - np.nanmean(volume, axis=tuple(i for i in range(len(volume.shape) - 1)))
            current_sample = sample - np.nanmean(sample, axis=tuple(i for i in range(len(sample.shape) - 1)))
        else:
            current_volume = volume
            current_sample = sample

        current_volume_unit = current_volume / np.linalg.norm(current_volume, axis=-1
                                                              ).reshape(current_volume.shape[:-1] + (1,))
        current_sample_unit = current_sample / np.linalg.norm(current_sample, axis=-1
                                                              ).reshape(current_sample.shape[:-1] + (1,))

        current_volume_unit[np.isnan(current_volume_unit)] = 0
        current_sample_unit[np.isnan(current_sample_unit)] = 0

        return_value[i_coord] = signal.correlate(current_volume_unit[..., i_coord],
                                                 current_sample_unit[..., i_coord],
                                                 mode='valid') / n_points
    return return_value


def get_single_sweep_autocorrelation(center, l_sample, l_volume, grid_size, velocity_function, average_removed=False, **velocity_kwargs):
    lx_sample, ly_sample, lz_sample = l_sample
    lx_volume, ly_volume, lz_volume = l_volume

    xyz_mg_sample = np.stack(np.meshgrid(np.arange(center[0] - lx_sample / 2, center[0] + lx_sample / 2 + 0.1*grid_size, grid_size),
                             np.arange(center[1] - ly_sample / 2, center[1] + ly_sample / 2 + 0.1*grid_size, grid_size),
                             np.arange(center[2] - lz_sample / 2, center[2] + lz_sample / 2 + 0.1*grid_size, grid_size),
                             indexing='ij'), axis=-1)

    xyz_mg_volume = np.stack(np.meshgrid(np.arange(center[0] - lx_volume / 2, center[0] + lx_volume / 2 + 0.1 * grid_size, grid_size),
                             np.arange(center[1] - ly_volume / 2, center[1] + ly_volume / 2 + 0.1 * grid_size, grid_size),
                             np.arange(center[2] - lz_volume / 2, center[2] + lz_volume / 2 + 0.1 * grid_size, grid_size),
                             indexing='ij'), axis=-1)
    # v_thermal_gt = ground_truth_avf.get_velocity(xyz_mg, include=['thermal', 'turbulence'], relative_to_ground=False)
    # v_thermal_gt[np.isnan(v_thermal_gt)] = 0

    v_thermal_sample = velocity_function(xyz_mg_sample, **velocity_kwargs)
    v_thermal_volume = velocity_function(xyz_mg_volume, **velocity_kwargs)
    # v_thermal_sample_mean = np.nanmean(v_thermal_sample, axis=tuple(i for i in range(len(v_thermal_sample.shape) - 1)))
    # v_thermal_volume_mean = np.nanmean(v_thermal_volume, axis=tuple(i for i in range(len(v_thermal_volume.shape) - 1)))
    del xyz_mg_sample, xyz_mg_volume
    if np.all(np.isnan(v_thermal_sample)) or np.all(np.isnan(v_thermal_volume)):
        return
    convolution_shape = np.array(v_thermal_volume.shape[:-1]) - np.array(v_thermal_sample.shape[:-1]) + np.array([1, 1, 1])
    sample_shape = v_thermal_sample[..., 0].shape

    convolution_array = np.empty(shape=tuple(convolution_shape) + (3,))
    for i0 in range(0, convolution_shape[0]):
        for i1 in range(0, convolution_shape[1]):
            for i2 in range(0, convolution_shape[2]):
                current_volume = v_thermal_volume[i0: i0 + sample_shape[0], i1: i1 + sample_shape[1], i2: i2 + sample_shape[2]].copy()
                current_sample = v_thermal_sample.copy()
                convolution_array[i0, i1, i2] = my_correlation(current_sample,
                                                               current_volume,
                                                               average_removed=average_removed)

    #
    # v_volume_unit = v_thermal_volume / np.linalg.norm(v_thermal_volume, axis=-1).reshape(v_thermal_volume.shape[:-1] + (1,))
    # v_sample_unit = v_thermal_sample / np.linalg.norm(v_thermal_sample, axis=-1).reshape(v_thermal_sample.shape[:-1] + (1,))
    # del v_thermal_sample #v_thermal_sample[np.isnan(v_thermal_sample)] = 0
    # del v_thermal_volume #v_thermal_volume[np.isnan(v_thermal_volume)] = 0
    # v_volume_unit[np.isnan(v_volume_unit)] = 0
    # v_sample_unit[np.isnan(v_sample_unit)] = 0

    # for i in range(3):
    #     v_thermal_sample[...,i] = v_thermal_sample[...,i] - v_thermal_sample_mean[i]
    #     v_thermal_volume[...,i] = v_thermal_volume[...,i] - v_thermal_volume_mean[i]

    cc = {coord: convolution_array[..., i_coord] for i_coord, coord in enumerate(['x', 'y','z'])}
    #
    # my_inner = np.sum(np.stack([cc[coord] for i, coord in enumerate(['x', 'y','z'])],
    #                            axis=-1),
    #                   axis=-1)

    cc['inner'] = np.nansum(convolution_array, axis=-1)

    delta_R_mg = np.linalg.norm(np.stack(np.meshgrid(np.arange(-(lx_volume - lx_sample) / 2,
                                                               (lx_volume - lx_sample) / 2  + 0.1 * grid_size,
                                                               grid_size) ,
                                                     np.arange(-(ly_volume - ly_sample) / 2,
                                                               (ly_volume - ly_sample) / 2  + 0.1 * grid_size,
                                                               grid_size) ,
                                                     np.arange(-(lz_volume - lz_sample) / 2,
                                                               (lz_volume - lz_sample) / 2  + 0.1 * grid_size,
                                                               grid_size) ,
                                                     indexing='ij'),
                                         axis=-1),
                                axis=-1)
    delta_R = grid_size
    c_r = {k: [] for k in cc.keys()}
    c_r['delta_R'] = []
    for current_r in np.arange(0, delta_R_mg.max(), delta_R):
        current_mask = (current_r <= delta_R_mg) & (delta_R_mg < current_r + delta_R)
        if np.all(~current_mask):
            continue
        else:
            for k in c_r.keys():
                if k == 'delta_R':
                    c_r['delta_R'].append(np.nanmean([delta_R_mg[current_mask]]))
                else:
                    c_r[k].append(np.nanmean([cc[k][current_mask]]))

    return c_r


def get_air_autocorrelation(avf, lz, grid_size=2, lx= 50, ly = 50, average_removed=False,
                            air_component='thermal'):
    z_min = np.min(avf.wind_spline['X'].x_min)
    z_max = np.max(avf.wind_spline['X'].x_max)

    df_all_centers = pd.DataFrame()

    for center_z in np.arange(z_min + 0.5 * lz, z_max - 1 * lz, 3 * lz)[:]:
        center = np.array([0, 0, center_z])

        c_r = get_single_sweep_autocorrelation(center, (lx, ly, lz), (lx, ly, 3 * lz), grid_size,
                                               avf.get_velocity,
                                               average_removed=average_removed,
                                               include=['thermal'] if air_component == 'thermal' else None,
                                               relative_to_ground=False)
        if c_r is None:
            continue
        current_center_df = pd.DataFrame.from_records(c_r)
        current_center_df['center_z'] = center_z
        df_all_centers = pd.concat([df_all_centers, current_center_df])
    df_avg = df_all_centers.groupby([ 'delta_R']).agg(**{f'{col}_{stat}': (col, stat)
                                                         for col in ['inner' ,'x' ,'y' ,'z']
                                                         for stat in ['mean', 'median', 'std', 'count']
                                                         }).reset_index()
    return df_avg, df_all_centers


def get_single_sweep_crosscorrelation(center, l_sample, l_volume, grid_size, velocity_function, velocity_kwargs):
    lx_sample, ly_sample, lz_sample = l_sample
    lx_volume, ly_volume, lz_volume = l_volume

    mgs_sample = np.meshgrid(np.arange(center[0] - lx_sample / 2, center[0] + lx_sample / 2 + 0.1*grid_size, grid_size),
                             np.arange(center[1] - ly_sample / 2, center[1] + ly_sample / 2 + 0.1*grid_size, grid_size),
                             np.arange(center[2] - lz_sample / 2, center[2] + lz_sample / 2 + 0.1*grid_size, grid_size),
                             indexing='ij')

    mgs_volume = np.meshgrid(np.arange(center[0] - lx_volume / 2, center[0] + lx_volume / 2 + 0.1 * grid_size, grid_size),
                             np.arange(center[1] - ly_volume / 2, center[1] + ly_volume / 2 + 0.1 * grid_size, grid_size),
                             np.arange(center[2] - lz_volume / 2, center[2] + lz_volume / 2 + 0.1 * grid_size, grid_size),
                             indexing='ij')
    xyz_mg_sample = np.stack(mgs_sample, axis=-1)
    xyz_mg_volume = np.stack(mgs_volume, axis=-1)
    # v_thermal_gt = ground_truth_avf.get_velocity(xyz_mg, include=['thermal', 'turbulence'], relative_to_ground=False)
    # v_thermal_gt[np.isnan(v_thermal_gt)] = 0

    v_thermal_sample = velocity_function[0](xyz_mg_sample, **(velocity_kwargs[0]))
    v_thermal_volume = velocity_function[1](xyz_mg_volume, **(velocity_kwargs[1]))
    if np.all(np.isnan(v_thermal_sample)) or np.all(np.isnan(v_thermal_volume)):
        return


    convolution_shape = np.array(v_thermal_volume.shape[:-1]) - np.array(v_thermal_sample.shape[:-1]) + np.array([1, 1, 1])
    sample_shape = v_thermal_sample[..., 0].shape
    convolution_n_points = np.empty(shape=convolution_shape)
    for i0 in range(0, convolution_shape[0]):
        for i1 in range(0, convolution_shape[1]):
            for i2 in range(0, convolution_shape[2]):
                current_volume = v_thermal_volume[i0: i0 + sample_shape[0], i1: i1 + sample_shape[1], i2: i2 + sample_shape[2]]
                current_sample = v_thermal_sample[...]
                n_points = np.sum((~np.any(np.isnan(current_sample),
                                           axis=-1))
                                  & (~np.any(np.isnan(current_volume),
                                             axis=-1))
                                  )
                convolution_n_points[i0, i1, i2] = n_points

    v_volume_unit = v_thermal_volume / np.linalg.norm(v_thermal_volume, axis=-1).reshape(v_thermal_volume.shape[:-1] + (1,))
    v_sample_unit = v_thermal_sample / np.linalg.norm(v_thermal_sample, axis=-1).reshape(v_thermal_sample.shape[:-1] + (1,))
    v_thermal_sample[np.isnan(v_thermal_sample)] = 0
    v_thermal_volume[np.isnan(v_thermal_volume)] = 0
    v_volume_unit[np.isnan(v_volume_unit)] = 0
    v_sample_unit[np.isnan(v_sample_unit)] = 0


    cc = {}
    for i, coord in enumerate(['x', 'y','z']):
        cc[coord] = signal.correlate(v_volume_unit[..., i],
                                     v_sample_unit[..., i],
                                     mode='valid'
                                     )
    my_inner = np.sum(np.stack([cc[coord] for i, coord in enumerate(['x', 'y','z'])],
                               axis=-1),
                      axis=-1)

    cc['inner'] = my_inner
    for i, coord in enumerate(['x', 'y','z', 'inner']):
        cc[coord][convolution_n_points != 0 ] = (cc[coord][convolution_n_points != 0 ]
                                                / convolution_n_points[convolution_n_points != 0 ])
        cc[coord][convolution_n_points == 0] = 0


    delta_mg = np.meshgrid(np.arange(-(lx_volume - lx_sample) / 2, (lx_volume - lx_sample) / 2  + 0.1 * grid_size, grid_size) ,
                           np.arange(-(ly_volume - ly_sample) / 2, (ly_volume - ly_sample) / 2  + 0.1 * grid_size, grid_size) ,
                           np.arange(-(lz_volume - lz_sample) / 2, (lz_volume - lz_sample) / 2  + 0.1 * grid_size, grid_size) ,
                           indexing='ij')
    delta_R_mg = np.linalg.norm(np.stack(delta_mg, axis=-1), axis=-1)
    delta_R = 1
    c_r = {k: [] for k in cc.keys()}
    c_r['delta_R'] = []
    for current_r in np.arange(0, delta_R_mg.max(), delta_R):
        current_mask = (current_r <= delta_R_mg) & (delta_R_mg < current_r + delta_R)
        if np.all(~current_mask):
            continue
        else:
            for k in c_r.keys():
                if k == 'delta_R':
                    c_r['delta_R'].append(np.mean(delta_R_mg[current_mask]))
                else:
                    c_r[k].append(np.mean(cc[k][current_mask]))

    return c_r


def get_fluctuations_per_center(df: pd.DataFrame, center: Iterable, v_cols: List[str], radius: float, min_occupation_number: int ):
    sphere_mask = np.linalg.norm(df[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC']] - np.array(center),
                                 axis=1) < radius
    current_sphere = df.loc[sphere_mask, v_cols].values
    na_mask = np.any(np.isnan(current_sphere), axis=1)
    current_sphere = current_sphere[~na_mask]

    if len(current_sphere) <= min_occupation_number:
        return None, None, None, None

    current_mean = np.nanmean(current_sphere, axis=0)
    current_sigma = np.nanstd(current_sphere, ddof=1, axis=0)

    fluctuations = current_sphere - current_mean
    n = len(fluctuations)
    averages = [float(m) for m in current_mean]
    sigmas = [float(m) for m in current_sigma]

    averages.append(float(np.linalg.norm(current_mean)))
    sigmas.append(float(np.linalg.norm(current_sigma)))
    fluctuations = np.column_stack((fluctuations, np.linalg.norm(fluctuations, axis=1), np.full(fluctuations.shape[0], fill_value=n)))

    fluctuations = fluctuations.astype(float).tolist()

    return averages, sigmas, fluctuations, n

def get_local_fluctuations(df, list_of_radii, v_cols, min_occupation_number=1):
    z_min, z_max = (df['Z_bird_TC'].min(), df['Z_bird_TC'].max())
    stats_per_radius_per_sphere = []
    fluctuations_per_radius = pd.DataFrame()

    for i_radius, radius in enumerate(list_of_radii):
        list_of_z_level = np.arange(z_min + radius,
                                    z_max - radius,
                                    2 * radius )
        list_of_centers = np.stack(np.meshgrid(radius * 2 * np.arange(-1, 1+1, 1), # -1, 0, 1 in 2*radius units
                                      radius * 2 * np.arange(-1, 1+1, 1), # -1, 0, 1 in 2*radius units
                                      list_of_z_level, indexing='ij'), axis=-1).reshape(-1,3)

        for i_center, current_center in enumerate(list_of_centers):
            (current_averages,
             current_sigmas,
             current_fluctuations,
             current_n) =get_fluctuations_per_center(df, center=current_center,
                                                     radius=radius,
                                                     v_cols=v_cols,
                                                     min_occupation_number=min_occupation_number)

            if current_averages is None:
                continue
            stats_per_radius_per_sphere.append([radius] + current_center.tolist() + current_averages + current_sigmas + [current_n])
            current_fluctuations = pd.DataFrame(current_fluctuations, columns=['x', 'y', 'z', 'xyz', 'n'])
            current_fluctuations['radius'] = radius
            current_fluctuations[['center_x', 'center_y', 'center_z']] = current_center
            fluctuations_per_radius = pd.concat([fluctuations_per_radius, current_fluctuations])
    return stats_per_radius_per_sphere, fluctuations_per_radius