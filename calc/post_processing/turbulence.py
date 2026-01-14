import numpy as np
from scipy import stats, fft
from scipy.optimize import curve_fit
from scipy.spatial import KDTree

from calc.stats import my_lorentz, my_lorentz_cdf


def get_turbulence_component_distributions(df):
    fluct_range = np.linspace(np.min(
        [np.percentile(df['dXdT_air_res'].dropna(), 1), np.percentile(df['dYdT_air_res'].dropna(), 1),
         np.percentile(df['dZdT_air_res'].dropna(), 1)], ),
                              np.max([np.percentile(df['dXdT_air_res'].dropna(), 99),
                                      np.percentile(df['dYdT_air_res'].dropna(), 99),
                                      np.percentile(df['dZdT_air_res'].dropna(), 99)], ),
                              100)
    fluct_norm_range = np.linspace(0, np.percentile(df['V_res'].dropna(), 99), 100)
    mean_dict = {col: np.mean(df[col].dropna()) for col in ['dXdT_air_res', 'dYdT_air_res',
                                                            'dZdT_air_res']}
    std_dict = {col: np.std(df[col].dropna(), ddof=1) for col in ['dXdT_air_res',
                                                                  'dYdT_air_res',
                                                                  'dZdT_air_res']}

    params_dict = {}
    values_dict = {}
    ks_dict = {}

    for col in ['dXdT_air_res',
                'dYdT_air_res',
                'dZdT_air_res']:
        params_dict[col] = {}
        values_dict[col] = {}
        ks_dict[col] = {}

        for d in [  #'laplace',
            #'cauchy',
            'norm']:
            distribution = getattr(stats, d)

            params_dict[col][d] = distribution.fit(df[col].dropna())
            values_dict[col][d] = distribution.pdf(fluct_range, *params_dict[col][d])
            ks_dict[col][d] = stats.kstest(df[col].dropna().values, d, args=params_dict[col][d], N=len(df[col].dropna()))

        aaa = np.histogram(df[col].dropna(), bins=round(np.sqrt(len(df[col].dropna()))), density=True)
        my_x = (aaa[1][1:] + aaa[1][:-1]) / 2
        params_dict[col]['lorentz'] = curve_fit(my_lorentz, my_x, aaa[0], p0=(0.02, 1, 0.02, 0))[0]

        values_dict[col]['lorentz'] = np.array([my_lorentz(x, *params_dict[col]['lorentz']) for x in fluct_range])
        ks_dict[col]['lorentz'] = stats.kstest(df[col].dropna().values,
                                               my_lorentz_cdf,
                                               args=params_dict[col]['lorentz'].tolist() + [np.min(my_x), 50])

    params_dict['V_res'] = {'gamma': stats.gamma.fit(df['V_res'].dropna())}
    values_dict['V_res'] = {'gamma': stats.gamma.pdf(fluct_norm_range, *params_dict['V_res']['gamma'])}
    ks_dict['V_res'] = {'gamma': stats.kstest(df['V_res'].dropna().values, 'gamma',
                                              args=params_dict['V_res']['gamma'],
                                              N=len(df['V_res'].dropna()))}

    return params_dict, values_dict, ks_dict, mean_dict, std_dict, fluct_range, fluct_norm_range


def get_energy_cascade_from_air_velocity_field(avf, x_array, y_array, z_array, max_distance):

    nx = x_array.size
    ny = y_array.size
    nz = z_array.size

    scale =[(np.max(x_array) - np.min(x_array)) / nx,
            (np.max(y_array) - np.min(y_array)) / ny,
            (np.max(z_array) - np.min(z_array)) / nz,]
    x_mg, y_mg, z_mg = np.meshgrid(x_array, y_array, z_array, indexing='ij')
    xyz_mg = np.stack([x_mg, y_mg, z_mg],  axis=-1).reshape((nx, ny, nz, 3))
    n_dim = xyz_mg.ndim - 1
    my_df = avf.df[['X_thermal', 'Y_thermal', 'Z_thermal',
                        'dXdT_air_res', 'dYdT_air_res', 'dZdT_air_res']]
    my_df = my_df[np.all(~my_df[['dXdT_air_res', 'dYdT_air_res', 'dZdT_air_res']].isna(), axis=1)]
    tree = KDTree(my_df[['X_thermal', 'Y_thermal', 'Z_thermal']].values)

    dist, ind = tree.query(xyz_mg.reshape((-1,3)))

    dist = dist.reshape((nx, ny, nz))
    indices_to_keep = dist < max_distance
    v_array = avf.get_velocity_fluctuations(X=xyz_mg, t=0, return_components=False, relative_to_ground=False)
    #t0 = air_obj.get_velocity(X=xyz_mg, t=0, include='turbulence', return_components=False, relative_to_ground=False)
    #del air_obj
    v_array[~indices_to_keep] = np.nan
    v_array[np.isnan(v_array)] = 0
    energy = 1/2 * np.sum(v_array ** 2, axis=-1)

    FS = fft.rfftn(energy)
    ns = FS.shape

    freq = [fft.fftfreq(ns[i], d=scale[i])[:ns[i]//2]
            for i in range(n_dim)]

    FS = np.abs(FS[:ns[0]//2,:ns[1]//2,:ns[2]//2])

    k_mgs = np.meshgrid(*freq)

    k_mg = np.stack(k_mgs, axis=-1)

    k_magnitude_mg = np.linalg.norm(k_mg, axis=-1)
    k_magnitude_mg = k_magnitude_mg.reshape(-1)
    sorting_indices = np.argsort(k_magnitude_mg)
    k_magnitude_mg = k_magnitude_mg[sorting_indices]
    FS_sorted = FS.reshape(-1)[sorting_indices]

    return  FS_sorted, k_magnitude_mg

