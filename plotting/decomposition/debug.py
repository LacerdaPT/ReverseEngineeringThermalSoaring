from itertools import product

import matplotlib
matplotlib.use('QtAgg')
import numpy as np

from calc.stats import quadratic_function2D, quadratic_function

from matplotlib import pyplot as plt


def debug_plot_decomposition_get_wind_from_thermal_core(df_calc):
    fig, ax_arr = plt.subplots(3, 2, figsize=(19, 12))
    ax_arr[0, 0].plot(df_calc['dXdZ'], df_calc['Z_avg'], label='calculated')
    ax_arr[0, 1].plot(df_calc['dYdZ'], df_calc['Z_avg'], label='calculated')
    ax_arr[1, 0].plot(df_calc['wind_X_raw'], df_calc['Z_avg'], label='calculated')
    ax_arr[1, 1].plot(df_calc['wind_Y_raw'], df_calc['Z_avg'], label='calculated')
    ax_arr[2, 0].plot(df_calc['wind_X'], df_calc['Z_avg'], label='calculated')
    ax_arr[2, 1].plot(df_calc['wind_Y'], df_calc['Z_avg'], label='calculated')
    ax_arr[0, 0].legend()
    ax_arr[0, 0].set_title('dXdZ')
    ax_arr[0, 1].set_title('dYdZ')
    ax_arr[1, 0].set_title('wind_X_raw')
    ax_arr[1, 1].set_title('wind_Y_raw')
    ax_arr[2, 0].set_title('wind_X smoothed')
    ax_arr[2, 1].set_title('wind_Y smoothed')
    plt.show(block=True)


def debug_plot_decomposition_get_maximum_vertical_velocity2d(X_array, Y_array, Vz_array, mask, popt,
                                                             x_max=None, y_max=None, dZdT_air_max=None):
    from matplotlib import patches as mpatches
    rho_array = np.sqrt(X_array ** 2 + Y_array ** 2)
    (a, b, c, d, e, f) = popt
    N = mask.sum()
    rho_median = np.median(rho_array)
    rho_std = np.std(rho_array)
    rho_maximum_allowed = np.max(rho_array[mask])
    rho_for_quad = np.arange(0, rho_maximum_allowed, 1)
    theta_for_quad = np.linspace(0, 2 * np.pi, 20)
    rho_theta_for_quad = np.array(list(product(rho_for_quad, theta_for_quad)))
    x_for_quad = rho_theta_for_quad[:, 0] * np.cos(rho_theta_for_quad[:, 1])
    y_for_quad = rho_theta_for_quad[:, 0] * np.sin(rho_theta_for_quad[:, 1])
    xy_for_quad = np.vstack([x_for_quad, y_for_quad]).T

    quadratic_values = np.apply_along_axis(quadratic_function2D, 1, xy_for_quad, a, b, c, d, e, f)
    fig = plt.figure(figsize=(19, 12))

    ax_hist = fig.add_subplot(121)
    ax_velocities = fig.add_subplot(122, projection='3d')

    ax_hist.hist2d(X_array, Y_array, bins=20)
    ax_hist.add_patch(mpatches.Circle((0, 0), rho_maximum_allowed, color='r', alpha=0.3))
    ax_velocities.scatter(X_array, Y_array, Vz_array, c=Vz_array, s=4, alpha=0.3)
    ax_velocities.scatter(X_array[mask], Y_array[mask], Vz_array[mask], s=4, alpha=0.3)
    if np.all([x_max is not None, y_max is not None, dZdT_air_max is not None]):
        ax_velocities.scatter3D([x_max], [y_max], [dZdT_air_max], marker='*')
        ax_velocities.set_title(f'{x_max=:.3g}, {y_max=:.3g}, {dZdT_air_max=:.3g}')
    ax_velocities.plot_trisurf(xy_for_quad[:, 0], xy_for_quad[:, 1],
                               quadratic_values,
                               # s=4,
                               color='r',
                               alpha=0.5)
    # ax_velocities.set_xlim(0, sigma_v)
    fig.suptitle(f'$\\rho_{{max}} = {rho_maximum_allowed:.4g}$\n '
                 f'$\\rho_{{median}} = {rho_median:.4g}$\n'
                 f'$\\rho_{{std}} = {rho_std:.4g}$\n'
                 f'{N=}')
    ax_hist.set_aspect('equal')
    plt.tight_layout()
    plt.show(block=True)

    return fig, [ax_hist, ax_velocities]


def debug_plot_decomposition_get_maximum_vertical_velocity(rho_array, Vz_array, mask, rho_max, dZdT_air_max, popt):
    a, b, c = popt
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    ax.scatter(rho_array,
               Vz_array, c='C2',
               s=7, alpha=0.7, label='all data')
    ax.scatter(rho_array[mask],
               Vz_array[mask], c='C1',
               s=7, alpha=0.7, label='used data')
    ax.plot(np.linspace(0, np.max(rho_array[mask]), 100),
            [quadratic_function(x, a, b, c) for x in np.linspace(0, np.max(rho_array[mask]), 100)],
            alpha=0.7, label='fit')
    ax.set_xlabel('$\\rho$ (m)')
    ax.set_ylabel('$V_{air,Z}$ (m/s)')
    ax.scatter([rho_max], [dZdT_air_max], s=38, c='g', marker='^',
               label=f"Maximum:\n"
                     f"$\\rho$={rho_max:.2g}\n"
                     "$V_{air,Z}$=" + f"{dZdT_air_max:.3g}"
               )
    ax.legend()
    fig.suptitle('Maximum by quadratic fit')
    plt.tight_layout()
    plt.show(block=True)

    return fig, ax