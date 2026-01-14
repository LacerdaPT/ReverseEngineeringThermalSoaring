import numpy as np
from matplotlib import pyplot as plt


def plot_turbulence_downsampling_comparison(delta_R_against, radial_corr_against, fit_against, errorbars_against,
                                            current_delta_R, current_radial_corr, current_radial_corr_std,
                                            fitting_limits, plotting_limits):

    polynom_against = np.poly1d(fit_against)
    fig, ax = plt.subplots(1, figsize=(19.2,  9.77), tight_layout=True)
    mask_to_plot = (current_delta_R < plotting_limits[1]) & (current_delta_R > plotting_limits[0])
    mask_to_fit = (current_delta_R < fitting_limits[1]) & (current_delta_R > fitting_limits[0])
    ax.plot(delta_R_against[mask_to_plot],
            radial_corr_against[mask_to_plot],
            # c=f'C{i_DS}', alpha=0.8,
            label=f'Original'
            )
    ax.plot(delta_R_against[mask_to_plot],
            np.exp(polynom_against(np.log(delta_R_against[mask_to_plot]))),
            label=f'Original\nslope={round(fit_against[0], 3)} +- {errorbars_against[0]:.2e}',
            # c=f'C{i_DS}',
            alpha=0.7,
            ls='--')

    # CURRENT DOWNSAMPLING

    mask_to_plot = (current_delta_R < plotting_limits[1]) & (current_delta_R > plotting_limits[0])
    mask_to_fit = (current_delta_R < fitting_limits[1]) & (current_delta_R > fitting_limits[0])
    errors_for_fitting = [1 / s if s != 0 else 0 for s in current_radial_corr_std[mask_to_fit]]

    current_linear_fit = np.polyfit(np.log(current_delta_R[mask_to_fit]),
                                    np.log(current_radial_corr[mask_to_fit]),
                                    w=errors_for_fitting, deg=1,
                                    cov=True)
    current_linear_fit, cov = current_linear_fit
    current_errorbars = np.sqrt(np.diag(cov))

    error_dict = {}
    error_dict['slope'], error_dict['intercept'] = current_linear_fit
    error_dict['error_slope'], error_dict['error_intercept'] = current_errorbars


    current_polynom = np.poly1d(current_linear_fit)
    current_fit_prediction = np.exp(current_polynom(np.log(current_delta_R[mask_to_plot])))
    against_fit_prediction = np.exp(polynom_against(np.log(current_delta_R[mask_to_plot])))

    current_fit_mae = np.mean(np.abs(current_fit_prediction - against_fit_prediction))
    current_fit_mse = np.sqrt(np.mean((current_fit_prediction - against_fit_prediction) ** 2))

    current_data_mae = np.mean(np.abs(current_radial_corr - radial_corr_against))
    current_data_mse = np.sqrt(np.mean((current_radial_corr - radial_corr_against) ** 2))

    error_dict['data_mae'] = float(current_data_mae)
    error_dict['data_mse'] = float(current_data_mse)
    error_dict['fit_mae'] = float(current_fit_mae)
    error_dict['fit_mse'] = float(current_fit_mse)

    print(f'fitting {np.sum(mask_to_fit)} points')

    ax.plot(current_delta_R[mask_to_plot],
            current_radial_corr[mask_to_plot],
            # c=f'C{i_DS}', alpha=0.8,
            label=f'current correlation'
            )
    ax.plot(current_delta_R[mask_to_plot],
            np.exp(current_polynom(np.log(current_delta_R[mask_to_plot]))),
            label=f'current fit\nslope={round(current_linear_fit[0], 3)} +- {current_errorbars[0]:.2e}',
               # c=f'C{i_DS}',
            alpha=0.7,
            ls='--')
    ax.loglog()
    ax.grid(which='both', axis='x')
    ax.set_xlabel('$\\Delta R (m)$')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.set_title('$C(R)$')

    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax.text(0.33, 0.95,
            f'fit MAE = {current_fit_mae:.2g}\n'
            f'fit MSE = {current_fit_mse:.2g}\n'
            f'data MAE = {current_data_mae:.2g}\n'
            f'data MSE = {current_data_mse:.2g}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    return fig, ax, error_dict


def plot_turbulence_correlation(ax, deltas_array, corr, delta_R, radial_corr, radial_corr_std,
                                linear_fit_params=None, z_offset=0, do_example_plots=False,
                                plotting_limits=(10, 100)):
    if do_example_plots:
        ax_contour, ax_angles, ax_radial = ax
    else:
        ax_contour, ax_radial = ax

    n_dim = corr.ndim
    lower_plotting_limit = plotting_limits[0]
    upper_plotting_limit = plotting_limits[1]

    delta_x = deltas_array[0]
    delta_y = deltas_array[1]
    delta_R_mg = np.sqrt(delta_x ** 2 + delta_y ** 2)

    radius_mask = delta_R_mg < upper_plotting_limit
    if n_dim == 3:
        delta_z = deltas_array[-1]
        z_mask = delta_z == z_offset
    else:
        z_mask = np.ones(shape=delta_x.shape, dtype=bool)

    # ================================================================================================================ #
    # =================================   CONTOUR CORRELATION     ==================================================== #
    # ================================================================================================================ #

    if n_dim == 3:
        m = ax_contour.contourf(delta_x[z_mask].reshape(delta_x.shape[:-1]),
                                delta_y[z_mask].reshape(delta_y.shape[:-1]),
                                corr[z_mask].reshape(corr.shape[:-1]), levels=30)
    else:
        m = ax_contour.contourf(delta_x[z_mask].reshape(delta_x.shape),
                                delta_y[z_mask].reshape(delta_y.shape),
                                corr[z_mask].reshape(corr.shape), levels=30)

    cb = plt.colorbar(ax=ax_contour, mappable=m, orientation='horizontal', label='Correlation')

    # ================================================================================================================ #
    # =================================   RADIAL CORRELATION     ===================================================== #
    # ================================================================================================================ #

    mask_to_plot = (delta_R < upper_plotting_limit) & (delta_R > lower_plotting_limit)
    ax_radial.plot(delta_R[mask_to_plot], radial_corr[mask_to_plot], label='current\ncorrelation')
    ax_radial.fill_between(delta_R[mask_to_plot],
                           radial_corr[mask_to_plot] + radial_corr_std[mask_to_plot],
                           radial_corr[mask_to_plot] - radial_corr_std[mask_to_plot],
                           alpha=0.1)

    # ================================================================================================================ #
    # =================================    ANGLE CORRELATION     ===================================================== #
    # ================================================================================================================ #

    if do_example_plots:
        delta_theta = np.arctan2(delta_y, delta_x)
        for i_angle, angle in enumerate([0, np.pi / 4, np.pi / 2]):
            angle_mask = np.isclose(delta_theta, angle)
            if angle_mask.size == 0:
                continue
            angle_mask = np.logical_and(angle_mask, radius_mask)
            angle_mask = np.logical_and(angle_mask, z_mask)
            [m] = ax_angles.plot(delta_R_mg[angle_mask], corr[angle_mask],
                                 label=f'$\\theta = {round(angle, 3)}, z={z_offset}$')

            # if i_DS == 0:
            ax_contour.plot([0, np.max(delta_x[angle_mask])],
                            [0, np.max(delta_y[angle_mask])], c=m.get_color())
            ax_angles.set_title('$C(R, \\theta)$')
            ax_angles.set_xlabel('R (m)')
            ax_angles.set_ylabel('Correlation')
            ax_angles.legend()
            ax_angles.loglog()
            ax_angles.legend()

    # ================================================================================================================ #
    # ===================================     LINEAR FIT       ======================================================= #
    # ================================================================================================================ #

    if linear_fit_params is not None:
        fit_parameters = linear_fit_params['parameters']
        errorbars = linear_fit_params['errorbars']
        plotting_poly = np.poly1d(fit_parameters)
        ax_radial.plot(delta_R[mask_to_plot],
                       np.exp(plotting_poly(np.log(delta_R[mask_to_plot]))),
                       label=f'slope={round(fit_parameters[0], 4)} +- {errorbars[0]:.2e}',
                       ls='--')
        if do_example_plots:
            ax_angles.plot(delta_R[mask_to_plot],
                           np.exp(plotting_poly(np.log(delta_R[mask_to_plot]))),
                           label=f'slope={round(fit_parameters[0], 4)} +- {errorbars[0]:.2e}')

    ax_contour.set_xlabel('$\\Delta X (m)$')
    ax_contour.set_ylabel('$\\Delta Y (m)$')
    ax_contour.set_aspect('equal')
    ax_radial.loglog()
    ax_radial.grid(which='both', axis='x')
    ax_radial.set_xlabel('R (m)')
    ax_radial.set_ylabel('Correlation')
    ax_radial.legend()
    
    return ax
