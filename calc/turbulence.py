import os
import dill as pickle
import time
from itertools import product

import numpy as np
import yaml

from calc.stats import get_correlation_differences_and_stats_from_sample
from object.air import AirVelocityField
from plotting.turbulence.multiscale import plot_turbulence_correlation


def get_correlation_multiplier_from_fits(x_array, original_fit_parameters, fit_parameters_to_change, old_norm):
    best_intercept = (original_fit_parameters[1] +
                      (original_fit_parameters[0] - fit_parameters_to_change[0])
                      * np.mean(np.log(x_array))
                      )

    best_norm = np.exp(best_intercept - fit_parameters_to_change[1]) * old_norm
    multiplier = best_norm / old_norm
    return best_intercept, best_norm, multiplier


def calculate_turbulence_correlation_differences(p, air_parameters_dict, DS, L_field, N_field, n_dim,
                                                 correlation_span,  # span of correlation in meters *in each side*
                                                 original_spatial_scale, original_velocity_scale,
                                                 other_spatial_scales, fitting_limits, plotting_limits,
                                                 original_fit_data,
                                                 meter_per_index=1,
                                                 fit_norm=False,
                                                 original_correlation_data=None,
                                                 save_folder=None, history=None,
                                                 do_plots=False, z_offset=0, loss_type='fit_difference'):
    if fit_norm:
        current_weight_list = [float(param) for param in p[1:]]
    else:
        current_weight_list = [float(param) for param in p]

    current_dict_of_scales = {other_spatial_scales[i]: current_weight
                          for i, current_weight in enumerate(current_weight_list)}
    current_dict_of_scales[original_spatial_scale] = original_velocity_scale
    current_list_of_scales = list(current_dict_of_scales.keys())
    current_dict_of_scales_str = {k: round(v, 4) for k, v in current_dict_of_scales.items()}

    air_parameters_dict['turbulence']['scales'] = current_dict_of_scales
    air_parameters_dict['turbulence']['downsampling_factor'] = DS

    current_norm = float(p[0]) if fit_norm else False

    air_parameters_dict['turbulence']['normalization'] = current_norm

    calculation_start_time = time.time()

    # ================================================================================================================ #
    # ==================================     GET DATA    ============================================================= #
    # ================================================================================================================ #
    air_velocity_obj = AirVelocityField(air_parameters=air_parameters_dict, t_start_max=2)
    air_velocity_obj.reset_turbulence_function(0)

    x_array = np.linspace(0, L_field, N_field, endpoint=True)
    y_array = np.linspace(0, L_field, N_field, endpoint=True)
    z_array = np.linspace(0, L_field, N_field, endpoint=True) if n_dim == 3 else np.array([z_offset])
    positions_array = list(product(x_array, y_array, z_array))
    turbulence_data_flatten = air_velocity_obj.velocity_functions['turbulence'](positions_array, t=0, velocity_component=0)

    turbulence_data = turbulence_data_flatten.reshape(x_array.size,
                                                      y_array.size,
                                                      z_array.size)
    if n_dim == 2:
        turbulence_data = turbulence_data[:, :, 0]
    del turbulence_data_flatten
    # ================================================================================================================ #
    # ==================================      CALCULATE CORRELATIONS      ============================================ #
    # ================================================================================================================ #

    if save_folder:
        suffix = f'current_scales={current_dict_of_scales_str}_{current_norm=:.2f}'
        correlation_data_filename = f'all_correlation_data_{suffix}.pkl'
        correlation_fit_filename = f'correlation_fit_{suffix}.pkl'
    else:
        correlation_data_filename = None
        correlation_fit_filename = None
    current_correlation_data = get_correlation_differences_and_stats_from_sample(turbulence_data,
                                                                                 correlation_span=correlation_span,
                                                                                 n_dimensions=n_dim,
                                                                                 fitting_limits=fitting_limits,
                                                                                 scale=meter_per_index,
                                                                                 plot_debug=False,
                                                                                 save_folder=save_folder,
                                                                                 correlation_data_filename=correlation_data_filename,
                                                                                 correlation_fit_filename=correlation_fit_filename)

    corr = current_correlation_data['cartesian_correlation']['values']
    deltas_array = current_correlation_data['cartesian_correlation']['deltas']
    delta_R = current_correlation_data['radial_correlation']['deltas']
    radial_corr = current_correlation_data['radial_correlation']['values']
    radial_corr_std = current_correlation_data['radial_correlation']['std']
    current_fit = current_correlation_data['linear_fit']

    # ================================================================================================================ #
    # =================================          CALCULATE LOSS           ============================================ #
    # ================================================================================================================ #

    calculation_end_time = time.time()
    print('done with calculations')
    print(f'it took {round((calculation_end_time - calculation_start_time) / 60, 2)} min')

    lower_fitting_limit = fitting_limits[0]
    upper_fitting_limit = fitting_limits[1]
    mask_to_fit = (delta_R < upper_fitting_limit) & (delta_R > lower_fitting_limit)

    print(f'fitting {np.sum(mask_to_fit)} points')
    current_search_fit_parameters = current_fit['parameters']
    current_polynom = np.poly1d(current_search_fit_parameters)
    original_fit_parameters = original_fit_data['parameters']
    original_polynom = np.poly1d(original_fit_parameters)
    if loss_type == 'fit_difference':
        current_search_distance_to_original = np.sqrt(np.mean((original_polynom(np.log(delta_R[mask_to_fit]))
                                                               - current_polynom(np.log(delta_R[mask_to_fit]))) ** 2
                                                              )
                                                      )
    elif loss_type == 'slope_difference':
        current_search_distance_to_original = np.abs(original_fit_parameters[0] - current_search_fit_parameters[0])

    errorbars = current_correlation_data['linear_fit']['errorbars']
    if fit_norm:
        plotting_poly = np.poly1d(current_search_fit_parameters)
        current_correlation_data['multiplier'] = 1.0

    else:
        best_intercept = (original_fit_parameters[1] +
                          (original_fit_parameters[0] - current_search_fit_parameters[0])
                          * np.mean(np.log(delta_R[mask_to_fit]))
                          )
        sum_of_velocity_scales = sum(list(current_dict_of_scales.values()))

        best_norm = np.exp(best_intercept - current_search_fit_parameters[1]) * sum_of_velocity_scales
        multiplier = best_norm / sum_of_velocity_scales
        current_norm = best_norm
        current_fit['parameters'][1] = best_intercept

        current_correlation_data['radial_correlation']['values'] = radial_corr * multiplier
        current_correlation_data['radial_correlation']['std'] = radial_corr_std * multiplier
        current_correlation_data['multiplier'] = multiplier

    if save_folder is not None:

        data = {'correlation': corr,
                'deltas': deltas_array,
                'radial_correlation': radial_corr,
                'delta_R': delta_R,
                'radial_correlation_std': radial_corr_std
                }
        data_folder = os.path.join(save_folder, 'data')
        os.makedirs(data_folder, exist_ok=True)

        if len(data['deltas']) == 2:
            data['deltas'] = {'X': data['deltas'][0].tolist(),
                              'Y': data['deltas'][1].tolist()}
        else:
            data['deltas'] = {'X': data['deltas'][0].tolist(),
                              'Y': data['deltas'][1].tolist(),
                              'Z': data['deltas'][2].tolist()}
        with open(os.path.join(data_folder, correlation_data_filename), 'w') as f:
            yaml.dump(data, f)

        with open(os.path.join(data_folder, correlation_fit_filename), 'w') as f:
            yaml.dump({k: v.tolist() for k, v in current_fit.items()}, f,
                      default_flow_style=False)
    # ================================================================================================================ #
    # ==================================           PLOTTING             ============================================== #
    # ================================================================================================================ #
    if do_plots:
        plotting_start_time = time.time()
        import matplotlib
        if not save_folder:
            matplotlib.use('QtAgg')
        else:
            matplotlib.use('GTK4Cairo')

        from matplotlib import pyplot as plt
        if not save_folder:
            plt.ioff()
        if len(deltas_array) == 3:
            delta_z = deltas_array[-1]
            z_index = np.round(np.array(delta_z.shape) / 2).astype(np.int32)
            z_value = delta_z[z_index[0], z_index[1], z_index[2]]
        else:
            z_value = 0

        fig, ax_dict = plt.subplot_mosaic([['Contour Current', 'examples Current', 'radial'],
                                           ['Contour Original', 'examples Original', 'radial']],
                                          figsize=(19.2, 9.77), tight_layout=True)
        ax_current = plot_turbulence_correlation([ax_dict['Contour Current'], ax_dict['examples Current'], ax_dict['radial']],
                                                 deltas_array, corr, delta_R, radial_corr, radial_corr_std,
                                                 linear_fit_params=current_correlation_data['linear_fit'],
                                                 z_offset=z_value, do_example_plots=True,
                                                 plotting_limits=plotting_limits)
        ax_original = plot_turbulence_correlation([ax_dict['Contour Original'],
                                                   ax_dict['examples Original'],
                                                   ax_dict['radial']],
                                                  original_correlation_data['deltas'],
                                                  original_correlation_data['correlation'],
                                                  original_correlation_data['delta_R'],
                                                  original_correlation_data['radial_correlation'],
                                                  original_correlation_data['radial_correlation_std'],
                                                  linear_fit_params=current_correlation_data['linear_fit'],
                                                  z_offset=z_value, do_example_plots=True,
                                                  plotting_limits=plotting_limits)

        suffix = f'current_scales={current_dict_of_scales_str}_{current_norm=:.4f}'

        scales_for_title = {k: round(v, 2) for k, v in current_dict_of_scales.items()}
        fig_title = f'Correlation $\\Delta V_X$\n' \
                    f'{DS=} - scales: {scales_for_title}\n' \
                    f'{current_norm=:.2f}'
        fig.suptitle(fig_title)
        ax_current[0].set_title(f'Correlation for {DS=}')
        ax_current[1].set_title('$C(R, \\theta)$' + f'\n{DS=}')
        ax_current[2].set_title('$C(R)$\n' + f'{DS=}\n' +
                                f'fitted in {lower_fitting_limit} < R < {upper_fitting_limit} ')
        ax_original[0].set_title(f'Correlation for Original')
        ax_original[1].set_title('$C(R, \\theta)$' + f'\nOriginal')
        ax_original[2].set_title('$C(R)$\n' + f'Original\n' +
                                 f'fitted in {lower_fitting_limit} < R < {upper_fitting_limit} ')
        if save_folder:
            fig_filename = f'correlation_{suffix}.png'
            os.makedirs(os.path.join(save_folder, 'figures'), exist_ok=True)
            from gi.overrides.Gtk import Window as gtkwindow

            if isinstance(fig.canvas.manager.window, gtkwindow):
                fig.canvas.manager.window.maximize()
            else:
                fig.canvas.manager.window.showMaximized()
            plt.draw()
            #plt.close(fig)
            plt.savefig(os.path.join(save_folder, 'figures', fig_filename),
                        dpi=200)
        else:
            plt.show(block=True)

        plotting_end_time = time.time()
        print('done with plotting')
        print(f'it took {round((plotting_end_time - plotting_start_time) / 60, 2)} min')
    if history is not None:
        history.append(current_weight_list + [current_norm, current_search_distance_to_original])

    if history is not None:
        print(len(history), f'norm={current_norm:.6g}', '_'.join([f'{s}={w:.6g}' for s, w in current_dict_of_scales.items()]),
              f'loss={current_search_distance_to_original:.3g}')
    else:
        print(f'norm={current_norm:.6g}', '_'.join([f'{s}={w:.6g}' for s, w in current_dict_of_scales.items()]),
              f'loss={current_search_distance_to_original:.3g}')
    return current_search_distance_to_original
