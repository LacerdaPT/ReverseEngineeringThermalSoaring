import logging
import os
import pprint
from argparse import ArgumentParser

from decomposition.auxiliar import parse_decomposition_arguments, decomposition_preparation
from decomposition.core import start_decomposition
from misc.auxiliar import config_logger

parser = ArgumentParser('Decompose air velocity from bird velocity')

parser.add_argument('-y', '--yaml', dest='yaml', type=str,
                    help='path to yaml file to read configuration from')
parser.add_argument('-i', '--input', dest='input_folder', type=str,
                    help='folder to read data.pkl and parameters.pkl from')
parser.add_argument('-o', '--output', dest='output_folder', type=str, default=None,
                    help='folder to write iterations.pkl, thermal_core.pkl and bins.pkl to')
parser.add_argument('-dt', '--delta-t', dest='dt', type=float, default=None, help='time interval between points')
parser.add_argument('-n', '--n-iterations', dest='n_iterations', type=int, default=None,
                    help='number of iterations to calculate')
parser.add_argument('-a', '--alpha', dest='alpha', type=float, default=None,
                    help='defines the initial condition of air velocity and bird velocity. '
                         'Air velocity = alpha * ground velocity and '
                         'Bird velocity = (1 - alpha ) * ground velocity')
parser.add_argument('-timec', '--time-col', dest='time_col', type=str, default=None,
                    help='name of the time column in the input data')
parser.add_argument('-birdc', '--bird-name-col', dest='bird_name_col', type=str, default=None,
                    help='name of the bird column in the input data')
parser.add_argument('-xc', '--x-col', dest='X_col', type=str, default=None,
                    help='name of the x column in the input data')
parser.add_argument('-yc', '--y-col', dest='Y_col', type=str, default=None,
                    help='name of the y column in the input data')
parser.add_argument('-zc', '--z-col', dest='Z_col', type=str, default=None,
                    help='name of the z column in the input data')
parser.add_argument('-tws', '--thermal-windowsize', dest='thermal_window_size', type=int, default=None,
                    help='size of the window to using on moving average to calculate thermal core')
parser.add_argument('-tmp', '--thermal-minperiods', dest='thermal_min_periods', type=int, default=None,
                    help='size of the window to using on moving average to calculate thermal core')
parser.add_argument('-tc', '--thermal-center', dest='thermal_center', action='store_true', default=None,
                    help='whether or not to show debugging features')
parser.add_argument('-twt', '--thermal-windowtype', dest='thermal_window_type', type=str,
                    help='window type to using on moving average to calculate thermal core. '
                         'Check '
                         'https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows'
                         'too see all option')
parser.add_argument('-twp', '--thermal-window-parameters', dest='thermal_window_params', nargs='*',
                    help='parameters for window type to using on moving average to calculate thermal core. '
                         'Check '
                         'https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows'
                         'too see all option')
parser.add_argument('-sws', '--smoothing-window-size', dest='smoothing_window_size', type=int,
                    help='size of the window to using on moving average to smooth velocities')
parser.add_argument('-smp', '--smoothing-min-periods', dest='smoothing_min_periods', type=int,
                    help='size of the window to using on moving average to smooth velocities')
parser.add_argument('-sc', '--smoothing-center', dest='smoothing_center', action='store_true', default=None,
                    help='whether or not to show debugging features')
parser.add_argument('-swt', '--smoothing-window-type', dest='smoothing_window_type', type=str,
                    help='window type to using on moving average to smooth velocities. '
                         'Check '
                         'https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows'
                         'too see all option')
parser.add_argument('-swp', '--smoothing-window-parameters', dest='smoothing_window_params', nargs='*',
                    help='parameters for window type to using on moving average to smooth velocities. '
                         'Check '
                         'https://docs.scipy.org/doc/scipy/reference/signal.windows.html#module-scipy.signal.windows'
                         'too see all option')
parser.add_argument('-L', '--CL', dest='CL', type=float, help='CL')
parser.add_argument('-D', '--CD', dest='CD', type=float, help='CD')
parser.add_argument('-M', '--mass', dest='mass', type=float, help='mass')
parser.add_argument('-W', '--wing-area', dest='wing_area', type=float, help='wing_area')
parser.add_argument('-d', '--debug', dest='debug', action='store_true', default=None,
                    help='whether or not to show debugging features')
parser.add_argument('-dr', '--dryrun', dest='save', action='store_false', default=None,
                    help='use this flag to run a dry-run (running without saving')
parser.add_argument('-v', '--verbose', dest='verbosity', type=int, choices=[1, 2, 3, 4, 5], default=None,
                    help='Level of verbosity')

def main():
    parse_args = parser.parse_args().__dict__

    decomposition_args = parse_decomposition_arguments(parse_args)
    (df, _, run_parameters,
     data_parameters,
     thermal_core_ma_args,
     smoothing_ma_args,
     physical_parameters,
     binning_parameters,
     spline_parameters, debug_dict) = decomposition_preparation(decomposition_args)

    logger = logging.getLogger()
    config_logger(logger, output_dir=run_parameters['output_folder'],
                  verbosity=run_parameters['verbosity'], log_to_file=run_parameters['save'])

    logger.info('RUN PARAMETERS')
    for line in pprint.pformat(run_parameters).split('\n'):
        logger.log(level=logging.INFO, msg=line)

    (df_iterations,
     df_thermal_core,
     df_aero,
     df_bins,
     df_splines_stats,
     list_of_losses) = start_decomposition(df, run_parameters=run_parameters,
                                    thermal_core_ma_args=thermal_core_ma_args,
                                    smoothing_ma_args=smoothing_ma_args,
                                    initial_physical_parameters=physical_parameters,
                                    binning_parameters=binning_parameters,
                                    spline_parameters=spline_parameters,
                                    debug_dict=debug_dict)
    if run_parameters['save']:
        with open(os.path.join(run_parameters['output_folder'], 'decomposition_args.yaml'), 'w') as f:
            import yaml

            yaml.dump({'run_parameters':       run_parameters,
                       'thermal_core_ma_args': thermal_core_ma_args,
                       'smoothing_ma_args':    smoothing_ma_args,
                       'physical_parameters': physical_parameters.set_index('bird_name').to_dict(orient='index'),
                       'binning_parameters':   binning_parameters,
                       'spline_parameters':    spline_parameters
                       },
                      f, default_flow_style=False)
    return df_iterations


if __name__ == '__main__':
    main()
