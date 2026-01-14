import os
import time
from copy import copy
from itertools import product
from typing import Union

import matplotlib as mpl
from matplotlib.collections import QuadMesh
from matplotlib.lines import Line2D

from calc.geometry import get_cartesian_velocity_on_rotating_frame_from_inertial_frame
from data.auxiliar import downsample_dataframe
from data.get_data import load_synthetic_and_decomposed

mpl.use('QtAgg')
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.backends._backend_tk import FigureManagerTk
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Circle
from matplotlib.widgets import Button, RadioButtons, RangeSlider
from mpl_toolkits.axes_grid1 import make_axes_locatable

from calc.auxiliar import parse_projection_string, UnivariateSplineWrapper
from calc.stats import average_over_vars, is_outlier
from object.air import AirVelocityField, DecomposedAirVelocityField, AirVelocityFieldVisualization, \
    AirVelocityFieldBase, IterativeReconstructedAirVelocityField, ReconstructedAirVelocityField
from plotting.plot import plot_tracks_scatter, get_outlier_idx, plot_scatter3D, plot_interpolated


class PlotKeyboard(object):
    partition_index = 0
    dataset_size_threshold = 10

    def __init__(self, data, x_col='X', color_col='dZdT', y_col='Y', z_col='Z', fig=None, ax=None, ax_kw=None,
                 partition='bird_name', outlier_detection=True, sort_by=True):
        self.data = data.copy()

        self.x_col = x_col
        self.y_col = y_col
        self.z_col = z_col
        self.color_col = color_col
        self.x_data = None
        self.y_data = None
        self.z_data = None
        self.color_data = None
        self.outlier_detection = outlier_detection
        self.n_sigmas = 3
        self.save_data_array = []

        self.partition_col = None
        self.partition_names = None
        self.num_of_indices = None
        self.set_partitions(partition, sort_by)

        self.partition_index = 0
        self.set_new_data()
        if ax_kw is None:
            self.ax_kw = {'xlabel': self.x_col,
                          'ylabel': self.y_col,
                          'zlabel': self.z_col}
        else:
            self.ax_kw = ax_kw
            self.ax_kw.update({'xlabel': self.x_col, 'ylabel': self.y_col, 'zlabel': self.z_col})

        self.ax = ax
        self.fig = fig
        self.plot(show=True)

    def set_fig_and_axes(self, fig, ax):
        print('setting figs and axes')
        # None
        if (fig is None) and (ax is None):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', **self.ax_kw)
        # AX
        elif (fig is None) and (ax is not None):
            fig = plt.figure()
            fig.axes.append(ax)
        # FIG
        elif (fig is not None) and (ax is None):
            fig = fig
            ax = fig.add_subplot(111, projection='3d', **self.ax_kw)
        # FIG AND AX
        elif (fig is not None) and (ax is not None):
            ax = ax
            fig.axes.append(ax)

        self.fig = fig
        self.ax = ax
        self.fig.canvas.mpl_connect('key_press_event', self.handler)
        plt.sca(self.ax)

    def set_partitions(self, partition, sort_by):
        from types import FunctionType, LambdaType

        if isinstance(partition, str):
            self.partition_col = partition
        elif isinstance(partition, (LambdaType, FunctionType)):
            self.partition_col = 'index'
            self.data[self.partition_col] = self.data.apply(partition, axis=1)
        elif isinstance(partition, (list, np.ndarray)):
            self.partition_col = 'index'
            self.data[self.partition_col] = partition

        if sort_by is True:
            self.data = self.data.sort_values(self.partition_col)
        elif isinstance(sort_by, str):
            self.data = self.data.sort_values(by=[sort_by])
        elif isinstance(sort_by, list):
            self.data = self.data.sort_values(by=sort_by)

        self.partition_names = np.unique(self.data[self.partition_col].values)
        self.num_of_indices = len(self.partition_names)

        print(f'there are {self.num_of_indices} partitions in the data')

    def handler(self, event):
        if event.key == 'left':
            self.previous()
        elif event.key == 'right':
            self.next()
        elif event.key == 'up':
            self.more_data()
        elif event.key == 'down':
            self.less_data()
        if event.key == 's':
            self.save_partition_index()
        elif event.key == 'escape':
            self.fig.close()

    def save_partition_index(self):
        self.save_data_array.append(self.partition_names[self.partition_index])
        print(f'{self.partition_names[self.partition_index]} has been saved')

    def plot(self, show=True):
        if self.ax:
            self.ax_kw.update({'azim': self.ax.azim, 'elev': self.ax.elev})

        if self.fig:
            self.fig.clear()
            self.ax = None
        self.set_fig_and_axes(self.fig, self.ax)

        self.fig, self.ax = plot_tracks_scatter(ax=self.ax, x_data=self.x_data, y_data=self.y_data, z_data=self.z_data,
                                                color_data=self.color_data,
                                                fig=self.fig)

        plt.draw()
        if show:
            plt.show()

    def set_new_data(self):

        current_data = self.data.loc[self.data[self.partition_col] == self.partition_names[self.partition_index]]
        self.x_data = current_data[self.x_col].values
        self.y_data = current_data[self.y_col].values
        self.z_data = current_data[self.z_col].values
        self.color_data = current_data[self.color_col].values
        if self.outlier_detection:
            # _, x_indices = get_outlier_idx(self.x_data)
            # _, y_indices = get_outlier_idx(self.y_data)
            _, z_indices = get_outlier_idx(self.z_data, n_sigmas=self.n_sigmas)
            # from functools import reduce
            # reduce(union1d, (x_indices, y_indices, z_indices))
            indices_to_keep = z_indices

            print(f'keeping {len(indices_to_keep)} out of {len(self.z_data)}')

            self.x_data = self.x_data[indices_to_keep]
            self.y_data = self.y_data[indices_to_keep]
            self.z_data = self.z_data[indices_to_keep]
            self.color_data = self.color_data[indices_to_keep]

        print(f'Partition index is {self.partition_index}, (out of {len(self.partition_names)} '
              f'Partition name is {self.partition_names[self.partition_index]}')

    def more_data(self):

        if self.n_sigmas < 5:
            self.n_sigmas += 0.1
        else:
            print('maximum reached')
        print('n_sigma = ', self.n_sigmas)

        # Cycle over the existing indices
        self.set_new_data()
        self.plot()

    def less_data(self):
        if self.n_sigmas > 1:
            self.n_sigmas -= 0.1
        else:
            print('minimum reached')
        print('n_sigma = ', self.n_sigmas)

        # Cycle over the existing indices
        self.set_new_data()
        self.plot()

    def next(self):
        self.partition_index += 1
        # Cycle over the existing indices
        self.partition_index = self.partition_index % self.num_of_indices
        self.set_new_data()

        self.plot()

    def previous(self):
        self.partition_index -= 1
        # Cycle over the existing indices
        self.partition_index = self.partition_index % self.num_of_indices
        self.set_new_data()

        self.plot()


class RealDataDashboard:
    plot_styles_dict = {'colors': {'real':       {'X': 'r--', 'Y': 'y--'},
                                   'calculated': {'X': 'b', 'Y': 'g'}}
                        }
    config = {'Z_padding': 100}


    def __init__(self,
                 decomposed_air_velocity_field: Union[AirVelocityFieldBase, IterativeReconstructedAirVelocityField],
                 sliced_var='Z',
                 resolution=20, block=False, plot_3d=True, interactive=True, side_plot='Stream',
                 average_statistic='median', debug=False):

        # Data Attributes
        self.is_iterative = isinstance(decomposed_air_velocity_field, IterativeReconstructedAirVelocityField)
        self.current_vz_max_spline = None
        self.current_thermal_core = None
        self.current_wind = None
        self.current_wind_correction = None
        self.current_dCoorddZ = None
        self.current_wind_before = None
        self.debug = debug
        self.df_histogram = None
        self.real_wind = None
        self.plot_3d = plot_3d
        #self.df = df  # .copy()
#
        #self.df_splines = df_splines

        self.decomposed_air_velocity_field = decomposed_air_velocity_field
        self.real_thermal_core = None
        self.block = block
        self.interactive = interactive

        # Figure, axes, artists
        self.fig_plots = None
        self.fig_buttons = None
        self.axes = {}
        self.colorbar_axes = {}
        self.artists = {}
        self.widgets = {}
        self.gridspec_wrapper = None
        self.gridspec_buttons = None
        self.gridspec_right_panel = None
        self.gridspec_left_panel = None
        self.circle = None

        # STATE VARIABLES
        self.projection_string = 'XY'
        first_var, second_var, first_index, second_index, section_index, section_var = parse_projection_string(
            self.projection_string)
        self.plotting_X_var = first_var.upper()
        self.plotting_Y_var = second_var.upper()
        self.plotting_section_var = section_var.upper()
        self.side_plot = side_plot
        self.resolution = resolution
        self.current_limits = {'X': [], 'Y': [], 'Z': []}
        self.var_limits = {'X': [], 'Y': [], 'Z': []}
        self.active_data_axis_label = None

        self.list_of_iterations = self.decomposed_air_velocity_field.list_of_iterations

        self.current_iteration = self.list_of_iterations[0]
        self.max_iteration = max(self.list_of_iterations)
        self.min_iteration = np.min(self.list_of_iterations)
        self.current_iteration = self.min_iteration
        self.current_bin = 0
        self.current_iteration_data = None
        self.current_data = None
        self.current_thermal_core_iteration = None
        self.current_real_thermal_core = None
        self.current_wind_spline = None
        self.current_thermal_core_spline = None
        self.current_z_array = None
        self.current_bins_iteration = None
        self.current_bins_data = None
        self.histogram_data = None

        self.animation = None
        self.next_iteration_data = None
        self.next_bins_iteration = None
        self.run_time = round(time.time())

        self.cols_to_bin = ['Z_bird_TC', 'phi_bird_TC', 'rho_bird_TC']
        self.bin_index_cols = [f'bin_index_{var}' for var in self.cols_to_bin]
        self.plotting_args = {'scatter_1':        {'alpha': 0.2,
                                                   'c':     'C0',
                                                   's':     3,
                                                   },
                              'scatter_2':        {'alpha': 0.2,
                                                   'c':     'C1',
                                                   's':     3,
                                                   },
                              'moving_average_1': {'c': 'C2',
                                                   },
                              'moving_average_2': {'c': 'C3',
                                                   }
                              }
        self.average_statistic = average_statistic

    # @classmethod
    # def from_path_decomposition(cls, path_to_decomposition, input_folder=None, **kwargs):
    #
    #     synthetic_data_dict, decomposition_dict = load_synthetic_and_decomposed(path_to_decomposition,
    #                                                                             input_folder=input_folder
    #                                                                             )
    #
    #     if 'air_velocity_field' in synthetic_data_dict:
    #         air_velocity_field = synthetic_data_dict['air_velocity_field']
    #     else:
    #         air_velocity_field = synthetic_data_dict['air_parameters']
    #
    #     return ComparativeDashboard(df=decomposition_dict['iterations'],  #df_bins=decomposition_dict['bins'],
    #                                 df_splines=decomposition_dict['splines'],
    #                                 air_velocity_field=air_velocity_field,
    #                                 df_ground_truth=synthetic_data_dict['data_real'],
    #                                 **kwargs)

    def initial_setup(self):
#        self.air_velocity_field_preprocessing()
        #self.avf_vis = AirVelocityFieldVisualization(self.decomposed_air_velocity_field)
        self.data_preparation()
        self.set_iteration_data()

        self.set_sliced_data()
        self.set_gui()
        self.plots_per_iteration()
        self.go_plot()

    def reset(self, event):
        self.current_iteration = self.min_iteration
        self.widgets['iteration_text'].set_text(f'{self.current_iteration}')
        self.set_iteration_data()
        self.set_sliced_data()
        self.go_plot()

    def data_preparation(self):
        self.decomposed_data_preparation()
        # [self.df['iteration'] != 0]
        # [self.df['iteration'] != 0]
        # [self.df['iteration'] != 0]
        # [self.df['iteration'] != 0]
        # [self.df['iteration'] != 0]
        # [self.df['iteration'] != 0]

        self.var_limits['X'] = [self.df['X_bird_TC'].min(),
                                self.df['X_bird_TC'].max()]
        self.var_limits['Y'] = [self.df['Y_bird_TC'].min(),
                                self.df['Y_bird_TC'].max()]
        self.var_limits['Z'] = [self.df['Z_bird_TC'].min(),
                                self.df['Z_bird_TC'].max()]

        self.current_limits['X'] = self.var_limits['X']
        self.current_limits['Y'] = self.var_limits['Y']
        self.current_limits['Z'] = self.var_limits['Z']

    def decomposed_data_preparation(self):
        step = 3
        self.df = self.decomposed_air_velocity_field.df
        self.df = self.df[['bird_name', 'time', # 'iteration',
                           'X', 'Y', 'Z',
                           'X_bird_TC', 'Y_bird_TC', 'Z_bird_TC',
                           #'X_bird_TC_avg', 'Y_bird_TC_avg', 'Z_bird_TC_avg',
                           'rho_bird_TC', 'phi_bird_TC',
                           'wind_X', 'wind_Y']
                          + [f'dXdT_thermal_ground',
                             f'dYdT_thermal_ground',
                             f'dZdT_thermal_ground',
                             'epsilon_X',
                             'epsilon_Y',
                             'epsilon_Z',
                             'V_horizontal_ground',
                             f'bank_angle',
                             f'curvature']
                          + self.bin_index_cols
                          ]

        # self.df = self.df[np.abs(self.df['curvature']) < 0.10]
        # self.df.rename(columns={f'dXdT_air_{step}': 'dXdT_air',
        #                         f'dYdT_air_{step}': 'dYdT_air',
        #                         f'dZdT_air_{step}': 'dZdT_air',
        #                         #f'curvature_{step}': 'curvature'
        #                         }, inplace=True)

        self.df_agg = self.decomposed_air_velocity_field.df_agg



    def set_gui(self):
        plt.interactive(self.interactive)
        self.set_plotting_gui()
        if self.interactive:
            self.set_widget_gui()

    def set_plotting_gui(self):
        self.set_plotting_figure()
        self.set_plotting_axes()

    def set_widget_gui(self):
        self.set_widget_figure()
        self.set_widget_axes()
        self.set_widgets()

    def set_plotting_figure(self):
        self.fig_plots = plt.figure(figsize=(12 + 1, 9 + 1),
                                    #layout='constrained'
                                    )
        if isinstance(self.fig_plots.canvas.manager, FigureManagerTk):
            self.fig_plots.canvas.manager.window.attributes('-zoomed', 1)
        else:
            self.fig_plots.canvas.manager.window.showMaximized()

        self.gridspec_wrapper = self.fig_plots.add_gridspec(1, 2, width_ratios=[1, 3])

        # ===============================        LEFT PANEL    ================================
        self.gridspec_left_panel = GridSpecFromSubplotSpec(3, 3, hspace=0.4, height_ratios=[1, 1, 1],
                                                           width_ratios=[1, 1, 1], wspace=0.2,
                                                           subplot_spec=self.gridspec_wrapper[0])

        # ===============================     RIGHT PANEL =====================================

        self.gridspec_right_panel = GridSpecFromSubplotSpec(3, 3, hspace=0.4, wspace=0.4,
                                                            subplot_spec=self.gridspec_wrapper[1])

    def set_plotting_axes(self):
        # =====================    RIGHT PANEL       ===============================
        self.axes['Contour'] = []
        for j in range(3):
            if not len(self.axes['Contour']):
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[0, j])
            else:
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[0, j],
                                                          sharex=self.axes['Contour'][-1],
                                                          sharey=self.axes['Contour'][-1])
            self.axes['Contour'].append(current_axis)

        self.axes['Section'] = []
        for j in range(3):
            if not len(self.axes['Section']):
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[1, j],
                                                          sharex=self.axes['Contour'][-1],
                                                          sharey=self.axes['Contour'][-1])
            else:
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[1, j],
                                                          sharex=self.axes['Section'][-1],
                                                          sharey=self.axes['Section'][-1])
            self.axes['Section'].append(current_axis)

        self.axes['histogram'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, 0])
        self.axes['radius'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, 1])
        self.axes['rotation'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, 2])
        # ======================    COLORBARS    =================================
        self.colorbar_axes = {'Contour': [], 'Section': []}
        for contour_axis in self.axes['Contour']:
            divider = make_axes_locatable(contour_axis)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            self.colorbar_axes['Contour'].append(cax)

        for contour_axis in self.axes['Section']:
            divider = make_axes_locatable(contour_axis)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            self.colorbar_axes['Section'].append(cax)

        divider = make_axes_locatable(self.axes['histogram'])
        self.colorbar_axes['histogram'] = divider.append_axes('right', size='5%', pad=0.2)

        # =====================    LEFT PANEL       ===============================
        if self.plot_3d:
            self.axes['3D'] = self.fig_plots.add_subplot(self.gridspec_left_panel[0, :], projection='3d')

        if self.debug:
            self.axes['thermal_core'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, 0])
            self.axes['dCoorddZ'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, 1],
                                                              sharey=self.axes['thermal_core'])
            self.axes['vz_max'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, 2],
                                                              sharey=self.axes['dCoorddZ'])

            self.axes['wind'] = self.fig_plots.add_subplot(self.gridspec_left_panel[2, 0],
                                                              sharey=self.axes['thermal_core'])
            self.axes['wind_before'] = self.fig_plots.add_subplot(self.gridspec_left_panel[2, 1],
                                                              sharey=self.axes['wind'])
            self.axes['wind_correction'] = self.fig_plots.add_subplot(self.gridspec_left_panel[2, 2],
                                                              sharey=self.axes['wind_before'])
        else:
            self.axes['thermal_core'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, :])
            self.axes['wind'] = self.fig_plots.add_subplot(self.gridspec_left_panel[2, :],
                                                              sharey=self.axes['thermal_core'])


    def set_widget_figure(self):
        if self.plot_3d:
            self.fig_buttons = plt.figure(figsize=(3.3, 2),
                                          # tight_layout=True
                                          )
        else:
            self.fig_buttons = self.fig_plots

        # ===============================        BUTTONS       ================================

        if self.plot_3d:
            self.gridspec_buttons = self.fig_buttons.add_gridspec(7, 6, hspace=0, wspace=0,
                                                                  width_ratios=[1, 1, 1, 1, 1, 3],
                                                                  )
        else:
            self.gridspec_buttons = GridSpecFromSubplotSpec(7, 6, hspace=0, wspace=0, width_ratios=[1, 1, 1, 1, 1, 3],
                                                            subplot_spec=self.gridspec_left_panel[0, :]
                                                            )

    def set_widget_axes(self):

        # =====================    BUTTONS     ===============================

        self.axes['radio_projection'] = self.fig_buttons.add_subplot(self.gridspec_buttons[0:3, 0:2])
        self.axes['radio_plot'] = self.fig_buttons.add_subplot(self.gridspec_buttons[0:3, 2:5])
        self.axes['button_sweep_iterations'] = self.fig_buttons.add_subplot(self.gridspec_buttons[0, 5])
        self.axes['button_sweep_var'] = self.fig_buttons.add_subplot(self.gridspec_buttons[1, 5])
        self.axes['button_reset'] = self.fig_buttons.add_subplot(self.gridspec_buttons[2, 5])
        self.axes['button_screenshot'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 5])

        self.axes['button_iteration-5'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 0])
        self.axes['button_iteration-1'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 1])
        self.axes['iteration_text'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 2])
        self.axes['button_iteration+1'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 3])
        self.axes['button_iteration+5'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 4])

        for i, coord in enumerate(['X', 'Y', 'Z']):
            self.axes[f'slider_{coord}'] = self.fig_buttons.add_subplot(self.gridspec_buttons[4 + i, 0:5])

        # STYLE
        self.axes['iteration_text'].axis('off')

    def set_widgets(self):
        # ==================================         WIDGET DECLARATION     ===================================#
        self.widgets['radio_projection'] = RadioButtons(self.axes['radio_projection'], ['XY', 'XZ', 'YZ'],
                                                        active=['XY', 'XZ', 'YZ'].index(self.projection_string))

        self.widgets['radio_plot'] = RadioButtons(self.axes['radio_plot'], ['Quiver', 'Contour', 'Stream'],
                                                  active=['Quiver', 'Contour', 'Stream'].index(self.side_plot))

        self.widgets['button_sweep_iterations'] = Button(self.axes['button_sweep_iterations'], label='Sweep Iterations')
        self.widgets['button_sweep_var'] = Button(self.axes['button_sweep_var'], label='Sweep Var')
        self.widgets['button_reset'] = Button(self.axes['button_reset'], label='Reset')
        self.widgets['button_screenshot'] = Button(self.axes['button_screenshot'], label='Scr.Shot')

        self.widgets['button_iteration-5'] = Button(self.axes['button_iteration-5'], label='-5')
        self.widgets['button_iteration-1'] = Button(self.axes['button_iteration-1'], label='-1')
        self.widgets['iteration_text'] = self.axes['iteration_text'].text(0.45, 0.45, f'{self.current_iteration}')
        self.widgets['button_iteration+1'] = Button(self.axes['button_iteration+1'], label='+1')
        self.widgets['button_iteration+5'] = Button(self.axes['button_iteration+5'], label='+5')

        # The +- 0.01 is just to make sure the valinit is inside the upper and lower bounds
        self.widgets['slider_X'] = RangeSlider(ax=self.axes['slider_X'], label='X',
                                               valmin=self.var_limits['X'][0],
                                               valmax=self.var_limits['X'][1])

        self.widgets['slider_Y'] = RangeSlider(ax=self.axes['slider_Y'], label='Y',
                                               valmin=self.var_limits['Y'][0],
                                               valmax=self.var_limits['Y'][1])

        self.widgets['slider_Z'] = RangeSlider(ax=self.axes['slider_Z'], label='Z',
                                               valmin=self.var_limits['Z'][0],
                                               valmax=self.var_limits['Z'][1])

        # ==================================         WIDGET CONFIGURATION     ===================================#
        self.widgets['slider_X'].set_min(self.current_limits['X'][0])
        self.widgets['slider_X'].set_max(self.current_limits['X'][1])
        self.widgets['slider_Y'].set_min(self.current_limits['Y'][0])
        self.widgets['slider_Y'].set_max(self.current_limits['Y'][1])
        self.widgets['slider_Z'].set_min(self.current_limits['Z'][0])
        self.widgets['slider_Z'].set_max(self.current_limits['Z'][1])

        # self.widgets['button_reset'] = Button(self.axes['button_reset'], 'Reset')

        # ===================================    CALLBACKS      ====================================================== #
        self.widgets['slider_X'].on_changed(lambda event: self.on_slider_change(event, 'X'))
        self.widgets['slider_Y'].on_changed(lambda event: self.on_slider_change(event, 'Y'))
        self.widgets['slider_Z'].on_changed(lambda event: self.on_slider_change(event, 'Z'))
        self.widgets['radio_projection'].on_clicked(self.on_radio_projection_clicked)
        self.widgets['radio_plot'].on_clicked(self.on_radio_plot_clicked)

        self.widgets['slider_X'].on_changed(lambda event: self.set_sliced_data())
        self.widgets['slider_Y'].on_changed(lambda event: self.set_sliced_data())
        self.widgets['slider_Z'].on_changed(lambda event: self.set_sliced_data())
        self.widgets['radio_projection'].on_clicked(lambda event: self.set_sliced_data())
        self.widgets['radio_plot'].on_clicked(lambda event: self.set_sliced_data())

        self.widgets['slider_X'].on_changed(lambda event: self.go_plot())
        self.widgets['slider_Y'].on_changed(lambda event: self.go_plot())
        self.widgets['slider_Z'].on_changed(lambda event: self.go_plot())
        self.widgets['radio_projection'].on_clicked(lambda event: self.go_plot())
        self.widgets['radio_plot'].on_clicked(lambda event: self.go_plot())

        self.widgets['button_iteration-5'].on_clicked(lambda event: self.on_iteration_change(event, -5))
        self.widgets['button_iteration-1'].on_clicked(lambda event: self.on_iteration_change(event, -1))
        self.widgets['button_iteration+1'].on_clicked(lambda event: self.on_iteration_change(event, +1))
        self.widgets['button_iteration+5'].on_clicked(lambda event: self.on_iteration_change(event, +5))

        self.widgets['button_sweep_var'].on_clicked(lambda event: self.set_bin())
        self.widgets['button_sweep_var'].on_clicked(lambda event: self.go_plot())

        self.widgets['button_sweep_iterations'].on_clicked(self.sweep_iterations)
        self.widgets['button_sweep_var'].on_clicked(self.sweep_var)
        self.widgets['button_screenshot'].on_clicked(lambda event: self.save_figure())

        circle = Circle((0.8, 0.8), 0.05, facecolor='r', linewidth=3, alpha=1, visible=False)
        self.circle = self.axes['radio_plot'].add_patch(circle)
    #
    # def plot_constant_artists(self):
    #     for col in ['X', 'Y']:
    #         self.axes['thermal_core'].add_artist(copy(self.artists[f'real_thermal_core_{col}']))

    @staticmethod
    def restore_legend(ax, artist_types=None):
        if artist_types is None:
            artist_types = ['lines']
        list_of_artists = []
        for at in artist_types:
            list_of_artists += [a for a in getattr(ax, at)]

        list_of_labels = [a.get_label() for a in list_of_artists]
        ax.legend(list_of_artists, list_of_labels, )
        ax.relim()
        # update ax.viewLim using the new dataLim
        ax.autoscale_view()

    def set_full_iteration(self, iteration):
        self.set_iteration(iteration)
        self.set_iteration_data()
        self.plots_per_iteration()
        self.set_sliced_data()
        self.go_plot()

    def set_iteration(self, iteration):
        self.current_iteration = iteration

        self.widgets['iteration_text'].set_text(f'{self.current_iteration}')

    def on_iteration_change(self, event, increment):
        if self.current_iteration + increment > self.max_iteration:
            self.set_full_iteration(self.current_iteration + increment - self.max_iteration)
        elif self.current_iteration + increment == 0:
            self.set_full_iteration(self.max_iteration)
        elif self.current_iteration + increment < 0:
            self.set_full_iteration(self.max_iteration + self.current_iteration + increment)
        else:
            self.set_full_iteration(self.current_iteration + increment)

    def on_slider_change(self, event, coordinate):

        self.current_limits[coordinate] = [event[0], event[1]]

    def on_radio_projection_clicked(self, event):

        self.projection_string = event
        first_var, second_var, first_index, second_index, section_index, section_var = parse_projection_string(
            self.projection_string)
        self.plotting_X_var = first_var.upper()
        self.plotting_Y_var = second_var.upper()
        self.plotting_section_var = section_var.upper()

        print('plotting_section_var', self.plotting_section_var)

    def on_radio_plot_clicked(self, event):
        self.side_plot = event

    def set_iteration_data(self):
        if self.is_iterative:
            return
        # self.bin()
        #self.current_bins_iteration = self.df_bins[self.df_bins['iteration'] == self.current_iteration]
        self.decomposed_air_velocity_field.set_iteration(self.current_iteration)
        self.decomposed_data_preparation()
        self.df = self.decomposed_air_velocity_field.df
        self.df_agg = self.decomposed_air_velocity_field.df_agg
        self.df = self.df[self.df['in_hull'] & (~self.df['interpolated']) & (np.abs(self.df['curvature']) < 0.1)]
        self.df_agg = self.df_agg[self.df_agg['in_hull'] & (~self.df_agg['interpolated'])]
        self.current_iteration_data = self.df
        self.current_iteration_df_agg = self.df_agg
        current_spline = self.decomposed_air_velocity_field.df_splines.to_dict(orient='records')[0]
        current_thermal_core_spline_parameters = current_spline['thermal_core_positions']


        current_z_array = np.arange(np.min(current_thermal_core_spline_parameters['Y_avg']['tck'][0]),
                                         np.max(current_thermal_core_spline_parameters['Y_avg']['tck'][0]),
                                         10)
        current_thermal_core_xy = self.decomposed_air_velocity_field.get_thermal_core(current_z_array)
        self.current_thermal_core = np.empty(shape=(len(current_z_array), 3))
        self.current_thermal_core[:, 0] = current_thermal_core_xy[:, 0]
        self.current_thermal_core[:, 1] = current_thermal_core_xy[:, 1]
        self.current_thermal_core[:, -1] = current_z_array
        self.current_wind = self.decomposed_air_velocity_field.get_velocity(self.current_thermal_core, t=0, include='wind',
                                                                            relative_to_ground=True, return_components=False)

        if self.debug:
            current_vz_max_parameters = current_spline['vz_max']['dZdT_air_max']
            self.current_vz_max_spline = UnivariateSplineWrapper.from_tck(current_vz_max_parameters['tck'])
            current_wind_correction_parameters = {col: current_spline['wind_correction'][f'wind_correction_{col}']['tck'] for col in ['X', 'Y']}
            self.current_wind_correction_spline = {col: UnivariateSplineWrapper.from_tck(tck)
                                                   for col, tck in current_wind_correction_parameters.items()}

            self.current_vz_max = self.current_vz_max_spline(current_z_array)
            self.current_dCoorddZ = self.decomposed_air_velocity_field.get_thermal_core(current_z_array, d=1)

            self.current_wind_correction = np.stack([current_correction_spline(current_z_array)
                                                 for current_correction_spline in self.current_wind_correction_spline.values()]
                                                    + [np.zeros_like(current_z_array)],
                                                axis=-1)
            self.current_wind_before = self.current_wind - self.current_wind_correction

    def set_bin(self):
        self.current_bin += 1

        print(f'bin {self.current_bin}')
        self.fig_plots.suptitle(f'Bin Z: {self.current_bin}')
        self.current_data = self.current_iteration_data.copy()
        self.current_bins_data = self.current_bins_iteration.copy()

        self.current_data = self.current_data[self.current_data['bin_index_Z_bird_TC'] == self.current_bin]

        self.current_limits['Z'] = self.current_bins_data[['Z_bird_TC_min', 'Z_bird_TC_max']].values[0]
        # self.remove_outliars()

    def set_sliced_data(self):
        self.current_data = self.current_iteration_data.copy()
        self.current_df_agg = self.current_iteration_df_agg.copy()

        for coord, lims in self.current_limits.items():
            self.current_data = self.current_data[self.current_data[f'{coord}_bird_TC'].between(lims[0], lims[1])]
            self.current_df_agg = self.current_df_agg[self.current_df_agg[f'{coord}_bird_TC_bin_avg'].between(lims[0], lims[1])]

        self.remove_outliers()
        self.process_sliced_data()

    def process_sliced_data(self):
        statistic = 'median'
        rho_resolution = 5
        phi_resolution = 1
        rho_max = self.current_data['rho_bird_TC'].max()
        rho_min = self.current_data['rho_bird_TC'].min()

        window_scale = {'phi': 1 / (2 * np.pi) * phi_resolution,
                        'rho': 1 / (rho_max - rho_min) * rho_resolution
                        }

        self.current_data_agg_rolling = {}
        for col in ['phi', 'rho']:
            df_roll = self.current_data[[f'{col}_bird_TC', 'V_horizontal_ground', f'dZdT_thermal_ground']].dropna()
            window_size = round(df_roll[f'{col}_bird_TC'].count() * window_scale[col])
            df_roll = df_roll.sort_values(by=f'{col}_bird_TC')
            self.current_data_agg_rolling[col] = pd.DataFrame()
            self.current_data_agg_rolling[col][f'{col}_bird_TC_avg'] = df_roll[f'{col}_bird_TC'].rolling(window=window_size).agg(statistic)
            self.current_data_agg_rolling[col][f'dZdT_thermal_ground_avg'] = df_roll['dZdT_thermal_ground'].rolling(window=window_size).agg(statistic)
            self.current_data_agg_rolling[col][f'dZdT_thermal_ground_std'] = df_roll['dZdT_thermal_ground'].rolling(window=window_size).agg('std')
            self.current_data_agg_rolling[col]['V_horizontal_ground_avg'] = df_roll['V_horizontal_ground'].rolling(window=window_size).agg(statistic)
            self.current_data_agg_rolling[col]['V_horizontal_ground_std'] = df_roll['V_horizontal_ground'].rolling(window=window_size).agg('std')

            self.current_data_agg_rolling[col].dropna(inplace=True)

    def remove_outliers(self):
        self.current_data = self.current_data[np.abs(self.current_data[f'curvature']) < 0.1]

    def aggregate_data(self, cols_to_bin=None, statistic='mean'):
        if cols_to_bin is None:
            cols_to_bin = ['rho_bird_TC', 'phi_bird_TC']
        bin_edges_dict = {'rho_bird_TC': self.histogram_data['bins']['polar']['rho'],
                          'phi_bird_TC': self.histogram_data['bins']['polar']['phi']}
        # Collapse one direction

        groupby_cols = [f'{self.plotting_X_var}_bird_TC_bin_avg',
                        f'{self.plotting_Y_var}_bird_TC_bin_avg'] + list(bin_edges_dict.keys())

        df_grouped = self.current_data.groupby(by=groupby_cols,
                                               as_index=False)

        # Thermal Real Profile per bin

        df_group = df_grouped.agg({'dXdT_thermal_ground':             statistic,
                                   'epsilon_X': statistic,
                                   'dYdT_thermal_ground':             statistic,
                                   'epsilon_Y': statistic,
                                   'dZdT_thermal_ground':             statistic,
                                   'epsilon_Z': statistic,
                                   f'vis_counts':            statistic}
                                  )


        return df_group

    def plot_wind(self):

        colors = ComparativeDashboard.plot_styles_dict['colors']

        self.axes['wind'].clear()
        for i_col, col in enumerate(['X', 'Y']):
            self.artists['wind'] = self.axes['wind'].plot(self.current_wind[:, i_col],
                                                          self.current_thermal_core[:, -1], colors['calculated'][col],
                                                          label=f'Calc {col}')


        self.axes['wind'].set_title('wind')
        self.axes['wind'].set_xlabel('$V_X$, $V_y$ (m/s)')
        self.axes['wind'].set_ylabel('Z (m)')

    def plot_thermal_core(self):

        colors = {'real':       {'X': 'r--', 'Y': 'y--'},
                  'calculated': {'X': 'b', 'Y': 'g'}}

        self.axes['thermal_core'].clear()

        for i, col in enumerate(['X', 'Y']):
            self.axes['thermal_core'].plot(self.current_thermal_core[:, i],
                                           self.current_thermal_core[:, -1], colors['calculated'][col],
                                           label=f'Calc {col}')

        self.axes['thermal_core'].set_title('Thermal_core')
        self.axes['thermal_core'].set_xlabel('X,Y (m)')
        self.axes['thermal_core'].set_ylabel('Z (m)')
        self.axes['thermal_core'].legend()

    def plot_debug(self):
        colors = ComparativeDashboard.plot_styles_dict['colors']

        self.axes['wind_before'].clear()
        self.axes['wind_correction'].clear()
        self.axes['dCoorddZ'].clear()
        self.axes['vz_max'].clear()
        (self.artists['vz_max'],) = self.axes['vz_max'].plot(self.current_vz_max, self.current_thermal_core[:, -1], )
        #self.axes['vz_max'].axis
        for i_col, col in enumerate(['X', 'Y']):
            self.axes['dCoorddZ'].plot(self.current_dCoorddZ[:, i_col], self.current_thermal_core[:, -1],
                                       colors['calculated'][col],
                                       label=f'Calc {col}')
            self.axes['wind_before'].plot(self.current_wind_before[:, i_col],
                                          self.current_thermal_core[:, -1], colors['calculated'][col],
                                          label=f'Calc {col}')
            self.axes['wind_correction'].plot(self.current_wind_correction[:, i_col],
                                              self.current_thermal_core[:, -1], colors['calculated'][col],
                                              label=f'Calc {col}')
        self.axes['wind_before'].set_title('wind_before')
        self.axes['wind_correction'].set_title('wind_correction')
        self.axes['dCoorddZ'].set_title('dCoorddZ')
        self.axes['vz_max'].set_title('vz_max')
        for ax in [self.axes['dCoorddZ'], self.axes['wind_before'], self.axes['wind_correction'], self.axes['vz_max']]:
            ax.tick_params(
                axis='y',
                which='both',
                right=False,
                left=False,
                labelleft=False)

    def plot_overview(self):
        if not self.plot_3d:
            return
        self.axes['3D'].clear()
        df_plot = self.current_iteration_data.iloc[::10]
        df_plot = df_plot[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC', 'bank_angle']]
        self.artists['3D'] = plot_scatter3D(self.axes['3D'],
                                            X_array=df_plot['X_bird_TC'],
                                            Y_array=df_plot['Y_bird_TC'],
                                            Z_array=df_plot['Z_bird_TC'],
                                            color_array=np.abs(df_plot['bank_angle']),
                                            kwargs={'alpha': 0.2}
                                            )
        if '3D' in self.colorbar_axes.keys():
            self.colorbar_axes['3D'].clear()
            self.fig_plots.colorbar(self.artists['3D'], cax=self.colorbar_axes['3D'], label='Bank Angle (rad)')
        else:
            cb = self.fig_plots.colorbar(self.artists['3D'], ax=self.axes['3D'], location='left',
                                         label='Bank Angle (rad)')
            self.colorbar_axes['3D'] = cb.ax_contour

    def plots_per_iteration(self):

        self.plot_overview()

        self.plot_thermal_core()

        self.plot_wind()
        if self.debug:
            self.plot_debug()
        self.restore_legend(self.axes['wind'])
        self.restore_legend(self.axes['thermal_core'])
        self.fig_plots.suptitle(f'Iteration {self.current_iteration}')

    def plot_shades(self):
        list_of_axes = ['cube', 'slice_thermal_core', 'slice_wind']
        for artist in list_of_axes:
            if artist in self.artists.keys():
                self.artists[artist].remove()
        if self.plot_3d:
            self.artists['cube'] = self.axes['3D'].bar3d(x=self.current_limits['X'][0],
                                                         y=self.current_limits['Y'][0],
                                                         z=self.current_limits['Z'][0],
                                                         dx=self.current_limits['X'][1] - self.current_limits['X'][0],
                                                         dy=self.current_limits['Y'][1] - self.current_limits['Y'][0],
                                                         dz=self.current_limits['Z'][1] - self.current_limits['Z'][0],
                                                         shade=False,
                                                         color='b', alpha=0.1)

        for art in list_of_axes[1:]:
            ax_name = art.replace('slice_', '')
            current_ax = self.axes[ax_name]
            if ax_name == 'wind_diff':
                x_limits = current_ax.get_xlim()
            else:
                x_limits = current_ax.dataLim.intervalx
            self.artists[art] = current_ax.fill_between(
                x=x_limits,
                y1=self.current_limits['Z'][0],
                y2=self.current_limits['Z'][1],
                color='b', alpha=0.1)

    def plots_per_slice(self):

        self.plot_radius()
        self.plot_rotation()
        #self.plot_thermal()
        self.plot_thermal_contour()
        self.plot_thermal_section()
        self.plot_shades()
#        self.plot_white_shadow()

    def set_bins(self):

        n_bins = self.resolution
        h, rho_edges, phi_edges = np.histogram2d(self.current_data['rho_bird_TC'],
                                                 self.current_data['phi_bird_TC'],
                                                 bins=(n_bins // 2, n_bins // 2),
                                                 range=[[0, self.current_data['rho_bird_TC'].max() + 0.01],
                                                        [-np.pi, np.pi + 0.01]])

        rho_edges[0] = 0
        rho_edges[-1] += 0.1  # This ensures that the maximum value is included in the bin
        rho_mg, phi_mg = np.meshgrid(rho_edges, phi_edges)

        x_edges = rho_mg * np.cos(phi_mg)
        y_edges = rho_mg * np.sin(phi_mg)

        self.histogram_data = {'counts': h,
                               'bins':   {
                                   'polar':     {'rho': rho_edges,
                                                 'phi': phi_edges},
                                   'cartesian': {'x': x_edges,
                                                 'y': y_edges}
                               }}

        rho_phi_lower_edges = np.array(list(product(rho_edges[:-1], phi_edges[:-1])))
        rho_phi_upper_edges = np.array(list(product(rho_edges[1:], phi_edges[1:])))
        indices_list = np.array(list(product(range(len(rho_edges) - 1), range(len(phi_edges) - 1))))
        rho_lower_edge = rho_phi_lower_edges[:, 0]
        rho_upper_edge = rho_phi_upper_edges[:, 0]
        phi_lower_edge = rho_phi_lower_edges[:, 1]
        phi_upper_edge = rho_phi_upper_edges[:, 1]

        histogram_data = {'vis_rho_index':       indices_list[:, 0],
                          'vis_phi_index':       indices_list[:, 1],
                          'vis_counts':          h.flatten(),
                          'vis_rho_bird_TC_min': rho_lower_edge,
                          'vis_rho_bird_TC_max': rho_upper_edge,
                          'vis_phi_bird_TC_min': phi_lower_edge,
                          'vis_phi_bird_TC_max': phi_upper_edge
                          }

        self.df_histogram = pd.DataFrame(histogram_data)
        self.df_histogram['vis_rho_bird_TC_avg'] = (self.df_histogram['vis_rho_bird_TC_max'] + self.df_histogram[
            'vis_rho_bird_TC_min']) / 2
        self.df_histogram['vis_phi_bird_TC_avg'] = (self.df_histogram['vis_phi_bird_TC_max'] + self.df_histogram[
            'vis_phi_bird_TC_min']) / 2

        for st in ['avg', 'min', 'max']:
            self.df_histogram[f'vis_x_{st}'] = self.df_histogram[f'vis_rho_bird_TC_{st}'] * np.cos(
                self.df_histogram[f'vis_phi_bird_TC_{st}'])
            self.df_histogram[f'vis_y_{st}'] = self.df_histogram[f'vis_rho_bird_TC_{st}'] * np.sin(
                self.df_histogram[f'vis_phi_bird_TC_{st}'])


        self.current_data['vis_bin_index_rho_bird_TC'] = np.digitize(self.current_data['rho_bird_TC'].values,
                                                                     rho_edges) - 1
        self.current_data['vis_bin_index_phi_bird_TC'] = np.digitize(self.current_data['phi_bird_TC'].values,
                                                                     phi_edges) - 1

        self.current_data = pd.merge(self.current_data, self.df_histogram, how='left',
                                     left_on=['vis_bin_index_rho_bird_TC',
                                              'vis_bin_index_phi_bird_TC'],
                                     right_on=['vis_rho_index',
                                               'vis_phi_index'])

    def plot_histogram(self):
        h = self.histogram_data['counts']
        x_edges = self.histogram_data['bins']['cartesian']['x']
        y_edges = self.histogram_data['bins']['cartesian']['y']

        N = np.ceil(np.max(h)).astype(int)  # +1 to include zero
        cmap = mpl.colormaps['jet']

        cmap_list = [(1, 1, 1, 1)]  # This is white for zero
        cmap_list = cmap_list + [cmap(i) for i in np.linspace(1, cmap.N, N - 1).astype(int)]
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('Custom cmap', cmap_list, N)

        self.artists['histogram'] = self.axes['histogram'].pcolormesh(x_edges, y_edges, h.T, cmap=cmap, snap=False, )

        self.axes['histogram'].set_xlim(np.min(x_edges), np.max(x_edges))
        self.axes['histogram'].set_ylim(np.min(y_edges), np.max(y_edges))

        self.axes['histogram'].set_title('Histogram')
        if self.projection_string == 'XY':
            self.axes['histogram'].set_aspect('equal')
        else:
            self.axes['histogram'].set_aspect('auto')

        self.axes['histogram'].set_xlabel(f'{self.projection_string[0]}(m)')
        self.axes['histogram'].set_ylabel(f'{self.projection_string[1]}(m)')
        if N > 10:
            resolution = 10
        else:
            resolution = N
        labels = np.concatenate([[0], np.linspace(0, N - 1, resolution, endpoint=True, dtype=int)])
        self.fig_plots.colorbar(self.artists['histogram'], cax=self.colorbar_axes['histogram'],
                                ticks=labels
                                )

    def plot_thermal_contour(self):
        for ax in self.axes['Contour'][:3]:
            ax.clear()
        for ax in self.colorbar_axes['Contour'][:3]:
            ax.clear()

        min_n = 5

        self.set_bins()
        self.plot_histogram()
        # Collapse one direction
        my_norm = Normalize()
        var_X = self.plotting_X_var + '_bird_TC'
        var_Y = self.plotting_Y_var + '_bird_TC'

        df_plot = self.current_data[[var_X, var_Y, f'd{self.plotting_section_var}dT_thermal_ground']]
        df_plot = df_plot.dropna()
        m = plot_interpolated(self.axes['Contour'][0], 'contour',
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'd{self.plotting_section_var}dT_thermal_ground'].values,
                               resolution=self.resolution,
                              color_array=False,
                              background_contour_kwargs=False,  kwargs={'norm': my_norm,
                                                                        'cmap': 'gnuplot2'})

        plt.colorbar(m[0], cax=self.colorbar_axes['Contour'][0])

        df_plot = self.current_data[[var_X, var_Y, f'd{self.plotting_section_var}dT_thermal_ground']]
        df_plot = df_plot.dropna()
        ma = plot_interpolated(self.axes['Contour'][1], 'contour',
                               df_plot[var_X].values,
                               df_plot[var_Y].values,
                               df_plot[f'd{self.plotting_section_var}dT_thermal_ground'].values,
                               resolution=self.resolution,
                              color_array=False,  background_contour_kwargs=False,  kwargs={'norm': my_norm,
                                                                                            'cmap': 'gnuplot2'})
        plt.colorbar(ma[0], cax=self.colorbar_axes['Contour'][1])

        df_plot = self.current_data[[f'{var_X}', f'{var_Y}', f'epsilon_{self.plotting_section_var}']]
        df_plot = df_plot.dropna()
        mr = plot_interpolated(self.axes['Contour'][2], 'contour',
                               df_plot[f'{var_X}'].values,
                               df_plot[f'{var_Y}'].values,
                               df_plot[f'epsilon_{self.plotting_section_var}'].values,
                               resolution=self.resolution,
                               color_array=False,  background_contour_kwargs=False, kwargs={'cmap': 'bwr'})
        plt.colorbar(mr[0], cax=self.colorbar_axes['Contour'][2])

        [a.set_aspect('equal') for a in self.axes['Contour'][:3]]
        self.axes['Contour'][0].set_ylabel('Y(m)') # var_Y +
        self.axes['Contour'][0].set_title(f'$V_{{{self.plotting_section_var}}}$ - Raw')
        self.axes['Contour'][1].set_title(f'$V_{{{self.plotting_section_var}}}$ - Average')
        self.axes['Contour'][2].set_title(f'$V_{{{self.plotting_section_var}}}$ - Fluctuation')

    def plot_thermal_section(self):

        for ax in self.axes['Section']:
            ax.clear()
        for ax in self.colorbar_axes['Section']:
            ax.clear()

        min_n = 5

        # Collapse one direction

        var_X = self.plotting_X_var + '_bird_TC'
        var_Y = self.plotting_Y_var + '_bird_TC'
        norm = Normalize()# LogNorm()

        plot_specific_kwargs = {'Quiver': {'pivot': 'mid',
                                           'angles': 'uv',
                                           'scale_units': 'width',
                                           'scale': self.resolution,
                                           'units': 'width',
                                           'width': 0.01,
                                           'constant_length': 1,

                                           },
                                'Contour': {'levels': self.resolution,
                                            #'extend': 'both'
                                            },
                                'Stream': {'color': 'w',
                                           'density': 0.75}}
        current_plot_kwargs = plot_specific_kwargs[self.side_plot]
        current_plot_kwargs.update({'norm': norm, 'cmap': 'gnuplot2'})
        current_bg_kwargs = plot_specific_kwargs['Contour']
        current_bg_kwargs.update({'norm': norm, 'cmap': 'gnuplot2'})



        df_plot = self.current_data[[var_X, var_Y,
                                     f'd{self.plotting_X_var}dT_thermal_ground',
                                     f'd{self.plotting_Y_var}dT_thermal_ground']]
        df_plot = df_plot.dropna()
        m_plot, m_contour = plot_interpolated(self.axes['Section'][0], self.side_plot,
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'd{self.plotting_X_var}dT_thermal_ground'].values,
                              df_plot[f'd{self.plotting_Y_var}dT_thermal_ground'].values,
                               color_array=lambda x: np.linalg.norm(x),
                               resolution=self.resolution,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)

        plt.colorbar(m_contour if m_contour is not None else m_plot, cax=self.colorbar_axes['Section'][0])

        df_plot = self.current_data[[var_X, var_Y,
                                     f'd{self.plotting_X_var}dT_thermal_ground',
                                     f'd{self.plotting_Y_var}dT_thermal_ground']]
        df_plot = df_plot.dropna()
        m_plot, m_contour = plot_interpolated(self.axes['Section'][1], self.side_plot,
                               df_plot[var_X].values,
                               df_plot[var_Y].values,
                              df_plot[f'd{self.plotting_X_var}dT_thermal_ground'].values,
                              df_plot[f'd{self.plotting_Y_var}dT_thermal_ground'].values,
                               resolution=self.resolution,
                               color_array=lambda x: np.linalg.norm(x),
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        plt.colorbar(m_contour if m_contour is not None else m_plot, cax=self.colorbar_axes['Section'][1])

        df_plot = self.current_data[[var_X, var_Y,
                                     f'epsilon_{self.plotting_X_var}',
                                     f'epsilon_{self.plotting_Y_var}']]
        df_plot = df_plot.dropna()
        current_bg_kwargs.update({'norm': Normalize()})
        m_plot, m_contour = plot_interpolated(self.axes['Section'][2], self.side_plot,
                               df_plot[var_X].values,
                               df_plot[var_Y].values,
                               df_plot[f'epsilon_{self.plotting_X_var}'].values,
                               df_plot[f'epsilon_{self.plotting_Y_var}'].values,
                               resolution=self.resolution,
                               color_array=False,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        plt.colorbar(m_contour if m_contour is not None else m_plot, cax=self.colorbar_axes['Section'][2])

        [a.set_aspect('equal') for a in self.axes['Section']]
        self.axes['Section'][0].set_ylabel('Y(m)')
        self.axes['Section'][0].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Raw')
        self.axes['Section'][1].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Average')
        self.axes['Section'][2].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Fluctuation')


    def plot_radius(self):
        self.axes['radius'].clear()

        # =======================================
        #           SCATTER PLOTS
        # =======================================
        # VERTICAL
        self.axes['radius'].plot(self.current_data_agg_rolling['rho']['rho_bird_TC_avg'],
                                    self.current_data_agg_rolling['rho'][f'dZdT_thermal_ground_avg'],
                                    label='MA $V_{Z}$',
                                    **self.plotting_args['moving_average_1'],
                                    )
        self.axes['radius'].scatter(self.current_data['rho_bird_TC'],
                                    self.current_data[f'dZdT_thermal_ground'],
                                    label='Calc. $V_{Z}$',
                                    **self.plotting_args['scatter_1'],
                                    )
        # HORIZONTAL
        self.axes['radius'].plot(self.current_data_agg_rolling['rho']['rho_bird_TC_avg'],
                                 self.current_data_agg_rolling['rho']['V_horizontal_ground_avg'],
                                 label='MA $V_H$',
                                 **self.plotting_args['moving_average_2'], )

        self.axes['radius'].scatter(self.current_data['rho_bird_TC'],
                                    self.current_data['V_horizontal_ground'],
                                    label='Calc $V_H$',
                                    **self.plotting_args['scatter_2'], )
        # ==============================================================================
        #                             MOVING AVERAGES
        # ==============================================================================

        # # VERTICAL
        # self.axes['radius'].plot(self.current_data_rolling['rho']['rho_bird_TC_avg'],
        #                          self.current_data_rolling['rho']['vz_avg'],
        #                          label='M.A. $V_Z$',
        #                          **self.plotting_args['moving_average_1'])
        # # HORIZONTAL
        # self.axes['radius'].plot(self.current_data_rolling['rho']['rho_bird_TC_avg'],
        #                          self.current_data_rolling['rho']['vh_avg'],
        #                          label='M.A. $V_H$',
        #                          **self.plotting_args['moving_average_2'])

        # ==============================================================================
        #           REAL DATA (Get average and std for each rho)
        # ==============================================================================

        # PLOT STYLE
        self.axes['radius'].set_title('Radial Profile')
        self.axes['radius'].set_xlabel('$rho$ (m)')
        self.axes['radius'].set_ylabel('$V_Z$, $V_{H}$ (m/s)')
        #self.axes['radius'].legend(bbox_to_anchor=(1.0, 1))

    def plot_rotation(self, phi_resolution=10):
        self.axes['rotation'].clear()

        # =======================================
        #           SCATTER PLOTS
        # =======================================
        # VERTICAL
        self.axes['rotation'].scatter(self.current_data['phi_bird_TC'],
                                      self.current_data[f'dZdT_thermal_ground'],
                                      **self.plotting_args['scatter_1'])
        self.axes['rotation'].plot(self.current_data_agg_rolling['phi']['phi_bird_TC_avg'],
                                      self.current_data_agg_rolling['phi']['dZdT_thermal_ground_avg'],
                                      **self.plotting_args['moving_average_1'])
        # HORIZONTAL
        self.axes['rotation'].scatter(self.current_data['phi_bird_TC'],
                                    self.current_data['V_horizontal_ground'],
                                    label='Calc $V_H$',
                                    **self.plotting_args['scatter_2'], )
        self.axes['rotation'].plot(self.current_data_agg_rolling['phi']['phi_bird_TC_avg'],
                                      self.current_data_agg_rolling['phi']['V_horizontal_ground_avg'],
                                      **self.plotting_args['moving_average_2'])

        # =======================================
        #       MOVING AVERAGES
        # =======================================
        #
        # # VERTICAL
        # self.axes['rotation'].plot(self.current_data_rolling['phi']['phi_bird_TC_bin_avg'],
        #                            self.current_data_rolling['phi']['vz_avg'],
        #                            **self.plotting_args['moving_average_1'])
        #
        # # HORIZONTAL
        # self.axes['rotation'].plot(self.current_data_rolling['phi']['phi_bird_TC_bin_avg'],
        #                            self.current_data_rolling['phi']['vh_avg'],
        #                            **self.plotting_args['moving_average_2'])

        # PLOT STYLE
        self.axes['rotation'].set_title('Rotation')
        self.axes['rotation'].set_xlabel('$phi$ (rad)')
        self.axes['rotation'].set_ylabel('$V_Z$, $V_H$ (m/s)')

        self.axes['rotation'].yaxis.tick_right()
        self.axes['rotation'].yaxis.set_label_position("right")

    def plot_white_shadow(self):
        # TODO
        # Get rid of edges, somehow...
        current_alpha_value = 0.3
        h = self.histogram_data['counts']
        low_occupation_mask = self.artists['histogram'].get_array() < 3
        low_occupation_mask = low_occupation_mask.astype(float)

        x_edges = self.histogram_data['bins']['cartesian']['x']
        y_edges = self.histogram_data['bins']['cartesian']['y']
        alpha_value = current_alpha_value
        color_array = np.ones((h.shape[0] * h.shape[1], 4))
        color_array[:, -1] = alpha_value * low_occupation_mask.data.astype(float) + 0.0001
        artist = QuadMesh(np.dstack([x_edges, y_edges]), color=color_array, zorder=10)
        for i_ax, ax in enumerate(self.axes['Contour']):
            self.artists['shadow'] = copy(artist)
            ax.add_artist(self.artists['shadow'])
        for i_ax, ax in enumerate(self.axes['Section']):
            self.artists['shadow'] = copy(artist)
            ax.add_artist(self.artists['shadow'])


    def go_plot(self, block=None):
        self.plots_per_slice()
        #self.plot_shades()
        if self.interactive:
            self.show(block=block)
        # self.prepare_next_iteration()

    def show(self, block=None):
        if block is None:
            block = self.block
        self.fig_plots.canvas.draw()
        # self.fig_plots.show()
        if self.interactive:
            self.fig_buttons.canvas.draw()
            plt.show(block=block)

    def save_figure(self, destination_folder, title=None):

        os.makedirs(destination_folder, exist_ok=True)
        if title is None:
            title = self.fig_plots._suptitle.get_text()
            title = title.replace(':', '').replace(' ', '_')

        file_path = f'{destination_folder}/{title}.png'
        try:
            self.fig_plots.savefig(file_path)
        except UserWarning as e:
            print('ad')
        print(f'saved at {file_path}')

    def sweep_iterations(self, event, delay=3):
        if self.animation is None:
            self.current_iteration = 0
            self.animation = FuncAnimation(fig=self.fig_plots, func=self.plot_next_iteration, repeat=False,
                                           interval=delay,
                                           frames=self.max_iteration)
            print('animation started')
            self.fig_plots.canvas.draw()
            self.fig_plots.show()
        else:
            self.animation.pause()
            self.animation = None
            print('animation stopped')

    def plot_next_iteration(self, something):
        self.on_iteration_change(None, 1)
        self.set_iteration_data()
        self.plots_per_iteration()
        self.set_sliced_data()
        self.go_plot()

        return ()

    def sweep_var(self, event, delay=3, binned_var='Z_bird_TC'):
        if self.animation is None:
            list_of_bins = self.current_iteration_data[f'bin_index_{binned_var}'].unique()
            self.animation = FuncAnimation(fig=self.fig_plots, func=self.plot_next_bin, repeat=False, interval=delay,
                                           frames=len(list_of_bins))

            FFwriter = FFMpegWriter()
            print('animation started')
            filename = f'animation{round(time.time())}.mp4'
            self.animation.save(filename, writer=FFwriter)
            print(f'animation saved in {filename}')
            # self.fig_plots.canvas.draw()
            # self.fig_plots.show()
        else:
            self.animation.pause()
            self.animation = None
            self.widgets['slider_Z'].set_min(self.current_limits['Z'][0])
            self.widgets['slider_Z'].set_max(self.current_limits['Z'][1])
            print('animation stopped')

    def update_limits_on_button_figure(self):
        for coord in ['X', 'Y', 'Z']:
            self.widgets[f'slider_{coord}'].set_min(self.current_limits[coord][0])
            self.widgets[f'slider_{coord}'].set_max(self.current_limits[coord][1])

    def plot_next_bin(self, something):
        print(something)
        current_bin = self.current_bins_iteration[self.current_bins_iteration['bin_index_Z_bird_TC'] == something]

        if current_bin.empty:
            return ()
        self.current_limits['Z'] = current_bin[['Z_bird_TC_min', 'Z_bird_TC_max']].values[0].tolist()
        self.set_sliced_data()
        self.go_plot()
        return ()


class ComparativeDashboard(RealDataDashboard):

    def __init__(self,
                 decomposed_air_velocity_field: Union[AirVelocityFieldBase, IterativeReconstructedAirVelocityField],
                 synthetic_air_velocity_field: Union[AirVelocityField,ReconstructedAirVelocityField],
                 sliced_var='Z',
                 resolution=20, block=False, plot_3d=True, interactive=True, side_plot='Stream',
                 average_statistic='median', debug=False):
        super().__init__(decomposed_air_velocity_field, sliced_var=sliced_var, resolution=resolution, block=block,
                         plot_3d=plot_3d, interactive=interactive, side_plot=side_plot,
                         average_statistic=average_statistic, debug=debug)
        self.synthetic_air_velocity_field = synthetic_air_velocity_field

    @classmethod
    def from_path_decomposition(cls, path_to_decomposition, input_folder=None, **kwargs):

        syn, dec = load_synthetic_and_decomposed(path_to_decomposition, input_folder=input_folder)
        df_iter = dec['iterations']

        decomposed_avf = DecomposedAirVelocityField(df=df_iter, df_splines=dec['splines'], df_bins=dec['bins'])

        return ComparativeDashboard(synthetic_air_velocity_field=syn['air_velocity_field'],
                                    decomposed_air_velocity_field=decomposed_avf, **kwargs)
    def prepare_synthetic_data(self):

        self.synthetic_df = self.decomposed_air_velocity_field.df

        self.current_synthetic_data = self.synthetic_df

        synthetic_thermal_core_xy = self.synthetic_air_velocity_field.get_thermal_core(self.current_thermal_core[:, -1])
        self.synthetic_thermal_core = np.empty(shape=(len(self.current_thermal_core[:, -1]), 3))
        self.synthetic_thermal_core[:, 0] = synthetic_thermal_core_xy[:, 0]
        self.synthetic_thermal_core[:, 1] = synthetic_thermal_core_xy[:, 1]
        self.synthetic_thermal_core[:, -1] = self.current_thermal_core[:, -1]
        self.synthetic_wind = self.synthetic_air_velocity_field.get_velocity(self.synthetic_thermal_core, t=0,
                                                                                     include='wind',
                                                                                     relative_to_ground=True,
                                                                                     return_components=False)
        #
        # if self.debug:
        #     current_vz_max_parameters = current_spline['vz_max']['dZdT_air_max']
        #     self.current_vz_max_spline = UnivariateSplineWrapper.from_tck(current_vz_max_parameters['tck'])
        #     self.current_vz_max = self.current_vz_max_spline(current_z_array)
        #     self.current_dCoorddZ = self.decomposed_air_velocity_field.get_thermal_core(current_z_array, d=1)
        #
        #     self.current_wind_before = self.decomposed_air_velocity_field.get_velocity(self.current_thermal_core, t=0,
        #                                                                                include='wind',
        #                                                                                corrected=False,
        #                                                                                relative_to_ground=True,
        #                                                                                return_components=False)
        #     self.current_wind_correction = self.current_wind - self.current_wind_before

    def set_synthetic_sliced_data(self):

        self.current_synthetic_data = self.current_synthetic_data.copy()

        for coord, lims in self.current_limits.items():
            self.current_synthetic_data = self.current_synthetic_data[self.current_synthetic_data[f'{coord}_bird_TC'].between(lims[0], lims[1])]

        # self.current_synthetic_data_agg = self.current_df_agg[[f'{coord}_bird_TC_avg' for coord, lims in self.current_limits.items()]]
        # self.current_synthetic_data_agg
        self.remove_outliers()
        self.process_sliced_data()

    def process_synthetic_sliced_data(self):
        statistic = 'median'
        rho_resolution = 5
        phi_resolution = 1
        rho_max = self.current_synthetic_data['rho_bird_TC'].max()
        rho_min = self.current_synthetic_data['rho_bird_TC'].min()

        df_roll = self.current_synthetic_data[['rho_bird_TC',
                                               'V_horizontal_ground',
                                               f'dZdT_thermal_ground']]

        df_roll = df_roll.sort_values(by='rho_bird_TC')
        window_size = df_roll['rho_bird_TC'].count() / (rho_max - rho_min) * rho_resolution
        window_size = round(window_size)

        self.current_synthetic_data_rolling_in_rho = df_roll.rolling(window=window_size,
                                                                     center=True).agg({'rho_bird_TC':  statistic,
                                                                             f'dZdT_thermal_ground':    (statistic, 'std'),
                                                                             'V_horizontal_ground': (statistic, 'std')}
                                                                            ).dropna()

        self.current_synthetic_data_rolling_in_rho['rho_bird_TC_avg'] = self.current_synthetic_data_rolling_in_rho['rho_bird_TC'][statistic]
        self.current_synthetic_data_rolling_in_rho['vz_avg'] = self.current_synthetic_data_rolling_in_rho[f'dZdT_thermal_ground'][statistic]
        self.current_synthetic_data_rolling_in_rho['vz_std'] = self.current_synthetic_data_rolling_in_rho[f'dZdT_thermal_ground']['std']
        self.current_synthetic_data_rolling_in_rho['vh_avg'] = self.current_synthetic_data_rolling_in_rho['V_horizontal_ground'][statistic]
        self.current_synthetic_data_rolling_in_rho['vh_std'] = self.current_synthetic_data_rolling_in_rho['V_horizontal_ground']['std']
        self.current_synthetic_data_rolling_in_rho.drop(columns=['rho_bird_TC',
                                                       f'dZdT_thermal_ground',
                                                       'V_horizontal_ground'], inplace=True)
        self.current_synthetic_data_rolling_in_rho.columns = self.current_synthetic_data_rolling_in_rho.columns.droplevel(1)

        window_scale = {'phi': 1 / (2 * np.pi) * phi_resolution,
                        'rho': 1 / (rho_max - rho_min) * rho_resolution
                        }

        self.current_synthetic_data_rolling = {}
        for col in ['phi', 'rho']:
            df_roll = self.current_synthetic_data[[f'{col}_bird_TC',
                                         'V_horizontal_ground',
                                         f'dZdT_thermal_ground']]
            window_size = round(df_roll[f'{col}_bird_TC'].count() * window_scale[col])
            df_roll = df_roll.sort_values(by=f'{col}_bird_TC')
            self.current_synthetic_data_rolling[col] = df_roll.rolling(window=window_size).agg(
                {f'{col}_bird_TC': statistic,
                 f'dZdT_thermal_ground':      (statistic, 'std'),
                 'V_horizontal_ground':   (statistic, 'std')}).dropna()

            self.current_synthetic_data_rolling[col][f'{col}_bird_TC_avg'] = self.current_synthetic_data_rolling[col][f'{col}_bird_TC'][
                statistic]
            self.current_synthetic_data_rolling[col]['vz_avg'] = self.current_synthetic_data_rolling[col][f'dZdT_thermal_ground'][statistic]
            self.current_synthetic_data_rolling[col]['vz_std'] = self.current_synthetic_data_rolling[col][f'dZdT_thermal_ground']['std']
            self.current_synthetic_data_rolling[col]['vh_avg'] = self.current_synthetic_data_rolling[col]['V_horizontal_ground'][statistic]
            self.current_synthetic_data_rolling[col]['vh_std'] = self.current_synthetic_data_rolling[col]['V_horizontal_ground']['std']
            self.current_synthetic_data_rolling[col].drop(columns=[f'{col}_bird_TC',
                                                         f'dZdT_thermal_ground',
                                                         'V_horizontal_ground'], inplace=True)
            self.current_synthetic_data_rolling[col].columns = self.current_synthetic_data_rolling[col].columns.droplevel(1)

    def initial_setup(self):
        self.avf_vis = AirVelocityFieldVisualization(self.decomposed_air_velocity_field)

        self.data_preparation()
        self.set_iteration_data()

        self.set_sliced_data()
        self.prepare_synthetic_data()
        self.set_synthetic_sliced_data()
        self.process_synthetic_sliced_data()
        self.set_gui()
        self.set_synthetic_axes()

        self.plots_per_iteration()
        self.go_plot()

    def set_iteration_data(self):
        super().set_iteration_data()

        self.current_iteration_data[['dXdT_air_real',
                 'dYdT_air_real',
                 'dZdT_air_real']] = self.synthetic_air_velocity_field.get_velocity(X=self.current_iteration_data[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC']].values,
                                                                                    t=0,
                                                                                    include=['thermal', 'rotation'],
                                                                                    relative_to_ground=False)
        self.current_iteration_data[['dXdT_air_turbulence_real',
                                     'dYdT_air_turbulence_real',
                                     'dZdT_air_turbulence_real']] = self.synthetic_air_velocity_field.get_velocity(X=self.current_iteration_data[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC']].values,
                                                                                    t=0,
                                                                                    include=['turbulence'],
                                                                                    relative_to_ground=False)
        self.current_iteration_df_agg[['dXdT_air_real',
                                       'dYdT_air_real',
                                       'dZdT_air_real']] = self.synthetic_air_velocity_field.get_velocity(X=self.current_iteration_df_agg[['X_bird_TC_bin_avg', 'Y_bird_TC_bin_avg', 'Z_bird_TC_bin_avg']].values,
                                                                                    t=0,
                                                                                    include=['thermal', 'rotation'],
                                                                                    relative_to_ground=False)

        self.current_iteration_data['diff_dXdT_thermal_ground'] = self.current_iteration_data['dXdT_air_real'] - self.current_iteration_data['dXdT_thermal_ground']
        self.current_iteration_data['diff_dYdT_thermal_ground'] = self.current_iteration_data['dYdT_air_real'] - self.current_iteration_data['dYdT_thermal_ground']
        self.current_iteration_data['diff_dZdT_thermal_ground'] = self.current_iteration_data['dZdT_air_real'] - self.current_iteration_data['dZdT_thermal_ground']
        self.current_iteration_data['diff_dXdT_thermal_ground_turbulence'] = self.current_iteration_data['dXdT_air_turbulence_real'] - self.current_iteration_data['epsilon_X']
        self.current_iteration_data['diff_dYdT_thermal_ground_turbulence'] = self.current_iteration_data['dYdT_air_turbulence_real'] - self.current_iteration_data['epsilon_Y']
        self.current_iteration_data['diff_dZdT_thermal_ground_turbulence'] = self.current_iteration_data['dZdT_air_turbulence_real'] - self.current_iteration_data['epsilon_Z']
        # self.current_iteration_df_agg['diff_dXdT_thermal_ground_avg'] = self.current_iteration_df_agg['dXdT_air_real'] - self.current_iteration_df_agg['dXdT_thermal_ground_avg']
        # self.current_iteration_df_agg['diff_dYdT_thermal_ground_avg'] = self.current_iteration_df_agg['dYdT_air_real'] - self.current_iteration_df_agg['dYdT_thermal_ground_avg']
        # self.current_iteration_df_agg['diff_dZdT_thermal_ground_avg'] = self.current_iteration_df_agg['dZdT_air_real'] - self.current_iteration_df_agg['dZdT_thermal_ground_avg']

    def set_plotting_figure(self):
        self.fig_plots = plt.figure(figsize=(19, 12),
                                    #tight_layout=True
                                    )
        if self.interactive:
            if isinstance(self.fig_plots.canvas.manager, FigureManagerTk):
                self.fig_plots.canvas.manager.window.attributes('-zoomed', 1)
            else:
                self.fig_plots.canvas.manager.window.showMaximized()

        self.gridspec_wrapper = self.fig_plots.add_gridspec(1, 3, width_ratios=[1, 1, 5],
                                                            hspace=0.4,wspace=0.1,
                                                            left=0.05, right=0.95, top=0.95, bottom=0.05)

        # ===============================        LEFT PANEL    ================================
        self.gridspec_left_panel = GridSpecFromSubplotSpec(3, 3, hspace=0.2, height_ratios=[1, 1, 1],
                                                           width_ratios=[1, 1, 1], wspace=0.2,
                                                           subplot_spec=self.gridspec_wrapper[0])

        # ===============================     CENTER PANEL =====================================

        self.gridspec_center_panel = GridSpecFromSubplotSpec(3, 1, hspace=0.2, wspace=0.2,
                                                            subplot_spec=self.gridspec_wrapper[1])
        # ===============================     RIGHT PANEL =====================================

        self.gridspec_right_panel = GridSpecFromSubplotSpec(4, 4, hspace=0.4, wspace=0.2,
                                                            subplot_spec=self.gridspec_wrapper[2])

    def set_synthetic_axes(self):

        self.axes['histogram'].remove()
        self.axes['radius'].remove()
        self.axes['rotation'].remove()
        self.axes['histogram'] = self.fig_plots.add_subplot(self.gridspec_center_panel[0])
        self.colorbar_axes['histogram'].remove()

        divider = make_axes_locatable(self.axes['histogram'])
        self.colorbar_axes['histogram'] = divider.append_axes('right', size='5%', pad=0.2)
        self.axes['radius'] = self.fig_plots.add_subplot(self.gridspec_center_panel[1])
        self.axes['rotation'] = self.fig_plots.add_subplot(self.gridspec_center_panel[2])
        self.axes['synthetic'] = {}
        self.axes['synthetic']['Contour'] = self.fig_plots.add_subplot(self.gridspec_right_panel[0, -1],
                                                                    sharex=self.axes['Contour'][-1],
                                                                    sharey=self.axes['Contour'][-1])
        self.axes['synthetic']['Section'] = self.fig_plots.add_subplot(self.gridspec_right_panel[1, -1],
                                                                    sharex=self.axes['synthetic']['Contour'],
                                                                    sharey=self.axes['synthetic']['Contour'])
        self.axes['synthetic']['Turbulence_Z'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, -1],
                                                                         sharex=self.axes['synthetic']['Section'],
                                                                         sharey=self.axes['synthetic']['Section'])
        self.axes['synthetic']['Turbulence_Horizontal'] = self.fig_plots.add_subplot(self.gridspec_right_panel[3, -1],
                                                                         sharex=self.axes['synthetic']['Turbulence_Z'],
                                                                         sharey=self.axes['synthetic']['Turbulence_Z'])

        self.colorbar_axes['synthetic'] = {}
        for k, syn_ax in self.axes['synthetic'].items():
            divider = make_axes_locatable(syn_ax)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            self.colorbar_axes['synthetic'][k] = cax

        self.axes['Diff_Contour'] = []
        for j in range(3):
            if not len(self.axes['Diff_Contour']):
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[2, j])
            else:
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[2, j],
                                                          sharex=self.axes['Diff_Contour'][-1],
                                                          sharey=self.axes['Diff_Contour'][-1])
            self.axes['Diff_Contour'].append(current_axis)

        self.axes['Diff_Section'] = []
        for j in range(3):
            if not len(self.axes['Diff_Section']):
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[3, j],
                                                          sharex=self.axes['Diff_Contour'][-1],
                                                          sharey=self.axes['Diff_Contour'][-1])
            else:
                current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[3, j],
                                                          sharex=self.axes['Diff_Section'][-1],
                                                          sharey=self.axes['Diff_Section'][-1])
            self.axes['Diff_Section'].append(current_axis)

        # ======================    COLORBARS    =================================
        self.colorbar_axes['Diff_Contour'] = []
        self.colorbar_axes['Diff_Section'] = []
        for contour_axis in self.axes['Diff_Contour']:
            divider = make_axes_locatable(contour_axis)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            self.colorbar_axes['Diff_Contour'].append(cax)

        for contour_axis in self.axes['Diff_Section']:
            divider = make_axes_locatable(contour_axis)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            self.colorbar_axes['Diff_Section'].append(cax)


    def plots_per_iteration(self):
        super().plots_per_iteration()
        self.plot_synthetic_thermal_core()
        self.plot_synthetic_wind()
        if self.debug:
            self.plot_synthetic_debug()

    def plot_synthetic_debug(self):

        colors = ComparativeDashboard.plot_styles_dict['colors']['real']

        #self.axes['wind'].clear()
        for i_col, col in enumerate(['X', 'Y']):
            self.artists['syn_wind_before'] = self.axes['wind_before'].plot(self.synthetic_wind[:, i_col],
                                                          self.synthetic_thermal_core[:, -1], colors[col],
                                                          label=f'Real {col}')

    def plots_per_slice(self):
        super().plots_per_slice()
        #self.plot_synthetic_radius()
        #self.plot_synthetic_rotation()
        #self.plotsynthetic__thermal()
        self.plot_synthetic_thermal_contour()
        self.plot_synthetic_thermal_section()
        self.plot_diff_thermal_contour()
        self.plot_diff_thermal_section()
        #self.plot_synthetic_shades()

#        self.plot_white_shadow()
    def plot_synthetic_wind(self):

        colors = ComparativeDashboard.plot_styles_dict['colors']['real']

        #self.axes['wind'].clear()
        for i_col, col in enumerate(['X', 'Y']):
            self.artists['wind'] = self.axes['wind'].plot(self.synthetic_wind[:, i_col],
                                                          self.synthetic_thermal_core[:, -1], colors[col],
                                                          label=f'Real {col}')

        self.axes['wind'].set_title('wind')
        self.axes['wind'].set_xlabel('$V_X$, $V_y$ (m/s)')
        self.axes['wind'].set_ylabel('Z (m)')
        #self.axes['wind'].legend()

    def plot_synthetic_thermal_core(self):

        colors = ComparativeDashboard.plot_styles_dict['colors']['real']

        #self.axes['thermal_core'].clear()

        for i, col in enumerate(['X', 'Y']):
            self.axes['thermal_core'].plot(self.synthetic_thermal_core[:, i],
                                           self.synthetic_thermal_core[:, -1], colors[col],
                                           label=f'Real {col}')

        self.axes['thermal_core'].set_title('Thermal_core')
        self.axes['thermal_core'].set_xlabel('X,Y (m)')
        self.axes['thermal_core'].set_ylabel('Z (m)')
        self.axes['thermal_core'].legend()


    def plot_synthetic_thermal_contour(self):
        self.axes['synthetic']['Contour'].clear()
        self.colorbar_axes['synthetic']['Contour'].clear()
        self.axes['synthetic']['Turbulence_Z'].clear()
        self.colorbar_axes['synthetic']['Turbulence_Z'].clear()

        min_n = 5

        # Collapse one direction
        my_norm = Normalize()
        var_X = self.plotting_X_var + '_bird_TC'
        var_Y = self.plotting_Y_var + '_bird_TC'

        df_plot = self.current_data.loc[~self.current_data['dYdT_thermal_ground'].isna(), [var_X, var_Y, 'Z_bird_TC',
                                     'dXdT_thermal_ground', 'dYdT_thermal_ground', 'dZdT_thermal_ground',
                                     'dXdT_air_real', 'dYdT_air_real', 'dZdT_air_real',
                                     'dXdT_air_turbulence_real',
                                     'dYdT_air_turbulence_real',
                                     'dZdT_air_turbulence_real'
                                     ]]
        df_plot = df_plot.dropna()
        df_plot = df_plot
        m = plot_interpolated(self.axes['synthetic']['Contour'], 'contour',
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'd{self.plotting_section_var}dT_air_real'].values,
                               resolution=self.resolution,
                              color_array=False,
                              background_contour_kwargs=False,  kwargs={'norm': my_norm,
                                                                        'cmap': 'gnuplot2'})

        plt.colorbar(m[0], cax=self.colorbar_axes['synthetic']['Contour'])

        m = plot_interpolated(self.axes['synthetic']['Turbulence_Z'], 'contour',
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'd{self.plotting_section_var}dT_air_turbulence_real'].values,
                               resolution=self.resolution,
                              color_array=False,
                              background_contour_kwargs=False,  kwargs={'norm': TwoSlopeNorm(vcenter=0),
                                                                        'cmap': 'bwr'})

        plt.colorbar(m[0], cax=self.colorbar_axes['synthetic']['Turbulence_Z'])

        self.axes['synthetic']['Contour'].set_aspect('equal')
        self.axes['synthetic']['Turbulence_Z'].set_aspect('equal')

        self.axes['synthetic']['Contour'].set_title(f'$V_{{{self.plotting_section_var}}}$ - Ground Truth')
        self.axes['synthetic']['Turbulence_Z'].set_title(
            f'Turbulence ${{{self.plotting_section_var}}}$ - Ground Truth')

    def plot_synthetic_thermal_section(self):
        self.axes['synthetic']['Section'].clear()
        self.axes['synthetic']['Turbulence_Horizontal'].clear()
        self.colorbar_axes['synthetic']['Section'].clear()
        self.colorbar_axes['synthetic']['Turbulence_Horizontal'].clear()
        min_n = 5

        # Collapse one direction

        var_X = self.plotting_X_var + '_bird_TC'
        var_Y = self.plotting_Y_var + '_bird_TC'
        norm = Normalize()# LogNorm()

        plot_specific_kwargs = {'Quiver': {'pivot': 'mid',
                                           'angles': 'uv',
                                           'scale_units': 'width',
                                           #'scale': self.resolution,
                                           'units': 'width',
                                           'width': 0.01,
                                           'constant_length': 1,

                                           },
                                'Contour': {'levels': self.resolution,
                                            #'extend': 'both'
                                            },
                                'Stream': {'color': 'w',
                                           'density': 0.75}}
        current_plot_kwargs = plot_specific_kwargs[self.side_plot]
        current_plot_kwargs.update({'norm': norm, 'cmap': 'gnuplot2'})
        current_bg_kwargs = plot_specific_kwargs['Contour']
        current_bg_kwargs.update({'norm': norm, 'cmap': 'gnuplot2'})


        df_plot = self.current_data.loc[~self.current_data['dYdT_thermal_ground'].isna(),
        [var_X, var_Y, 'Z_bird_TC',
         'dXdT_air_real',
                 'dYdT_air_real',
                 'dZdT_air_real','dXdT_air_turbulence_real',
                 'dYdT_air_turbulence_real',
                 'dZdT_air_turbulence_real']]


        df_plot = df_plot.dropna()
        m_plot, m_contour = plot_interpolated(self.axes['synthetic']['Section'], self.side_plot,
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'd{self.plotting_X_var}dT_air_real'].values,
                              df_plot[f'd{self.plotting_Y_var}dT_air_real'].values,
                               color_array=lambda x: np.linalg.norm(x),
                               resolution=self.resolution,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        if m_contour is not None:
            plt.colorbar(m_contour, cax=self.colorbar_axes['synthetic']['Section'])
        elif m_plot is not None:
            plt.colorbar(m_plot, cax=self.colorbar_axes['synthetic']['Section'])
        current_bg_kwargs.update({'norm': Normalize()})

        m_plot, m_contour = plot_interpolated(self.axes['synthetic']['Turbulence_Horizontal'], self.side_plot,
                               df_plot[var_X].values,
                               df_plot[var_Y].values,
                               df_plot[f'd{self.plotting_X_var}dT_air_turbulence_real'].values,
                               df_plot[f'd{self.plotting_Y_var}dT_air_turbulence_real'].values,
                               resolution=self.resolution,
                               color_array=False,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        if m_contour is not None:
            plt.colorbar(m_contour, cax=self.colorbar_axes['synthetic']['Turbulence_Horizontal'])
        elif m_plot is not None:
            plt.colorbar(m_plot, cax=self.colorbar_axes['synthetic']['Turbulence_Horizontal'])

        self.axes['synthetic']['Section'].set_aspect('equal')
        self.axes['synthetic']['Turbulence_Horizontal'].set_aspect('equal')

        self.axes['synthetic']['Section'].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Ground Truth')
        self.axes['synthetic']['Turbulence_Horizontal'].set_title(
            f'Turbulence $_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Ground Truth')


    def plot_diff_thermal_contour(self):
        for a in self.axes['Diff_Contour']:
            a.clear()

        for ax in self.colorbar_axes['Diff_Contour']:
            ax.clear()

        min_n = 5

        # Collapse one direction
        my_norm = Normalize()
        var_X = self.plotting_X_var + '_bird_TC'
        var_Y = self.plotting_Y_var + '_bird_TC'

        df_plot = self.current_data[[var_X, var_Y] + ['diff_dZdT_thermal_ground']]


        df_plot = df_plot.dropna()
        m = plot_interpolated(self.axes['Diff_Contour'][0], 'contour',
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'diff_d{self.plotting_section_var}dT_thermal_ground'].values,
                              resolution=self.resolution,
                              color_array=False,
                              background_contour_kwargs=False, kwargs={'norm': my_norm,
                                                                       'cmap': 'gnuplot2'})

        plt.colorbar(m[0], cax=self.colorbar_axes['Diff_Contour'][0])

        df_plot = self.current_data[[var_X, var_Y] + ['diff_dZdT_thermal_ground']]



        df_plot = df_plot.dropna()
        m = plot_interpolated(self.axes['Diff_Contour'][1], 'contour',
                              df_plot['X_bird_TC'].values,
                              df_plot['Y_bird_TC'].values,
                              df_plot[f'diff_d{self.plotting_section_var}dT_thermal_ground'].values,
                              resolution=self.resolution,
                              color_array=False,
                              background_contour_kwargs=False, kwargs={'norm': my_norm,
                                                                       'cmap': 'gnuplot2'})

        plt.colorbar(m[0], cax=self.colorbar_axes['Diff_Contour'][1])

        df_plot = self.current_data[[f'{var_X}', f'{var_Y}', f'diff_d{self.plotting_section_var}dT_thermal_ground_turbulence']]
        df_plot = df_plot.dropna()
        mr = plot_interpolated(self.axes['Diff_Contour'][2], 'contour',
                               df_plot[f'{var_X}'].values,
                               df_plot[f'{var_Y}'].values,
                               df_plot[f'diff_d{self.plotting_section_var}dT_thermal_ground_turbulence'].values,
                               resolution=self.resolution,
                               color_array=False,  background_contour_kwargs=False, kwargs={'cmap': 'bwr'})
        plt.colorbar(mr[0], cax=self.colorbar_axes['Diff_Contour'][2])


        self.axes['Diff_Contour'][0].set_title(f'Diff $V_{{{self.plotting_section_var}}}$ - Raw')
        self.axes['Diff_Contour'][1].set_title(f'Diff $V_{{{self.plotting_section_var}}}$ - Average')
        self.axes['Diff_Contour'][2].set_title(f'Diff $V_{{{self.plotting_section_var}}}$ - Fluctuation')
        self.axes['Diff_Contour'][0].set_ylabel('Y(m)') # var_Y +
        #
        for ax in self.axes['Diff_Contour']:
            ax.set_aspect('equal')


    def plot_diff_thermal_section(self):

        for a in self.axes['Diff_Section']:
            a.clear()

        for ax in self.colorbar_axes['Diff_Section']:
            ax.clear()

        min_n = 5

        # Collapse one direction

        var_X = self.plotting_X_var + '_bird_TC'
        var_Y = self.plotting_Y_var + '_bird_TC'
        norm = Normalize()# LogNorm()

        plot_specific_kwargs = {'Quiver': {'pivot': 'mid',
                                           'angles': 'uv',
                                           'scale_units': 'width',
                                           'scale': self.resolution,
                                           'units': 'width',
                                           'width': 0.01,
                                           'constant_length': 1,

                                           },
                                'Contour': {'levels': self.resolution,
                                            #'extend': 'both'
                                            },
                                'Stream': {'color': 'w',
                                           'density': 0.75}}
        current_plot_kwargs = plot_specific_kwargs[self.side_plot]
        current_plot_kwargs.update({'norm': norm, 'cmap': 'gnuplot2'})
        current_bg_kwargs = plot_specific_kwargs['Contour']
        current_bg_kwargs.update({'norm': norm, 'cmap': 'gnuplot2'})

        df_plot = self.current_data.loc[~self.current_data['dYdT_thermal_ground'].isna(), [var_X, var_Y] + [f'diff_d{self.plotting_X_var}dT_thermal_ground',
                                                      f'diff_d{self.plotting_Y_var}dT_thermal_ground']]
        df_plot = df_plot.dropna()
        m_plot, m_contour = plot_interpolated(self.axes['Diff_Section'][0], self.side_plot,
                              df_plot[var_X].values,
                              df_plot[var_Y].values,
                              df_plot[f'diff_d{self.plotting_X_var}dT_thermal_ground'].values,
                              df_plot[f'diff_d{self.plotting_Y_var}dT_thermal_ground'].values,
                               color_array=lambda x: np.linalg.norm(x),
                               resolution=self.resolution,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        if m_contour is not None:
            plt.colorbar(m_contour, cax=self.colorbar_axes['Diff_Section'][0])
        elif m_plot is not None:
            plt.colorbar(m_plot, cax=self.colorbar_axes['Diff_Section'][0])

        df_plot = self.current_data.loc[~self.current_data['dYdT_thermal_ground'].isna(), [var_X ,
                                                                                  var_Y]
                                                                                 + [f'diff_d{self.plotting_X_var}dT_thermal_ground',
                                                                                    f'diff_d{self.plotting_Y_var}dT_thermal_ground']
        ]
        df_plot = df_plot.dropna()
        m_plot, m_contour = plot_interpolated(self.axes['Diff_Section'][1], self.side_plot,
                               df_plot[var_X].values,
                               df_plot[var_Y].values,
                               df_plot[f'diff_d{self.plotting_X_var}dT_thermal_ground'].values,
                               df_plot[f'diff_d{self.plotting_Y_var}dT_thermal_ground'].values,
                               resolution=self.resolution,
                               color_array=False,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        if m_contour is not None:
            plt.colorbar(m_contour, cax=self.colorbar_axes['Diff_Section'][1])
        elif m_plot is not None:
            plt.colorbar(m_plot, cax=self.colorbar_axes['Diff_Section'][1])

        df_plot = self.current_data.loc[~self.current_data['dYdT_thermal_ground'].isna(), [var_X, var_Y] + [f'diff_d{self.plotting_X_var}dT_thermal_ground_turbulence',
                                                      f'diff_d{self.plotting_Y_var}dT_thermal_ground_turbulence']]
        df_plot = df_plot.dropna()

        current_bg_kwargs.update({'norm': Normalize()})
        m_plot, m_contour = plot_interpolated(self.axes['Diff_Section'][2], self.side_plot,
                               df_plot[var_X].values,
                               df_plot[var_Y].values,
                               df_plot[f'diff_d{self.plotting_X_var}dT_thermal_ground_turbulence'].values,
                               df_plot[f'diff_d{self.plotting_Y_var}dT_thermal_ground_turbulence'].values,
                               resolution=self.resolution,
                               color_array=False,
                              background_contour_kwargs=current_bg_kwargs if self.side_plot != 'Contour' else False,
                              kwargs=current_plot_kwargs)
        if m_contour is not None:
            plt.colorbar(m_contour, cax=self.colorbar_axes['Diff_Section'][2])
        elif m_plot is not None:
            plt.colorbar(m_plot, cax=self.colorbar_axes['Diff_Section'][2])

        self.axes['Diff_Section'][0].set_title(f'Diff $V_H$ - Raw')
        self.axes['Diff_Section'][1].set_title(f'Diff $V_H$ - Average')
        self.axes['Diff_Section'][2].set_title(f'Diff $V_H$ - Fluctuation')

        for ax in self.axes['Diff_Section']:
            ax.set_aspect('equal')
            ax.set_xlabel(var_X + '(m)')
        self.axes['Diff_Section'][0].set_ylabel('Y(m)') #var_Y +

        #
        # self.axes['Synthetic_Section'][0].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Raw')
        # self.axes['Synthetic_Section'][1].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Average')
        # self.axes['Synthetic_Section'][2].set_title(f'$V_{{{self.plotting_X_var},{self.plotting_Y_var}}}$ - Fluctuation')
#
# class ComparativeDashboard2:
#     plot_styles_dict = {'colors': {'real':       {'X': 'r--', 'Y': 'y--'},
#                                    'calculated': {'X': 'b', 'Y': 'g'}}
#                         }
#     config = {'Z_padding': 100}
#
#     def __init__(self, df, df_splines, df_ground_truth, air_velocity_field, sliced_var='Z',
#                  resolution=20, block=False, plot_3d=True, interactive=True, side_plot='Stream',
#                  average_statistic='median'):
#
#         # Data Attributes
#         self.df_histogram = None
#         self.real_wind = None
#         self.plot_3d = plot_3d
#         self.df = df  # .copy()
#         self.df_ground_truth = df_ground_truth
#
#         if not isinstance(df_splines, pd.DataFrame):
#             df_splines = pd.DataFrame.from_dict(df_splines)
#         self.df_splines = df_splines
#
#         if isinstance(air_velocity_field, AirVelocityField):
#             self.air_velocity_field = air_velocity_field
#             self.air_parameters = self.air_velocity_field._air_parameters.copy()
#         else:
#             self.air_parameters = air_velocity_field.copy()
#             self.air_velocity_field = AirVelocityField(self.air_parameters)
#         self.real_thermal_core = None
#         self.block = block
#         self.interactive = interactive
#
#         # Figure, axes, artists
#         self.fig_plots = None
#         self.fig_buttons = None
#         self.axes = {}
#         self.colorbar_axes = {}
#         self.artists = {}
#         self.widgets = {}
#         self.gridspec_wrapper = None
#         self.gridspec_buttons = None
#         self.gridspec_right_panel = None
#         self.gridspec_left_panel = None
#         self.circle = None
#
#         # STATE VARIABLES
#         self.projection_string = 'XY'
#         first_var, second_var, first_index, second_index, section_index, section_var = parse_projection_string(
#             self.projection_string)
#         self.plotting_X_var = first_var.upper()
#         self.plotting_Y_var = second_var.upper()
#         self.plotting_section_var = section_var.upper()
#         self.side_plot = side_plot
#         self.resolution = resolution
#         self.current_limits = {'X': [], 'Y': [], 'Z': []}
#         self.var_limits = {'X': [], 'Y': [], 'Z': []}
#         self.active_data_axis_label = None
#
#         self.max_iteration = self.df['iteration'].max()
#         self.min_iteration = self.df['iteration'].min()
#         if self.min_iteration == 0:
#             self.min_iteration = 1
#         self.current_iteration = self.min_iteration
#         self.current_bin = 0
#         self.current_iteration_data = None
#         self.current_data = None
#         self.current_thermal_core_iteration = None
#         self.current_real_thermal_core = None
#         self.current_wind_spline = None
#         self.current_thermal_core_spline = None
#         self.current_z_array = None
#         self.current_bins_iteration = None
#         self.current_bins_data = None
#         self.histogram_data = None
#
#         self.animation = None
#         self.next_iteration_data = None
#         self.next_bins_iteration = None
#         self.run_time = round(time.time())
#
#         self.cols_to_bin = ['Z_bird_TC', 'phi_bird_TC', 'rho_bird_TC']
#         self.bin_index_cols = [f'bin_index_{var}' for var in self.cols_to_bin]
#         self.plotting_args = {'scatter_1': {'alpha': 0.2,
#                                             'c': 'C0',
#                                             's': 3,
#                                             },
#                               'scatter_2': {'alpha': 0.2,
#                                             'c': 'C1',
#                                             's': 3,
#                                             },
#                               'moving_average_1': {'c': 'C2',
#                                                    },
#                               'moving_average_2': {'c': 'C3',
#                                                    }
#                               }
#         self.average_statistic = average_statistic
#
#     @classmethod
#     def from_path_decomposition(cls, path_to_decomposition, input_folder=None, **kwargs):
#
#         synthetic_data_dict, decomposition_dict = load_synthetic_and_decomposed(path_to_decomposition,
#                                                                                 input_folder=input_folder
#                                                                                 )
#
#         if 'air_velocity_field' in synthetic_data_dict:
#             air_velocity_field = synthetic_data_dict['air_velocity_field']
#         else:
#             air_velocity_field = synthetic_data_dict['air_parameters']
#
#         return ComparativeDashboard2(df=decomposition_dict['iterations'],  #df_bins=decomposition_dict['bins'],
#                                     df_splines=decomposition_dict['splines'],
#                                     df_ground_truth=synthetic_data_dict['data_real'],
#                                     **kwargs)
#
#     def initial_setup(self):
#         self.air_velocity_field_preprocessing()
#         self.data_preparation()
#         self.set_iteration_data()
#
#         self.set_sliced_data()
#         self.set_gui()
#         self.set_constant_artists()
#         self.plots_per_iteration()
#         self.go_plot()
#
#     def reset(self, event):
#         self.current_iteration = self.min_iteration
#         self.widgets['iteration_text'].set_text(f'{self.current_iteration}')
#         self.set_iteration_data()
#         self.set_sliced_data()
#         self.go_plot()
#
#     def data_preparation(self):
#         self.decomposed_data_preparation()
#         self.ground_truth_data_preparation()
#
#         self.df = pd.merge(self.df, self.df_ground_truth.drop(columns=['X', 'Y', 'Z']), how='left', on=['bird_name', 'time'])
#
#         self.var_limits['X'] = [self.df[self.df['iteration'] != 0]['X_bird_TC'].min(),
#                                 self.df[self.df['iteration'] != 0]['X_bird_TC'].max()]
#         self.var_limits['Y'] = [self.df[self.df['iteration'] != 0]['Y_bird_TC'].min(),
#                                 self.df[self.df['iteration'] != 0]['Y_bird_TC'].max()]
#         self.var_limits['Z'] = [self.df[self.df['iteration'] != 0]['Z_bird_TC'].min(),
#                                 self.df[self.df['iteration'] != 0]['Z_bird_TC'].max()]
#
#         self.current_limits['X'] = self.var_limits['X']
#         self.current_limits['Y'] = self.var_limits['Y']
#         self.current_limits['Z'] = self.var_limits['Z']
#
#     def decomposed_data_preparation(self):
#         step = 3
#         self.df = self.df[['bird_name', 'time', 'iteration',
#                            'X', 'Y', 'Z',
#                            'X_bird_TC', 'Y_bird_TC', 'Z_bird_TC',
#                            'rho_bird_TC', 'phi_bird_TC',
#                            'wind_X', 'wind_Y']
#                           + [f'dXdT_air_{step}',
#                              f'dYdT_air_{step}',
#                              f'dZdT_air_{step}',
#                              f'bank_angle',
#                              f'curvature']
#                           + self.bin_index_cols
#                           ]
#
#         self.df.rename(columns={f'dXdT_air_{step}': 'dXdT_air',
#                                 f'dYdT_air_{step}': 'dYdT_air',
#                                 f'dZdT_air_{step}': 'dZdT_air',
#                                 #f'curvature_{step}': 'curvature'
#                                 }, inplace=True)
#         self.list_of_iterations = self.df['iteration'].unique()
#         self.list_of_iterations = list(filter(lambda elem: elem != 0, self.list_of_iterations))
#         self.current_iteration = self.list_of_iterations[0]
#         self.max_iteration = max(self.list_of_iterations)
#
#         self.df['V_horizontal'] = np.linalg.norm(self.df[[f'dXdT_air', f'dYdT_air']], axis=1)
#
#     def ground_truth_data_preparation(self):
#         self.same_frame_of_reference = True
#         if not  self.same_frame_of_reference:
#             self.df_ground_truth = self.df_ground_truth[['bird_name', 'time',
#                                                          'X', 'Y', 'Z',
#                                                          'X_bird_TC_real',
#                                                          'Y_bird_TC_real',
#                                                          'rho_thermal_real',
#                                                          'phi_thermal_real',
#                                                          'dXdT_air_real', 'dYdT_air_real', 'dZdT_air_real',
#                                                          'dXdT_air_wind_real', 'dYdT_air_wind_real', 'dZdT_air_wind_real',
#                                                          'dXdT_air_rotation_real', 'dYdT_air_rotation_real',
#                                                          'dZdT_air_rotation_real',
#                                                          'dXdT_air_thermal_real', 'dYdT_air_thermal_real',
#                                                          'dZdT_air_thermal_real', 'V_H_air_rotation_real',
#                                                          'thermal_core_X_real', 'thermal_core_Y_real'
#                                                          ]]
#
#         else:
#             self.df_ground_truth = self.df.loc[self.df['iteration'] == self.current_iteration, ['bird_name', 'time', 'X', 'Y', 'Z']]
#
#             self.df_ground_truth[['thermal_core_X_real',
#                                   'thermal_core_Y_real']] = self.df_ground_truth[['Z', 'time']].apply(
#                 lambda row: self.air_velocity_field.get_thermal_core(row['Z'], row['time']),
#                 axis=1,
#                 result_type='expand')
#             v, components = self.air_velocity_field.get_velocity(self.df_ground_truth[['X', 'Y', 'Z']], t=0,
#                                                                  return_components=True)
#
#             for i_coord, coord in enumerate(['X', 'Y', 'Z']):
#                 self.df_ground_truth[f'd{coord}dT_air_real'] = v[:, i_coord]
#                 for comp, velocity in components.items():
#                     self.df_ground_truth[f'd{coord}dT_air_{comp}_real'] = velocity[:, i_coord]
#
#             self.df_ground_truth['X_bird_TC_real'] = self.df_ground_truth['X'] \
#                                                      - self.df_ground_truth['thermal_core_X_real']
#             self.df_ground_truth['Y_bird_TC_real'] = self.df_ground_truth['Y'] \
#                                                      - self.df_ground_truth['thermal_core_Y_real']
#             self.df_ground_truth['rho_thermal_real'] = np.linalg.norm(self.df_ground_truth[['X_bird_TC_real',
#                                                                                             'Y_bird_TC_real']], axis=1)
#             self.df_ground_truth['phi_thermal_real'] = np.arctan2(self.df_ground_truth['Y_bird_TC_real'],
#                                                                   self.df_ground_truth['X_bird_TC_real'])
#
#             for velocity_type in ['air_rotation', 'air_thermal']:
#                 self.df_ground_truth[[f'V_rho_rotating_{velocity_type}_real',
#                                       f'V_phi_rotating_{velocity_type}_real']] = self.df_ground_truth[['X_bird_TC_real',
#                                                                                                        'Y_bird_TC_real',
#                                                                                      f'dXdT_{velocity_type}_real',
#                                                                                      f'dYdT_{velocity_type}_real'
#                                                                                      ]].apply(
#                     lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
#                     axis=1, result_type='expand')
#                 self.df_ground_truth[f'V_H_{velocity_type}_real'] = np.linalg.norm(self.df_ground_truth[[f'dXdT_{velocity_type}_real',
#                                                                      f'dYdT_{velocity_type}_real']], axis=1)
#
#             for velocity_type in ['air', 'air_wind']:
#
#                 self.df_ground_truth[[f'V_rho_{velocity_type}_real',
#                     f'V_phi_{velocity_type}_real']] = self.df_ground_truth[['X', 'Y',
#                                                           f'dXdT_{velocity_type}_real',
#                                                           f'dYdT_{velocity_type}_real'
#                                                           ]].apply(lambda row: get_cartesian_velocity_on_rotating_frame_from_inertial_frame(*row),
#                                                                    axis=1, result_type='expand')
#                 self.df_ground_truth[f'V_H_{velocity_type}_real'] = np.linalg.norm(self.df_ground_truth[[f'dXdT_{velocity_type}_real',
#                                                                      f'dYdT_{velocity_type}_real']], axis=1)
#
#
#     def air_velocity_field_preprocessing(self):
#         pd.set_option('mode.chained_assignment', None)
#         self.real_thermal_core = {'Z': np.arange(self.df['Z'].min() - ComparativeDashboard.config['Z_padding'],
#                                                  self.df['Z'].max() + ComparativeDashboard.config['Z_padding'], 10)}
#         real_core = self.air_velocity_field.get_thermal_core(self.real_thermal_core['Z'])
#
#         self.real_thermal_core['X'] = real_core[:, 0]
#         self.real_thermal_core['Y'] = real_core[:, 1]
#         real_core = np.hstack([real_core, self.real_thermal_core['Z'].reshape(-1, 1)])
#
#         self.real_wind = {'Z': real_core[:, -1]}
#         real_wind = self.air_velocity_field.get_velocity(real_core, relative_to_ground=True, include='wind',
#                                                          return_components=False)
#
#         self.real_wind['X'] = real_wind[:, 0]
#         self.real_wind['Y'] = real_wind[:, 1]
#
#     def set_gui(self):
#         plt.interactive(self.interactive)
#         self.set_plotting_gui()
#         if self.interactive:
#             self.set_widget_gui()
#
#     def set_plotting_gui(self):
#         self.set_plotting_figure()
#         self.set_plotting_axes()
#
#     def set_widget_gui(self):
#         self.set_widget_figure()
#         self.set_widget_axes()
#         self.set_widgets()
#
#     def set_plotting_figure(self):
#         self.fig_plots = plt.figure(figsize=(19, 12),
#                                     tight_layout=True
#                                     )
#         if isinstance(self.fig_plots.canvas.manager, FigureManagerTk):
#             self.fig_plots.canvas.manager.window.attributes('-zoomed', 1)
#         else:
#             self.fig_plots.canvas.manager.window.showMaximized()
#
#         self.gridspec_wrapper = self.fig_plots.add_gridspec(1, 2, width_ratios=[1, 3])
#
#         # ===============================        LEFT PANEL    ================================
#         self.gridspec_left_panel = GridSpecFromSubplotSpec(3, 3, hspace=0.4, height_ratios=[1, 1, 1],
#                                                            width_ratios=[3, 1, 1], wspace=0,
#                                                            subplot_spec=self.gridspec_wrapper[0])
#
#         # ===============================     RIGHT PANEL =====================================
#
#         self.gridspec_right_panel = GridSpecFromSubplotSpec(3, 3, hspace=0.4, wspace=0.4,
#                                                             subplot_spec=self.gridspec_wrapper[1])
#
#     def set_plotting_axes(self):
#
#         # =====================    RIGHT PANEL       ===============================
#         self.axes['Contour'] = []
#         for i, j in product(range(2), range(3)):
#             if not len(self.axes['Contour']):
#                 current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[i, j])
#             else:
#                 current_axis = self.fig_plots.add_subplot(self.gridspec_right_panel[i, j],
#                                                           sharex=self.axes['Contour'][-1],
#                                                           sharey=self.axes['Contour'][-1])
#             self.axes['Contour'].append(current_axis)
#         self.axes['histogram'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, 0])
#         self.axes['radius'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, 1])
#         self.axes['rotation'] = self.fig_plots.add_subplot(self.gridspec_right_panel[2, 2])
#         # ======================    COLORBARS    =================================
#         self.colorbar_axes = {'Contour': []}
#         for contour_axis in self.axes['Contour']:
#             divider = make_axes_locatable(contour_axis)
#             cax = divider.append_axes('right', size='5%', pad=0.2)
#             self.colorbar_axes['Contour'].append(cax)
#
#         divider = make_axes_locatable(self.axes['histogram'])
#         self.colorbar_axes['histogram'] = divider.append_axes('right', size='5%', pad=0.2)
#
#         # =====================    LEFT PANEL       ===============================
#         if self.plot_3d:
#             self.axes['3D'] = self.fig_plots.add_subplot(self.gridspec_left_panel[0, :], projection='3d')
#         self.axes['thermal_core'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, :-2])
#         self.axes['thermal_core_diff_X'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, -2],
#                                                                       sharey=self.axes['thermal_core'])
#         self.axes['thermal_core_diff_Y'] = self.fig_plots.add_subplot(self.gridspec_left_panel[1, -1],
#                                                                       sharey=self.axes['thermal_core_diff_X'])
#         self.axes['wind'] = self.fig_plots.add_subplot(self.gridspec_left_panel[2, :-1])
#         self.axes['wind_diff'] = self.fig_plots.add_subplot(self.gridspec_left_panel[2, -1], sharey=self.axes['wind'])
#
#     def set_widget_figure(self):
#         if self.plot_3d:
#             self.fig_buttons = plt.figure(figsize=(3.3, 2),
#                                           # tight_layout=True
#                                           )
#         else:
#             self.fig_buttons = self.fig_plots
#
#         # ===============================        BUTTONS       ================================
#
#         if self.plot_3d:
#             self.gridspec_buttons = self.fig_buttons.add_gridspec(7, 6, hspace=0, wspace=0, width_ratios=[1, 1, 1, 1, 1, 3],
#                                                                   )
#         else:
#             self.gridspec_buttons = GridSpecFromSubplotSpec(7, 6, hspace=0, wspace=0, width_ratios=[1, 1, 1, 1, 1, 3],
#                                                             subplot_spec=self.gridspec_left_panel[0, 0]
#                                                             )
#
#     def set_widget_axes(self):
#
#         # =====================    BUTTONS     ===============================
#
#         self.axes['radio_projection'] = self.fig_buttons.add_subplot(self.gridspec_buttons[0:3, 0:2])
#         self.axes['radio_plot'] = self.fig_buttons.add_subplot(self.gridspec_buttons[0:3, 2:5])
#         self.axes['button_sweep_iterations'] = self.fig_buttons.add_subplot(self.gridspec_buttons[0, 5])
#         self.axes['button_sweep_var'] = self.fig_buttons.add_subplot(self.gridspec_buttons[1, 5])
#         self.axes['button_reset'] = self.fig_buttons.add_subplot(self.gridspec_buttons[2, 5])
#         self.axes['button_screenshot'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 5])
#
#         self.axes['button_iteration-5'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 0])
#         self.axes['button_iteration-1'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 1])
#         self.axes['iteration_text'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 2])
#         self.axes['button_iteration+1'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 3])
#         self.axes['button_iteration+5'] = self.fig_buttons.add_subplot(self.gridspec_buttons[3, 4])
#
#         for i, coord in enumerate(['X', 'Y', 'Z']):
#             self.axes[f'slider_{coord}'] = self.fig_buttons.add_subplot(self.gridspec_buttons[4 + i, 0:5])
#
#         # STYLE
#         self.axes['iteration_text'].axis('off')
#
#     def set_widgets(self):
#         # ==================================         WIDGET DECLARATION     ===================================#
#         self.widgets['radio_projection'] = RadioButtons(self.axes['radio_projection'], ['XY', 'XZ', 'YZ'],
#                                                         active=['XY', 'XZ', 'YZ'].index(self.projection_string))
#
#         self.widgets['radio_plot'] = RadioButtons(self.axes['radio_plot'], ['Quiver', 'Contour', 'Stream'],
#                                                   active=['Quiver', 'Contour', 'Stream'].index(self.side_plot))
#
#         self.widgets['button_sweep_iterations'] = Button(self.axes['button_sweep_iterations'], label='Sweep Iterations')
#         self.widgets['button_sweep_var'] = Button(self.axes['button_sweep_var'], label='Sweep Var')
#         self.widgets['button_reset'] = Button(self.axes['button_reset'], label='Reset')
#         self.widgets['button_screenshot'] = Button(self.axes['button_screenshot'], label='Scr.Shot')
#
#         self.widgets['button_iteration-5'] = Button(self.axes['button_iteration-5'], label='-5')
#         self.widgets['button_iteration-1'] = Button(self.axes['button_iteration-1'], label='-1')
#         self.widgets['iteration_text'] = self.axes['iteration_text'].text(0.45, 0.45, f'{self.current_iteration}')
#         self.widgets['button_iteration+1'] = Button(self.axes['button_iteration+1'], label='+1')
#         self.widgets['button_iteration+5'] = Button(self.axes['button_iteration+5'], label='+5')
#
#         # The +- 0.01 is just to make sure the valinit is inside the upper and lower bounds
#         self.widgets['slider_X'] = RangeSlider(ax=self.axes['slider_X'], label='X',
#                                                valmin=self.var_limits['X'][0],
#                                                valmax=self.var_limits['X'][1])
#
#         self.widgets['slider_Y'] = RangeSlider(ax=self.axes['slider_Y'], label='Y',
#                                                valmin=self.var_limits['Y'][0],
#                                                valmax=self.var_limits['Y'][1])
#
#         self.widgets['slider_Z'] = RangeSlider(ax=self.axes['slider_Z'], label='Z',
#                                                valmin=self.var_limits['Z'][0],
#                                                valmax=self.var_limits['Z'][1])
#
#         # ==================================         WIDGET CONFIGURATION     ===================================#
#         self.widgets['slider_X'].set_min(self.current_limits['X'][0])
#         self.widgets['slider_X'].set_max(self.current_limits['X'][1])
#         self.widgets['slider_Y'].set_min(self.current_limits['Y'][0])
#         self.widgets['slider_Y'].set_max(self.current_limits['Y'][1])
#         self.widgets['slider_Z'].set_min(self.current_limits['Z'][0])
#         self.widgets['slider_Z'].set_max(self.current_limits['Z'][1])
#
#         # self.widgets['button_reset'] = Button(self.axes['button_reset'], 'Reset')
#
#         # ===================================    CALLBACKS      ====================================================== #
#         self.widgets['slider_X'].on_changed(lambda event: self.on_slider_change(event, 'X'))
#         self.widgets['slider_Y'].on_changed(lambda event: self.on_slider_change(event, 'Y'))
#         self.widgets['slider_Z'].on_changed(lambda event: self.on_slider_change(event, 'Z'))
#         self.widgets['radio_projection'].on_clicked(self.on_radio_projection_clicked)
#         self.widgets['radio_plot'].on_clicked(self.on_radio_plot_clicked)
#
#         self.widgets['slider_X'].on_changed(lambda event: self.set_sliced_data())
#         self.widgets['slider_Y'].on_changed(lambda event: self.set_sliced_data())
#         self.widgets['slider_Z'].on_changed(lambda event: self.set_sliced_data())
#         self.widgets['radio_projection'].on_clicked(lambda event: self.set_sliced_data())
#         self.widgets['radio_plot'].on_clicked(lambda event: self.set_sliced_data())
#
#         self.widgets['slider_X'].on_changed(lambda event: self.go_plot())
#         self.widgets['slider_Y'].on_changed(lambda event: self.go_plot())
#         self.widgets['slider_Z'].on_changed(lambda event: self.go_plot())
#         self.widgets['radio_projection'].on_clicked(lambda event: self.go_plot())
#         self.widgets['radio_plot'].on_clicked(lambda event: self.go_plot())
#
#         self.widgets['button_iteration-5'].on_clicked(lambda event: self.on_iteration_change(event, -5))
#         self.widgets['button_iteration-1'].on_clicked(lambda event: self.on_iteration_change(event, -1))
#         self.widgets['button_iteration+1'].on_clicked(lambda event: self.on_iteration_change(event, +1))
#         self.widgets['button_iteration+5'].on_clicked(lambda event: self.on_iteration_change(event, +5))
#
#         self.widgets['button_iteration-5'].on_clicked(lambda event: self.set_iteration_data())
#         self.widgets['button_iteration-1'].on_clicked(lambda event: self.set_iteration_data())
#         self.widgets['button_iteration+1'].on_clicked(lambda event: self.set_iteration_data())
#         self.widgets['button_iteration+5'].on_clicked(lambda event: self.set_iteration_data())
#
#         self.widgets['button_iteration-5'].on_clicked(lambda event: self.plots_per_iteration())
#         self.widgets['button_iteration-1'].on_clicked(lambda event: self.plots_per_iteration())
#         self.widgets['button_iteration+1'].on_clicked(lambda event: self.plots_per_iteration())
#         self.widgets['button_iteration+5'].on_clicked(lambda event: self.plots_per_iteration())
#
#         self.widgets['button_iteration-5'].on_clicked(lambda event: self.set_sliced_data())
#         self.widgets['button_iteration-1'].on_clicked(lambda event: self.set_sliced_data())
#         self.widgets['button_iteration+1'].on_clicked(lambda event: self.set_sliced_data())
#         self.widgets['button_iteration+5'].on_clicked(lambda event: self.set_sliced_data())
#
#         self.widgets['button_iteration-5'].on_clicked(lambda event: self.go_plot())
#         self.widgets['button_iteration-1'].on_clicked(lambda event: self.go_plot())
#         self.widgets['button_iteration+1'].on_clicked(lambda event: self.go_plot())
#         self.widgets['button_iteration+5'].on_clicked(lambda event: self.go_plot())
#
#         self.widgets['button_sweep_var'].on_clicked(lambda event: self.set_bin())
#         self.widgets['button_sweep_var'].on_clicked(lambda event: self.go_plot())
#
#         self.widgets['button_sweep_iterations'].on_clicked(self.sweep_iterations)
#         self.widgets['button_sweep_var'].on_clicked(self.sweep_var)
#         self.widgets['button_screenshot'].on_clicked(lambda event: self.save_figure())
#
#         circle = Circle((0.8, 0.8), 0.05, facecolor='r', linewidth=3, alpha=1, visible=False)
#         self.circle = self.axes['radio_plot'].add_patch(circle)
#
#     def set_constant_artists(self):
#         colors = ComparativeDashboard.plot_styles_dict['colors']
#         for col in ['X', 'Y']:
#             self.artists[f'real_wind_{col}'] = Line2D(self.real_wind[col],
#                                                       self.real_wind['Z'],
#                                                       color=colors['real'][col][0],
#                                                       linestyle=colors['real'][col][1:],
#                                                       label=f'Real {col}')
#             self.artists[f'real_thermal_core_{col}'] = Line2D(self.real_thermal_core[col],
#                                                               self.real_thermal_core['Z'],
#                                                               color=colors['real'][col][0],
#                                                               linestyle=colors['real'][col][1:],
#                                                               label=f'Real {col}')
#
#     def plot_constant_artists(self):
#         for col in ['X', 'Y']:
#             self.axes['wind'].add_artist(copy(self.artists[f'real_wind_{col}']))
#             self.axes['thermal_core'].add_artist(copy(self.artists[f'real_thermal_core_{col}']))
#
#     @staticmethod
#     def restore_legend(ax, artist_types=None):
#         if artist_types is None:
#             artist_types = ['lines']
#         list_of_artists = []
#         for at in artist_types:
#             list_of_artists += [a for a in getattr(ax, at)]
#
#         list_of_labels = [a.get_label() for a in list_of_artists]
#         ax.legend(list_of_artists, list_of_labels,)
#         ax.relim()
#         # update ax.viewLim using the new dataLim
#         ax.autoscale_view()
#
#     def on_iteration_change(self, event, increment):
#         if self.current_iteration + increment > self.max_iteration:
#             self.current_iteration = self.current_iteration + increment - self.max_iteration
#         elif self.current_iteration + increment == 0:
#             self.current_iteration = self.max_iteration
#         elif self.current_iteration + increment < 0:
#             self.current_iteration = self.max_iteration + self.current_iteration + increment
#         else:
#             self.current_iteration += increment
#
#         self.widgets['iteration_text'].set_text(f'{self.current_iteration}')
#
#     def on_slider_change(self, event, coordinate):
#
#         self.current_limits[coordinate] = [event[0], event[1]]
#
#     def on_radio_projection_clicked(self, event):
#
#         self.projection_string = event
#         first_var, second_var, first_index, second_index, section_index, section_var = parse_projection_string(
#             self.projection_string)
#         self.plotting_X_var = first_var.upper()
#         self.plotting_Y_var = second_var.upper()
#         self.plotting_section_var = section_var.upper()
#
#         print('plotting_section_var', self.plotting_section_var)
#
#     def on_radio_plot_clicked(self, event):
#         self.side_plot = event
#
#     def set_iteration_data(self):
#
#         if self.next_iteration_data is not None:
#             self.current_iteration_data = self.next_iteration_data
#             self.current_bins_iteration = self.next_bins_iteration
#         else:
#             self.current_iteration_data = self.df[self.df['iteration'] == self.current_iteration]
#             # self.bin()
#             #self.current_bins_iteration = self.df_bins[self.df_bins['iteration'] == self.current_iteration]
#
#             current_spline = self.df_splines[self.df_splines['iteration'] == self.current_iteration
#                                              ].to_dict(orient='records')[0]
#             current_thermal_core_spline_parameters = current_spline['thermal_core']
#             current_wind_spline_parameters = current_spline['wind']
#
#             self.current_z_array = np.arange(np.min(current_thermal_core_spline_parameters['thermal_core_Y']['tck'][0]),
#                                              np.max(current_thermal_core_spline_parameters['thermal_core_Y']['tck'][0]),
#                                              10)
#             self.current_thermal_core_spline = {}
#             self.current_wind_spline = {}
#
#             for col in ['X', 'Y']:
#                 self.current_thermal_core_spline[col] = UnivariateSplineWrapper.from_tck(
#                     current_thermal_core_spline_parameters[f'thermal_core_{col}']['tck'])
#                 self.current_wind_spline[col] = UnivariateSplineWrapper.from_tck(
#                     current_wind_spline_parameters[f'wind_{col}']['tck'])
#
#     def set_bin(self):
#         self.current_bin += 1
#
#         print(f'bin {self.current_bin}')
#         self.fig_plots.suptitle(f'Bin Z: {self.current_bin}')
#         self.current_data = self.current_iteration_data.copy()
#         self.current_bins_data = self.current_bins_iteration.copy()
#
#         self.current_data = self.current_data[self.current_data['bin_index_Z_bird_TC'] == self.current_bin]
#
#         self.current_limits['Z'] = self.current_bins_data[['Z_bird_TC_min', 'Z_bird_TC_max']].values[0]
#         # self.remove_outliars()
#
#     def set_sliced_data(self):
#
#         self.current_data = self.current_iteration_data.copy()
#         for coord, lims in self.current_limits.items():
#             self.current_data = self.current_data[self.current_data[f'{coord}_bird_TC'].between(lims[0], lims[1])]
#
#         self.remove_outliers()
#         self.process_sliced_data()
#         self.process_sliced_ground_truth_data()
#
#     def process_sliced_data(self):
#         rho_resolution = 1
#         rho_max = self.current_data['rho_bird_TC'].max()
#         rho_min = self.current_data['rho_bird_TC'].min()
#
#         df_roll = self.current_data[['rho_bird_TC', 'V_horizontal', f'dZdT_air']]
#
#         df_roll = df_roll.sort_values(by='rho_bird_TC')
#         window_size = df_roll['rho_bird_TC'].count() / (rho_max - rho_min) * rho_resolution
#         window_size = round(window_size)
#
#         self.current_data_rolling_in_rho = df_roll.rolling(window=window_size,
#                                                            center=True).agg({'rho_bird_TC': 'mean',
#                                                                                     f'dZdT_air': ('mean', 'std'),
#                                                                                     'V_horizontal': ('mean', 'std')}
#                                                                             ).dropna()
#
#         self.current_data_rolling_in_rho['rho_bird_TC_avg'] = self.current_data_rolling_in_rho['rho_bird_TC']['mean']
#         self.current_data_rolling_in_rho['vz_avg'] = self.current_data_rolling_in_rho[f'dZdT_air']['mean']
#         self.current_data_rolling_in_rho['vz_std'] = self.current_data_rolling_in_rho[f'dZdT_air']['std']
#         self.current_data_rolling_in_rho['vh_avg'] = self.current_data_rolling_in_rho['V_horizontal']['mean']
#         self.current_data_rolling_in_rho['vh_std'] = self.current_data_rolling_in_rho['V_horizontal']['std']
#         self.current_data_rolling_in_rho.drop(columns=['rho_bird_TC', f'dZdT_air', 'V_horizontal'], inplace=True)
#         self.current_data_rolling_in_rho.columns = self.current_data_rolling_in_rho.columns.droplevel(1)
#
#         window_scale = {'phi': 1 / (2 * np.pi) * rho_resolution,
#                         'rho': 1 / (rho_max - rho_min) * rho_resolution
#                         }
#
#         self.current_data_rolling = {}
#         for col in ['phi', 'rho']:
#             df_roll = self.current_data[[f'{col}_bird_TC', 'V_horizontal', f'dZdT_air']]
#             window_size = round(df_roll[f'{col}_bird_TC'].count() * window_scale[col])
#             df_roll = df_roll.sort_values(by=f'{col}_bird_TC')
#             self.current_data_rolling[col] = df_roll.rolling(window=window_size).agg(
#                 {f'{col}_bird_TC': 'mean',
#                  f'dZdT_air': ('mean', 'std'),
#                  'V_horizontal': ('mean', 'std')}).dropna()
#
#             self.current_data_rolling[col][f'{col}_bird_TC_avg'] = self.current_data_rolling[col][f'{col}_bird_TC']['mean']
#             self.current_data_rolling[col]['vz_avg'] = self.current_data_rolling[col][f'dZdT_air']['mean']
#             self.current_data_rolling[col]['vz_std'] = self.current_data_rolling[col][f'dZdT_air']['std']
#             self.current_data_rolling[col]['vh_avg'] = self.current_data_rolling[col]['V_horizontal']['mean']
#             self.current_data_rolling[col]['vh_std'] = self.current_data_rolling[col]['V_horizontal']['std']
#             self.current_data_rolling[col].drop(columns=[f'{col}_bird_TC', f'dZdT_air', 'V_horizontal'], inplace=True)
#             self.current_data_rolling[col].columns = self.current_data_rolling[col].columns.droplevel(1)
#
#     def process_sliced_ground_truth_data(self):
#         epsilon = 0.01
#         bin_edges = {'rho': np.linspace(0, self.current_data['rho_bird_TC'].max() * (1 + epsilon), 20),
#                      'phi': np.linspace(-np.pi, np.pi * (1 + epsilon), 20)}
#
#         self.current_ground_truth_data_grouped = {}
#         for col in ['rho', 'phi']:
#             self.current_data[f'{col}_bin_indices'] = np.digitize(self.current_data[f'{col}_bird_TC_real'],
#                                                                   bin_edges[col]) - 1
#             self.current_data[f'{col}_bin'] = bin_edges[col][self.current_data[f'{col}_bin_indices'].values]
#             self.current_ground_truth_data_grouped[col] = self.current_data[[f'{col}_bin',
#                                                                              f'{col}_bin_indices',
#                                                                              'dZdT_air_thermal_real',
#                                                                              'V_H_air_rotation_real'
#                                                                              ]].groupby([f'{col}_bin',
#                                                                                          f'{col}_bin_indices']).agg(
#                 vz_real_avg=('dZdT_air_thermal_real', 'mean'),
#                 vz_real_std=('dZdT_air_thermal_real', 'std'),
#                 vh_real_avg=('V_H_air_rotation_real', 'mean'),
#                 vh_real_std=('V_H_air_rotation_real', 'std')
#             ).reset_index()
#
#     def remove_outliers(self):
#         self.current_data = self.current_data[np.abs(self.current_data[f'curvature']) < 0.1]
#
#     def aggregate_data(self, cols_to_bin=None, statistic='mean'):
#         if cols_to_bin is None:
#             cols_to_bin = ['rho_bird_TC', 'phi_bird_TC']
#         bin_edges_dict = {'rho_bird_TC': self.histogram_data['bins']['polar']['rho'],
#                           'phi_bird_TC': self.histogram_data['bins']['polar']['phi']}
#
#         self.current_data[f'X_bird_TC_avg'] = self.current_data['vis_rho_bird_TC_avg'] \
#                                               * np.cos(self.current_data['vis_phi_bird_TC_avg'])
#         self.current_data[f'Y_bird_TC_avg'] = self.current_data['vis_rho_bird_TC_avg'] \
#                                               * np.sin(self.current_data['vis_phi_bird_TC_avg'])
#
#         # Collapse one direction
#
#         groupby_cols = [f'{self.plotting_X_var}_bird_TC_avg',
#                         f'{self.plotting_Y_var}_bird_TC_avg'] + list(bin_edges_dict.keys())
#
#         df_grouped = self.current_data.groupby(by=groupby_cols,
#                                                as_index=False)
#
#         # Thermal Real Profile per bin
#
#         df_group = df_grouped.agg({'dXdT_air_rotation_real': statistic,
#                                    'dYdT_air_rotation_real': statistic,
#                                    'dZdT_air_rotation_real': statistic,
#                                    'dXdT_air_thermal_real': statistic,
#                                    'dYdT_air_thermal_real': statistic,
#                                    'dZdT_air_thermal_real': statistic,
#                                    f'dXdT_air': statistic,
#                                    f'dYdT_air': statistic,
#                                    f'dZdT_air': statistic,
#                                    f'vis_counts': statistic}
#
#                                   )
#         df_group['thermal_real_X'] = df_group['dXdT_air_rotation_real'] + df_group['dXdT_air_thermal_real']
#         df_group['thermal_real_Y'] = df_group['dYdT_air_rotation_real'] + df_group['dYdT_air_thermal_real']
#         df_group['thermal_real_Z'] = df_group['dZdT_air_rotation_real'] + df_group['dZdT_air_thermal_real']
#
#         df_group['thermal_diff_X'] = df_group['thermal_real_X'] - df_group[f'dXdT_air']
#         df_group['thermal_diff_Y'] = df_group['thermal_real_Y'] - df_group[f'dYdT_air']
#         df_group['thermal_diff_Z'] = df_group['thermal_real_Z'] - df_group[f'dZdT_air']
#         return df_group
#
#     def prepare_next_iteration(self):
#
#         self.next_iteration_data = self.df[self.df['iteration'] == self.current_iteration + 1]
#         # self.bin()
#         #self.next_bins_iteration = self.df_bins[self.df_bins['iteration'] == self.current_iteration + 1]
#
#     def plot_wind(self):
#
#         colors = ComparativeDashboard.plot_styles_dict['colors']
#
#         self.axes['wind'].clear()
#         current_thermal_core = np.empty(shape=(len(self.current_z_array), 3) )
#         current_thermal_core[:, 0] = self.current_thermal_core_spline['X'](self.current_z_array)
#         current_thermal_core[:, 1] = self.current_thermal_core_spline['Y'](self.current_z_array)
#         current_thermal_core[:, 2] = self.current_z_array
#         for col in ['X', 'Y']:
#             self.axes['wind'].plot(self.current_wind_spline[col](self.current_z_array),
#                                    self.current_z_array, colors['calculated'][col],
#                                    label=f'Calc {col}')
#
#         # dXdT_air_i_avg
#         # dYdT_air_i_avg
#         self.axes['wind'].set_title('wind')
#         self.axes['wind'].set_xlabel('$V_X$, $V_y$ (m/s)')
#         self.axes['wind'].set_ylabel('Z (m)')
#         self.axes['wind'].legend()
#
#         # WIND DIFF
#         self.axes['wind_diff'].clear()
#         wind_diff = np.array([self.real_wind[col] - self.current_wind_spline[col](self.real_wind['Z'])
#                               for col in ['X', 'Y']]).T
#         plotting_indices = np.full(len(self.real_wind['Z']), fill_value=False)
#         plotting_indices[::7] = True
#         valid_indices = np.logical_and(np.min(self.current_z_array) < self.real_wind['Z'],
#                                          self.real_wind['Z'] < np.max(self.current_z_array))
#         plotting_indices[~valid_indices] = False
#         wind_diff = wind_diff[plotting_indices.flatten()]
#         valid_z_array = self.real_wind['Z'][plotting_indices]
#
#         Q = self.axes['wind_diff'].quiver(np.zeros(shape=len(valid_z_array)),
#                                           valid_z_array,
#                                           wind_diff[:, 0],
#                                           wind_diff[:, 1],
#                                           pivot='middle',
#                                           units='height',
#                                           scale=1,
#                                           scale_units='x',
#                                           width=0.01
#                                           )
#         wind_diff_avg = np.mean(np.linalg.norm(wind_diff, axis=1))
#         wind_diff_x_max = np.nanpercentile(np.abs(wind_diff[:, 0]), 99)
#
#         if wind_diff_x_max != 0:
#             self.axes['wind_diff'].set_xlim(-wind_diff_x_max, wind_diff_x_max)
#         qk = self.axes['wind_diff'].quiverkey(Q, 0.7, -0.2, wind_diff_avg, f'${wind_diff_avg:.2g} m/s$', labelpos='E', )
#         self.axes['wind_diff'].set_axis_off()
#         self.axes['wind_diff'].set_title('Diff')
#
#     def plot_thermal_core(self):
#
#         colors = {'real':       {'X': 'r--', 'Y': 'y--'},
#                   'calculated': {'X': 'b', 'Y': 'g'}}
#
#         self.axes['thermal_core'].clear()
#         self.axes['thermal_core_diff_X'].clear()
#         self.axes['thermal_core_diff_Y'].clear()
#         current_thermal_core = np.empty(shape=(len(self.current_z_array), 2))
#         thermal_core_diff = np.empty(shape=(len(self.current_z_array), 2))
#         real_tc = self.air_velocity_field.get_thermal_core(self.current_z_array, t=0)
#         for i, col in enumerate(['X', 'Y']):
#             current_thermal_core[:, i] = self.current_thermal_core_spline[col](self.current_z_array)
#             thermal_core_diff[:, i] = real_tc[:, i] - current_thermal_core[:, i]
#             self.axes['thermal_core'].plot(current_thermal_core[:, i],
#                                            self.current_z_array, colors['calculated'][col],
#                                            label=f'Calc {col}')
#             self.axes[f'thermal_core_diff_{col}'].plot(thermal_core_diff[:, i],
#                                                        self.current_z_array, colors['calculated'][col],
#                                                        label=f'Calc {col}')
#             self.axes[f'thermal_core_diff_{col}'].axvline(x=0, ls='--', alpha=0.3)
#             self.axes[f'thermal_core_diff_{col}'].get_yaxis().set_visible(False)
#             self.axes[f'thermal_core_diff_{col}'].set_title(f'Diff {col}')
#
#         self.axes['thermal_core'].set_title('Thermal_core')
#         self.axes['thermal_core'].set_xlabel('X,Y (m)')
#         self.axes['thermal_core'].set_ylabel('Z (m)')
#         self.axes['thermal_core'].legend()
#
#     def plot_overview(self):
#         if not self.plot_3d:
#             return
#         self.axes['3D'].clear()
#         df_plot = self.current_iteration_data.iloc[::10]
#         df_plot = df_plot[['X_bird_TC', 'Y_bird_TC', 'Z_bird_TC', 'bank_angle']]
#         self.artists['3D'] = plot_scatter3D(self.axes['3D'],
#                                             X_array=df_plot['X_bird_TC'],
#                                             Y_array=df_plot['Y_bird_TC'],
#                                             Z_array=df_plot['Z_bird_TC'],
#                                             color_array=np.abs(df_plot['bank_angle']),
#                                             kwargs={'alpha': 0.2}
#                                             )
#         if '3D' in self.colorbar_axes.keys():
#             self.colorbar_axes['3D'].clear()
#             self.fig_plots.colorbar(self.artists['3D'], cax=self.colorbar_axes['3D'], label='Bank Angle (rad)')
#         else:
#             cb = self.fig_plots.colorbar(self.artists['3D'], ax=self.axes['3D'], location='left',
#                                          label='Bank Angle (rad)')
#             self.colorbar_axes['3D'] = cb.ax
#
#     def plots_per_iteration(self):
#
#         self.plot_overview()
#
#         self.plot_thermal_core()
#
#         self.plot_wind()
#         self.plot_constant_artists()
#         self.restore_legend(self.axes['wind'])
#         self.restore_legend(self.axes['thermal_core'])
#         self.fig_plots.suptitle(f'Iteration {self.current_iteration}')
#
#     def plot_shades(self):
#         list_of_axes = ['cube', 'slice_thermal_core', 'slice_wind', 'slice_wind_diff',
#                         'slice_thermal_core_diff_X', 'slice_thermal_core_diff_Y']
#         for artist in list_of_axes:
#             if artist in self.artists.keys():
#                 self.artists[artist].remove()
#         if self.plot_3d:
#             self.artists['cube'] = self.axes['3D'].bar3d(x=self.current_limits['X'][0],
#                                                          y=self.current_limits['Y'][0],
#                                                          z=self.current_limits['Z'][0],
#                                                          dx=self.current_limits['X'][1] - self.current_limits['X'][0],
#                                                          dy=self.current_limits['Y'][1] - self.current_limits['Y'][0],
#                                                          dz=self.current_limits['Z'][1] - self.current_limits['Z'][0],
#                                                          shade=False,
#                                                          color='b', alpha=0.1)
#
#         for art in list_of_axes[1:]:
#             ax_name = art.replace('slice_', '')
#             current_ax = self.axes[ax_name]
#             if ax_name == 'wind_diff':
#                 x_limits = current_ax.get_xlim()
#             else:
#                 x_limits = current_ax.dataLim.intervalx
#             self.artists[art] = current_ax.fill_between(
#                 x=x_limits,
#                 y1=self.current_limits['Z'][0],
#                 y2=self.current_limits['Z'][1],
#                 color='b', alpha=0.1)
#
#     def plots_per_slice(self):
#
#         self.plot_radius()
#         self.plot_rotation()
#         self.plot_thermal()
#         self.plot_shades()
#         self.plot_white_shadow()
#
#     def set_bins(self):
#
#         n_bins = self.resolution
#         h, rho_edges, phi_edges = np.histogram2d(self.current_data['rho_bird_TC'],
#                                                  self.current_data['phi_bird_TC'],
#                                                  bins=n_bins,
#                                                  range=[[0, self.current_data['rho_bird_TC'].max() + 0.01],
#                                                         [-np.pi, np.pi + 0.01]])
#
#         rho_edges[0] = 0
#         rho_edges[-1] += 0.1  # This ensures that the maximum value is included in the bin
#         rho_mg, phi_mg = np.meshgrid(rho_edges, phi_edges)
#
#         x_edges = rho_mg * np.cos(phi_mg)
#         y_edges = rho_mg * np.sin(phi_mg)
#
#         self.histogram_data = {'counts': h,
#                                'bins': {
#                                    'polar': {'rho': rho_edges,
#                                              'phi': phi_edges},
#                                    'cartesian': {'x': x_edges,
#                                                  'y': y_edges}
#                                }}
#
#         rho_phi_lower_edges = np.array(list(product(rho_edges[:-1], phi_edges[:-1])))
#         rho_phi_upper_edges = np.array(list(product(rho_edges[1:], phi_edges[1:])))
#         indices_list = np.array(list(product(range(n_bins), range(n_bins))))
#         rho_lower_edge = rho_phi_lower_edges[:, 0]
#         rho_upper_edge = rho_phi_upper_edges[:, 0]
#         phi_lower_edge = rho_phi_lower_edges[:, 1]
#         phi_upper_edge = rho_phi_upper_edges[:, 1]
#
#         histogram_data = {'vis_rho_index': indices_list[:, 0],
#                           'vis_phi_index': indices_list[:, 1],
#                           'vis_counts':    h.flatten(),
#                           'vis_rho_bird_TC_min':   rho_lower_edge,
#                           'vis_rho_bird_TC_max':   rho_upper_edge,
#                           'vis_phi_bird_TC_min':   phi_lower_edge,
#                           'vis_phi_bird_TC_max':   phi_upper_edge
#                           }
#
#         self.df_histogram = pd.DataFrame(histogram_data)
#         self.df_histogram['vis_rho_bird_TC_avg'] = (self.df_histogram['vis_rho_bird_TC_max'] + self.df_histogram['vis_rho_bird_TC_min']) / 2
#         self.df_histogram['vis_phi_bird_TC_avg'] = (self.df_histogram['vis_phi_bird_TC_max'] + self.df_histogram['vis_phi_bird_TC_min']) / 2
#
#         self.df_histogram['vis_x_avg'] = self.df_histogram['vis_rho_bird_TC_avg'] * np.cos(self.df_histogram['vis_phi_bird_TC_avg'])
#         self.df_histogram['vis_y_avg'] = self.df_histogram['vis_rho_bird_TC_avg'] * np.sin(self.df_histogram['vis_phi_bird_TC_avg'])
#
#         self.df_histogram['vis_x_max'] = self.df_histogram['vis_rho_bird_TC_max'] * np.cos(self.df_histogram['vis_phi_bird_TC_max'])
#         self.df_histogram['vis_y_max'] = self.df_histogram['vis_rho_bird_TC_max'] * np.sin(self.df_histogram['vis_phi_bird_TC_max'])
#
#         self.df_histogram['vis_x_min'] = self.df_histogram['vis_rho_bird_TC_min'] * np.cos(self.df_histogram['vis_phi_bird_TC_max'])
#         self.df_histogram['vis_y_min'] = self.df_histogram['vis_rho_bird_TC_min'] * np.sin(self.df_histogram['vis_phi_bird_TC_max'])
#
#         self.current_data['vis_bin_index_rho_bird_TC'] = np.digitize(self.current_data['rho_bird_TC'].values,
#                                                                      rho_edges) - 1
#         self.current_data['vis_bin_index_phi_bird_TC'] = np.digitize(self.current_data['phi_bird_TC'].values,
#                                                                phi_edges) - 1
#
#         self.current_data = pd.merge(self.current_data, self.df_histogram, how='left',
#                                      left_on=['vis_bin_index_rho_bird_TC',
#                                               'vis_bin_index_phi_bird_TC'],
#                                      right_on=['vis_rho_index',
#                                                'vis_phi_index'])
#
#     def plot_histogram(self):
#         h = self.histogram_data['counts']
#         x_edges = self.histogram_data['bins']['cartesian']['x']
#         y_edges = self.histogram_data['bins']['cartesian']['y']
#
#         N = np.ceil(np.max(h)).astype(int)  # +1 to include zero
#         cmap = mpl.colormaps['jet']
#
#         cmap_list = [(1, 1, 1, 1)]  # This is white for zero
#         cmap_list = cmap_list + [cmap(i) for i in np.linspace(1, cmap.N, N - 1).astype(int)]
#         from matplotlib.colors import LinearSegmentedColormap
#         cmap = LinearSegmentedColormap.from_list('Custom cmap', cmap_list, N)
#
#         self.artists['histogram'] = self.axes['histogram'].pcolormesh(x_edges, y_edges, h.T, cmap=cmap, snap=False, )
#
#         self.axes['histogram'].set_xlim(np.min(x_edges), np.max(x_edges))
#         self.axes['histogram'].set_ylim(np.min(y_edges), np.max(y_edges))
#
#         self.axes['histogram'].set_title('Histogram')
#         if self.projection_string == 'XY':
#             self.axes['histogram'].set_aspect('equal')
#         else:
#             self.axes['histogram'].set_aspect('auto')
#
#         self.axes['histogram'].set_xlabel(f'{self.projection_string[0]}(m)')
#         self.axes['histogram'].set_ylabel(f'{self.projection_string[1]}(m)')
#         if N > 10:
#             resolution = 10
#         else:
#             resolution = N
#         labels = np.concatenate([[0], np.linspace(0, N - 1, resolution, endpoint=True, dtype=int)])
#         self.fig_plots.colorbar(self.artists['histogram'], cax=self.colorbar_axes['histogram'],
#                                 ticks=labels
#                                 )
#
#     def plot_thermal(self):
#
#         for axis in self.axes['Contour'] + self.colorbar_axes['Contour']:
#             axis.clear()
#         self.axes['histogram'].clear()
#         self.colorbar_axes['histogram'].clear()
#
#         self.set_bins()
#         self.plot_histogram()
#         df_group = self.aggregate_data(statistic=self.average_statistic)
#         min_n = 5
#         df_group = df_group[df_group['vis_counts'] > min_n]
#         # Collapse one direction
#         var_X = self.plotting_X_var + '_bird_TC_avg'
#         var_Y = self.plotting_Y_var + '_bird_TC_avg'
#
#         df_group['thermal_real_average'] = df_group[f'thermal_real_{self.plotting_section_var}']
#
#         df_group['V_calculated_average'] = df_group[f'd{self.plotting_section_var}dT_air']
#
#         df_group['thermal_diff_average'] = df_group['thermal_real_average'] - df_group['V_calculated_average']
#
#         zi_dict = {'air_average': 'V_calculated_average',
#                    'real_average': 'thermal_real_average',
#                    'diff_average': 'thermal_diff_average',
#                    'air_section': [f'd{self.plotting_X_var}dT_air',
#                                    f'd{self.plotting_Y_var}dT_air'],
#                    'real_section': [f'thermal_real_{self.plotting_X_var}',
#                                     f'thermal_real_{self.plotting_Y_var}'],
#                    'diff_section': [f'thermal_diff_{self.plotting_X_var}',
#                                     f'thermal_diff_{self.plotting_Y_var}']}
#
#         titles_dict = {'air_average': f'$V_{self.plotting_section_var}$ - Calculated',
#                        'real_average': f'$V_{self.plotting_section_var}$ - Real Data',
#                        'diff_average': f'$V_{self.plotting_section_var}$ - Residuals',
#                        'air_section': '$V_{Section}$ - Calculated',
#                        'real_section': '$V_{Section}$ - Real Data',
#                        'diff_section': '$V_{Section}$ - Residuals'}
#
#         vmin_section = np.min([np.nanpercentile(np.linalg.norm(df_group.loc[df_group['vis_counts'] > min_n, col], axis=1), 1)
#                               for col in [zi_dict['air_section'], zi_dict['real_section']]
#                                ])
#         vmax_section = np.max([np.nanpercentile(np.linalg.norm(df_group.loc[df_group['vis_counts'] > min_n, col], axis=1), 99)
#                               for col in [zi_dict['air_section'], zi_dict['real_section']]
#                                ])
#
#         vmin_average = np.min([np.nanpercentile(df_group.loc[df_group['vis_counts'] > min_n, col], 1)
#                               for col in [zi_dict['air_average'], zi_dict['real_average']]
#                                ])
#         vmax_average = np.max([np.nanpercentile(df_group.loc[df_group['vis_counts'] > min_n, col], 99)
#                               for col in [zi_dict['air_average'], zi_dict['real_average']]
#                                ])
#         contour_kwargs = {'air_average': {'norm': Normalize(vmin=vmin_average,
#                                                             vmax=vmax_average,
#                                                             clip=True),
#                                           'cmap': None},
#                           'real_average': {'norm': Normalize(vmin=vmin_average,
#                                                              vmax=vmax_average,
#                                                              clip=True)},
#                           'diff_average': {'norm': TwoSlopeNorm(vcenter=0),
#                                            'cmap': 'bwr'},
#                           'air_section': {'norm': Normalize(vmin=vmin_section,
#                                                             vmax=vmax_section,
#                                                             clip=True),
#                                           'cmap': 'turbo'},
#                           'real_section': {'norm': Normalize(vmin=vmin_section,
#                                                              vmax=vmax_section,
#                                                              clip=True),
#                                            'cmap': 'turbo'},
#                           'diff_section': {'norm': Normalize(),
#                                            'cmap': 'turbo'}
#                           }
#         plot_specific_kwargs = {'Quiver': {'pivot': 'mid',
#                                            'angles': 'uv',
#                                            'scale_units': 'width',
#                                            'scale': self.resolution,
#                                            'units': 'width',
#                                            'width': 50 / (df_group[var_X].max() - df_group[var_X].min()),
#                                            'constant_length': 1
#                                            },
#                                 'Contour': {'levels': self.resolution,
#                                             'extend': 'both'}}
#         self.artists['Contour'] = []
#         self.plot_contour_background = True
#
#         for i, (plot_type, plotting_var) in enumerate(zi_dict.items()):
#             if i <= 2:  # Cross-Section
#                 df_plot = df_group[np.logical_not(df_group[plotting_var].isna())]
#
#                 artist, _ = plot_interpolated(self.axes['Contour'][i], 'contour',
#                                               x_array=df_plot[var_X].values,
#                                               y_array=df_plot[var_Y].values,
#                                               vx_array=df_plot[plotting_var].values,
#                                               background_contour_kwargs=False,
#                                               resolution=self.resolution,
#                                               **{'kwargs': contour_kwargs[plot_type]})
#                 # artist = plot_contour_tri(ax=self.axes['Contour'][i],
#                 #                           x_array=df_plot[var_X].values,
#                 #                           y_array=df_plot[var_Y].values,
#                 #                           color_array=df_plot[plotting_var].values,
#                 #                           resolution=self.resolution,
#                 #                           kwargs=contour_kwargs[plot_type])
#             else:  # Side View
#                 current_kwargs = copy(contour_kwargs[plot_type])
#                 if self.side_plot in plot_specific_kwargs:
#                     current_kwargs.update(plot_specific_kwargs[self.side_plot])
#                 field_X = plotting_var[0]
#                 field_Y = plotting_var[1]
#                 df_plot = df_group[np.logical_not(df_group[field_X].isna())
#                                    & np.logical_not(df_group[field_Y].isna())]
#                 artist, contour_artist = plot_interpolated(self.axes['Contour'][i],
#                                               plot_type=self.side_plot,
#                                               x_array=df_plot[var_X].values,
#                                               y_array=df_plot[var_Y].values,
#                                               vx_array=df_plot[field_X].values,
#                                               vy_array=df_plot[field_Y].values,
#                                               color_array='no_color' if self.plot_contour_background else None,
#                                               background_contour_kwargs=current_kwargs if (
#                                                       self.plot_contour_background and self.side_plot != 'Contour') else False,
#                                               resolution=self.resolution, **{'kwargs': current_kwargs})
#
#                 if self.plot_contour_background:
#                     artist = contour_artist
#             if artist is not None:
#                 self.fig_plots.colorbar(artist, cax=self.colorbar_axes['Contour'][i], label='$V_H$ (m/s)' if i > 2 else '$V_Z$ (m/s)')
#
#             self.artists['Contour'].append(artist)
#
#             # Style
#             self.axes['Contour'][i].set_title(titles_dict[plot_type])
#             if self.projection_string == 'XY':
#                 self.axes['Contour'][i].set_aspect('equal')
#             else:
#                 self.axes['Contour'][i].set_aspect('auto')
#             self.axes['Contour'][i].set_xlabel(f'{self.plotting_X_var} (m)')
#             self.axes['Contour'][i].set_ylabel(f'{self.plotting_Y_var} (m)')
#
#     def plot_radius(self):
#         self.axes['radius'].clear()
#
#         # =======================================
#         #           SCATTER PLOTS
#         # =======================================
#         # VERTICAL
#         self.axes['radius'].scatter(self.current_data['rho_bird_TC'],
#                                     self.current_data[f'dZdT_air'],
#                                     label='Calc. $V_{Z}$',
#                                     **self.plotting_args['scatter_1'],
#                                     )
#         # HORIZONTAL
#         self.axes['radius'].scatter(self.current_data['rho_bird_TC'],
#                                     self.current_data['V_horizontal'],
#                                     label='Calc $V_H$',
#                                     **self.plotting_args['scatter_2'], )
#
#         # ==============================================================================
#         #                             MOVING AVERAGES
#         # ==============================================================================
#
#         # VERTICAL
#         self.axes['radius'].plot(self.current_data_rolling['rho']['rho_bird_TC_avg'],
#                                  self.current_data_rolling['rho']['vz_avg'],
#                                  label='M.A. $V_Z$',
#                                  **self.plotting_args['moving_average_1'])
#         # HORIZONTAL
#         self.axes['radius'].plot(self.current_data_rolling['rho']['rho_bird_TC_avg'],
#                                  self.current_data_rolling['rho']['vh_avg'],
#                                  label='M.A. $V_H$',
#                                  **self.plotting_args['moving_average_2'])
#
#         # ==============================================================================
#         #           REAL DATA (Get average and std for each rho)
#         # ==============================================================================
#
#         # VERTICAL
#
#         self.axes['radius'].plot(self.current_ground_truth_data_grouped['rho']['rho_bin'],
#                                  self.current_ground_truth_data_grouped['rho']['vz_real_avg'], 'C8--', alpha=1)
#
#         if np.any(self.current_ground_truth_data_grouped['rho']['vz_real_std'] != 0):
#             self.axes['radius'].fill_between(self.current_ground_truth_data_grouped['rho']['rho_bin'],
#                                              self.current_ground_truth_data_grouped['rho']['vz_real_avg']
#                                              + self.current_ground_truth_data_grouped['rho']['vz_real_std'],
#                                              self.current_ground_truth_data_grouped['rho']['vz_real_avg']
#                                              - self.current_ground_truth_data_grouped['rho']['vz_real_std'], color='C8', alpha=0.2)
#
#         self.axes['radius'].plot(self.current_ground_truth_data_grouped['rho']['rho_bin'],
#                                  self.current_ground_truth_data_grouped['rho']['vh_real_avg'], 'C8--', alpha=1)
#
#         if np.any(self.current_ground_truth_data_grouped['rho']['vz_real_std'] != 0):
#             self.axes['radius'].fill_between(self.current_ground_truth_data_grouped['rho']['rho_bin'],
#                                              self.current_ground_truth_data_grouped['rho']['vh_real_avg']
#                                              + self.current_ground_truth_data_grouped['rho']['vh_real_std'],
#                                              self.current_ground_truth_data_grouped['rho']['vh_real_avg']
#                                              - self.current_ground_truth_data_grouped['rho']['vh_real_std'], color='C8', alpha=0.2)
#
#         # PLOT STYLE
#         self.axes['radius'].set_title('Radial Profile')
#         self.axes['radius'].set_xlabel('$rho$ (m)')
#         self.axes['radius'].set_ylabel('$V_Z$, $V_{H}$ (m/s)')
#         self.axes['radius'].legend(bbox_to_anchor=(1.0, 1))
#
#     def plot_rotation(self, phi_resolution=10):
#         self.axes['rotation'].clear()
#
#         # =======================================
#         #           SCATTER PLOTS
#         # =======================================
#         # VERTICAL
#         self.axes['rotation'].scatter(self.current_data['phi_bird_TC'],
#                                       self.current_data[f'dZdT_air'],
#                                       **self.plotting_args['scatter_1'])
#         # HORIZONTAL
#         self.axes['rotation'].scatter(self.current_data['phi_bird_TC'],
#                                       self.current_data['V_horizontal'],
#                                       **self.plotting_args['scatter_2'])
#
#         # =======================================
#         #       MOVING AVERAGES
#         # =======================================
#
#         # VERTICAL
#         self.axes['rotation'].plot(self.current_data_rolling['phi']['phi_bird_TC_avg'],
#                                    self.current_data_rolling['phi']['vz_avg'],
#                                    **self.plotting_args['moving_average_1'])
#
#         # HORIZONTAL
#         self.axes['rotation'].plot(self.current_data_rolling['phi']['phi_bird_TC_avg'],
#                                    self.current_data_rolling['phi']['vh_avg'],
#                                    **self.plotting_args['moving_average_2'])
#
#         # ==============================================================================
#         #           REAL DATA (Get average and std for each PHI)
#         # ==============================================================================
#
#         # VERTICAL
#
#         self.axes['rotation'].plot(self.current_ground_truth_data_grouped['phi']['phi_bin'],
#                                    self.current_ground_truth_data_grouped['phi']['vz_real_avg'], 'C8--', alpha=1)
#
#         if np.any(self.current_ground_truth_data_grouped['phi']['vz_real_std'] != 0):
#             self.axes['rotation'].fill_between(self.current_ground_truth_data_grouped['phi']['phi_bin'],
#                                                self.current_ground_truth_data_grouped['phi']['vz_real_avg']
#                                                + self.current_ground_truth_data_grouped['phi']['vz_real_std'],
#                                                self.current_ground_truth_data_grouped['phi']['vz_real_avg']
#                                                - self.current_ground_truth_data_grouped['phi']['vz_real_std'], color='C8', alpha=0.2)
#
#         self.axes['rotation'].plot(self.current_ground_truth_data_grouped['phi']['phi_bin'],
#                                    self.current_ground_truth_data_grouped['phi']['vh_real_avg'], 'C8--', alpha=1)
#
#         if np.any(self.current_ground_truth_data_grouped['phi']['vz_real_std'] != 0):
#             self.axes['rotation'].fill_between(self.current_ground_truth_data_grouped['phi']['phi_bin'],
#                                                self.current_ground_truth_data_grouped['phi']['vh_real_avg']
#                                                + self.current_ground_truth_data_grouped['phi']['vh_real_std'],
#                                                self.current_ground_truth_data_grouped['phi']['vh_real_avg']
#                                                - self.current_ground_truth_data_grouped['phi']['vh_real_std'], color='C8', alpha=0.2)
#
#         # PLOT STYLE
#         self.axes['rotation'].set_title('Rotation')
#         self.axes['rotation'].set_xlabel('$phi$ (rad)')
#         self.axes['rotation'].set_ylabel('$V_Z$, $V_H$ (m/s)')
#
#         self.axes['rotation'].yaxis.tick_right()
#         self.axes['rotation'].yaxis.set_label_position("right")
#
#     def plot_white_shadow(self):
#         # TODO
#         # Get rid of edges, somehow...
#         current_alpha_value = 0.3
#         h = self.histogram_data['counts']
#         low_occupation_mask = self.artists['histogram'].get_array() < 3
#         low_occupation_mask = low_occupation_mask.astype(float)
#
#         x_edges = self.histogram_data['bins']['cartesian']['x']
#         y_edges = self.histogram_data['bins']['cartesian']['y']
#         alpha_value = current_alpha_value
#         color_array = np.ones((h.shape[0] * h.shape[1], 4))
#         color_array[:, -1] = alpha_value * low_occupation_mask.data.astype(float) + 0.0001
#         artist = QuadMesh(np.dstack([x_edges, y_edges]), color=color_array, zorder=10)
#         for i_ax, ax in enumerate(self.axes['Contour']):
#             self.artists['shadow'] = copy(artist)
#             ax.add_artist(self.artists['shadow'])
#
#     def toggle_red_light(self):
#         current_visible = self.circle.get_visible()
#         self.circle.set_visible(not current_visible)
#         self.fig_buttons.canvas.draw()
#         self.fig_buttons.show()
#
#     def go_plot(self, block=None):
#         self.plots_per_slice()
#         #self.plot_shades()
#         if self.interactive:
#             self.show(block=block)
#         # self.prepare_next_iteration()
#
#     def show(self, block=None):
#         if block is None:
#             block = self.block
#         self.fig_plots.canvas.draw()
#         # self.fig_plots.show()
#         if self.interactive:
#             self.fig_buttons.canvas.draw()
#             plt.show(block=block)
#
#     def save_figure(self, destination_folder, title=None):
#
#         os.makedirs(destination_folder, exist_ok=True)
#         if title is None:
#             title = self.fig_plots._suptitle.get_text()
#             title = title.replace(':', '').replace(' ', '_')
#
#         file_path = f'{destination_folder}/{title}.png'
#         try:
#             self.fig_plots.savefig(file_path)
#         except UserWarning as e:
#             print('ad')
#         print(f'saved at {file_path}')
#
#     def sweep_iterations(self, event, delay=3):
#         if self.animation is None:
#             self.current_iteration = 0
#             self.animation = FuncAnimation(fig=self.fig_plots, func=self.plot_next_iteration, repeat=False,
#                                            interval=delay,
#                                            frames=self.max_iteration)
#             print('animation started')
#             self.fig_plots.canvas.draw()
#             self.fig_plots.show()
#         else:
#             self.animation.pause()
#             self.animation = None
#             print('animation stopped')
#
#     def plot_next_iteration(self, something):
#         self.on_iteration_change(None, 1)
#         self.set_iteration_data()
#         self.plots_per_iteration()
#         self.set_sliced_data()
#         self.go_plot()
#
#         return ()
#
#     def sweep_var(self, event, delay=3, binned_var='Z_bird_TC'):
#         if self.animation is None:
#             list_of_bins = self.current_iteration_data[f'bin_index_{binned_var}'].unique()
#             self.animation = FuncAnimation(fig=self.fig_plots, func=self.plot_next_bin, repeat=False, interval=delay,
#                                            frames=len(list_of_bins))
#
#             FFwriter = FFMpegWriter()
#             print('animation started')
#             filename = f'animation{round(time.time())}.mp4'
#             self.animation.save(filename, writer=FFwriter)
#             print(f'animation saved in {filename}')
#             # self.fig_plots.canvas.draw()
#             # self.fig_plots.show()
#         else:
#             self.animation.pause()
#             self.animation = None
#             self.widgets['slider_Z'].set_min(self.current_limits['Z'][0])
#             self.widgets['slider_Z'].set_max(self.current_limits['Z'][1])
#             print('animation stopped')
#
#     def update_limits_on_button_figure(self):
#         for coord in ['X', 'Y', 'Z']:
#             self.widgets[f'slider_{coord}'].set_min(self.current_limits[coord][0])
#             self.widgets[f'slider_{coord}'].set_max(self.current_limits[coord][1])
#
#     def plot_next_bin(self, something):
#         print(something)
#         current_bin = self.current_bins_iteration[self.current_bins_iteration['bin_index_Z_bird_TC'] == something]
#
#         if current_bin.empty:
#             return ()
#         self.current_limits['Z'] = current_bin[['Z_bird_TC_min', 'Z_bird_TC_max']].values[0].tolist()
#         self.set_sliced_data()
#         self.go_plot()
#         return ()

class InspectDecomposedFlock:

    def __init__(self, df,
                 X_col='rho_bird_TC',
                 Y_col='phi_bird_TC',
                 Z_col='Z_bird_TC',
                 time_col='time',
                 color='bank_angle', bird_name_col='bird_name', kwargs_3d_plot=None, kwargs_line_plots=None):

        self.ax_flight_histograms = None
        self.n_self.steps = 4
        if kwargs_line_plots is None:
            kwargs_line_plots = {}
        if kwargs_3d_plot is None:
            kwargs_3d_plot = {}

        self.df = df.copy()
        self.current_df_bird = None
        self.current_iteration = 1
        self.current_self.step = 0
        self.n_iterations = 30

        self.current_bird_name = None
        self.position_coordinate_type = 'cylindrical'
        self.velocity_coordinate_type = 'cylindrical'
        self.position_cols = {'cylindrical': ['rho_bird_TC',
                                              'phi_bird_TC',
                                              'Z_bird_TC'],
                              'cartesian': ['X_bird_TC',
                                            'Y_bird_TC',
                                            'Z_bird_TC']}
        self.flight_cols = ['radius', 'curvature', 'bank_angle']
        self.bird_velocity_cols = {'cylindrical': ['dRhodT_bird',
                                                   'dPhidT_bird',
                                                   'dZdT_bird'],
                                   'cartesian': ['dXdT_bird',
                                                 'dYdT_bird',
                                                 'dZdT_bird']}
        self.air_velocity_cols = {'cylindrical': ['dRhodT_air',
                                                  'dPhidT_air',
                                                  'dZdT_air'],
                                  'cartesian': ['dXdT_air',
                                                'dYdT_air',
                                                'dZdT_air']}
        self.time_col = time_col
        self.color_col = color
        self.bird_name_col = bird_name_col

        self.self.step_label = {0: 'Initial',
                           1: 'Alignment and Wind Removal',
                           2: 'Local Averages',
                           3: 'Steady Flight Rules'}

        self.kwargs_3d_plot = {'s': 4,
                               'alpha': 1}
        self.kwargs_3d_plot.update(kwargs_3d_plot)
        self.kwargs_line_plot = {}
        self.kwargs_line_plot.update(kwargs_line_plots)
        self.fig = None
        self.ax3d = None
        self.ax_flight_line_plots = None
        self.ax_bird_velocity_line_plots = None
        self.unique_bird_names = self.df[self.bird_name_col].unique()
        self.current_bird_name_idx = 0
        self.current_bird_name = self.unique_bird_names[self.current_bird_name_idx]

        self.preprocessing()

        self.set_figure_and_axes()
        self.set_data()
        self.go_plot()

    def preprocessing(self):
        for velocity_type in ['air', 'bird']:
            for i in range(3+1):
                self.df[f'dRhodT_{velocity_type}_{i}'] = ((self.df['X_bird_TC'] * self.df[f'dXdT_{velocity_type}_{i}']
                                                           + self.df['Y_bird_TC'] * self.df[f'dYdT_{velocity_type}_{i}'])
                                                          / self.df['rho_bird_TC'])

                self.df[f'dPhidT_{velocity_type}_{i}'] = ((self.df['X_bird_TC'] * self.df[f'dYdT_{velocity_type}_{i}']
                                                           - self.df['Y_bird_TC'] * self.df[f'dXdT_{velocity_type}_{i}'])
                                                          / self.df['rho_bird_TC'] ** 2)

    def set_data(self):
        self.current_df_bird = self.df[self.df[self.bird_name_col] == self.current_bird_name]
        self.current_df_bird = self.current_df_bird[self.current_df_bird['iteration'] == self.current_iteration]
        self.current_df_bird = self.current_df_bird.sort_values(self.time_col)
        self.current_df_bird = self.current_df_bird.iloc[1:]

    def set_figure_and_axes(self):
        self.fig = plt.figure(figsize=(12, 8), )

        gridspec_wrapper = self.fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
        # ===============================        LEFT PANEL    ================================
        gridspec_flight_plots = GridSpecFromSubplotSpec(3, 2, wspace=0, hspace=0, width_ratios=[3, 1],
                                                          subplot_spec=gridspec_wrapper[0])

        self.ax_flight_line_plots = [self.fig.add_subplot(gridspec_flight_plots[0, 0]),
                                     self.fig.add_subplot(gridspec_flight_plots[1, 0]),
                                     self.fig.add_subplot(gridspec_flight_plots[2, 0])]
        self.ax_flight_histograms = [self.fig.add_subplot(gridspec_flight_plots[0, 1]),
                                     self.fig.add_subplot(gridspec_flight_plots[1, 1]),
                                     self.fig.add_subplot(gridspec_flight_plots[2, 1])]


        # ===============================     CENTER PANEL    =====================================
        gridspec_bird_velocity_plots = GridSpecFromSubplotSpec(3, 2, wspace=0, hspace=0, width_ratios=[3, 1],
                                                      subplot_spec=gridspec_wrapper[2])
        self.ax_bird_velocity_line_plots = [self.fig.add_subplot(gridspec_bird_velocity_plots[0, 0]),
                                            self.fig.add_subplot(gridspec_bird_velocity_plots[1, 0]),
                                            self.fig.add_subplot(gridspec_bird_velocity_plots[2, 0])]
        self.ax_bird_velocity_histograms = [self.fig.add_subplot(gridspec_bird_velocity_plots[0, 1]),
                                            self.fig.add_subplot(gridspec_bird_velocity_plots[1, 1]),
                                            self.fig.add_subplot(gridspec_bird_velocity_plots[2, 1])]

        # ===============================     RIGHT PANEL =====================================
        gridspec_air_velocity_plots = GridSpecFromSubplotSpec(3, 2, wspace=0, hspace=0, width_ratios=[3, 1],
                                                      subplot_spec=gridspec_wrapper[1])
        self.ax_air_velocity_line_plots = [self.fig.add_subplot(gridspec_air_velocity_plots[0, 0]),
                                           self.fig.add_subplot(gridspec_air_velocity_plots[1, 0]),
                                           self.fig.add_subplot(gridspec_air_velocity_plots[2, 0])]

        self.ax_air_velocity_histograms = [self.fig.add_subplot(gridspec_air_velocity_plots[0, 1]),
                                            self.fig.add_subplot(gridspec_air_velocity_plots[1, 1]),
                                            self.fig.add_subplot(gridspec_air_velocity_plots[2, 1])]

        for i in range(3):
            #self.ax_position_line_plots[i].sharey(self.ax_position_histograms[i])
            self.ax_bird_velocity_line_plots[i].sharey(self.ax_bird_velocity_histograms[i])
            self.ax_air_velocity_line_plots[i].sharey(self.ax_air_velocity_histograms[i])

            self.ax_air_velocity_histograms[i].yaxis.set_tick_params(labelleft=False)
            self.ax_bird_velocity_histograms[i].yaxis.set_tick_params(labelleft=False)
            self.ax_air_velocity_histograms[i].xaxis.set_tick_params(labelbottom=False)
            self.ax_bird_velocity_histograms[i].xaxis.set_tick_params(labelbottom=False)



        #self.ax_position_line_plots[0].get_shared_x_axes().join(*self.ax_position_line_plots, *self.ax_velocity_line_plots)

    def plot_3d(self):

        self.ax3d.clear()
        _, im = plot_tracks_scatter(ax=self.ax3d,
                                    x_data=self.current_df_bird[self.position_cols['cartesian'][0]],
                                    y_data=self.current_df_bird[self.position_cols['cartesian'][1]],
                                    z_data=self.current_df_bird[self.position_cols['cartesian'][2]],
                                    color_data=self.current_df_bird[self.color_col],
                                    **self.kwargs_3d_plot)

        # Core Estimate

        x_lims = self.ax3d.get_xlim()
        self.ax3d.set_ylim(self.current_df_bird[self.position_cols['cartesian'][1]].mean() - (x_lims[1] - x_lims[0]) / 2,
                           self.current_df_bird[self.position_cols['cartesian'][1]].mean() + (x_lims[1] - x_lims[0]) / 2)
        self.ax3d.set_anchor('C')
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Z (m)')
        #if color:
        #    cbar = self.fig.colorbar(im[0], ax=ax3d, label='Bank Angle (rad)')
        #else:
        #    cbar = None

    def plot_position_line_plots(self):

        for i, coord in enumerate(self.position_cols[self.position_coordinate_type]):
            self.ax_position_line_plots[i].clear()
            self.ax_position_line_plots[i].plot(self.current_df_bird[self.time_col],
                                              self.current_df_bird[coord],
                                              **self.kwargs_line_plot)
            self.ax_position_line_plots[i].set_xlabel('time (s)')
            self.ax_position_line_plots[i].set_ylabel(coord)
            self.ax_position_line_plots[i].grid()

    def plot_bird_velocity_line_plots(self):

        for i, col in enumerate(self.bird_velocity_cols[self.velocity_coordinate_type]):
            self.ax_bird_velocity_line_plots[i].clear()
            self.ax_bird_velocity_line_plots[i].plot(self.current_df_bird[self.time_col],
                                                     self.current_df_bird[f'{col}_{self.current_self.step}'],
                                                     **self.kwargs_line_plot)
            self.ax_bird_velocity_line_plots[i].set_xlabel('time (s)')
            self.ax_bird_velocity_line_plots[i].set_ylabel(col)
            self.ax_bird_velocity_line_plots[i].grid()

    def plot_air_velocity_line_plots(self):

        for i, col in enumerate(self.air_velocity_cols[self.velocity_coordinate_type]):
            self.ax_air_velocity_line_plots[i].clear()
            self.ax_air_velocity_line_plots[i].plot(self.current_df_bird[self.time_col],
                                                    self.current_df_bird[f'{col}_{self.current_self.step}'],
                                                    **self.kwargs_line_plot)
            self.ax_air_velocity_line_plots[i].set_xlabel('time (s)')
            self.ax_air_velocity_line_plots[i].set_ylabel(col)
            self.ax_air_velocity_line_plots[i].grid()

    def plot_flight_line_plots(self):

        for i, col in enumerate(self.flight_cols):
            self.ax_flight_line_plots[i].clear()
            self.ax_flight_line_plots[i].plot(self.current_df_bird[self.time_col],
                                              self.current_df_bird[f'{col}'],
                                              **self.kwargs_line_plot)
            self.ax_flight_line_plots[i].set_xlabel('time (s)')
            self.ax_flight_line_plots[i].set_ylabel(col)
            self.ax_flight_line_plots[i].grid()

    def plot_histograms(self):

        #for i, col in enumerate(self.position_cols[self.position_coordinate_type]):
        #    self.ax_position_histograms[i].clear()
        #    self.ax_position_histograms[i].hist(self.current_df_bird[f'{col}'],
        #                                     bins=20, orientation='horizontal')

        for i, col in enumerate(self.bird_velocity_cols[self.velocity_coordinate_type]):
            hist_data = self.current_df_bird[f'{col}_{self.current_self.step}'].values
            outlier_mask = is_outlier(hist_data, thresh=2.5)
            self.ax_bird_velocity_histograms[i].clear()
            self.ax_bird_velocity_histograms[i].hist(hist_data[outlier_mask], bins=20, orientation='horizontal')

        for i, col in enumerate(self.air_velocity_cols[self.velocity_coordinate_type]):
            hist_data = self.current_df_bird[f'{col}_{self.current_self.step}'].values
            outlier_mask = is_outlier(hist_data, thresh=3.5)
            self.ax_air_velocity_histograms[i].clear()
            self.ax_air_velocity_histograms[i].hist(hist_data[outlier_mask], bins=20, orientation='horizontal')

        for i, col in enumerate(self.flight_cols):
            hist_data = self.current_df_bird[f'{col}'].values
            outlier_mask = is_outlier(hist_data, thresh=3.5)
            self.ax_flight_histograms[i].clear()
            self.ax_flight_histograms[i].hist(hist_data[outlier_mask], bins=20, orientation='horizontal')

    def on_press(self, event):
        if event.key == 'up':
            print('up pressed')
            if self.current_self.step == self.n_self.steps - 1:
                self.current_self.step = 0
                self.current_iteration = self.current_iteration + 1
            else:
                self.current_self.step = self.current_self.step + 1
            self.set_data()
            self.go_plot()
        elif event.key == 'down':
            if self.current_self.step == 0:
                self.current_self.step = self.n_self.steps - 1
                self.current_iteration = self.current_iteration - 1
            else:
                self.current_self.step = self.current_self.step - 1
            self.set_data()
            self.go_plot()
        elif event.key == 'left':
            print('i pressed')
            self.current_iteration = self.current_iteration - 1
            self.current_iteration = self.current_iteration % self.n_iterations
            self.set_data()
            self.go_plot()
        elif event.key == 'right':
            print('i pressed')
            self.current_iteration = self.current_iteration + 1
            self.current_iteration = self.current_iteration % self.n_iterations
            self.set_data()
            self.go_plot()
        elif event.key == 'b':
            print('b pressed')
            self.current_bird_name_idx = self.current_bird_name_idx + 1
            self.current_bird_name_idx = self.current_bird_name_idx % len(self.unique_bird_names)
            self.current_bird_name = self.unique_bird_names[self.current_bird_name_idx]
            self.set_data()
            self.go_plot()
        elif event.key == 'v':
            print('v pressed')
            self.current_bird_name_idx = self.current_bird_name_idx - 1
            self.current_bird_name_idx = self.current_bird_name_idx % len(self.unique_bird_names)
            self.current_bird_name = self.unique_bird_names[self.current_bird_name_idx]
            self.set_data()
            self.go_plot()
        elif event.key == 'c':
            if self.position_coordinate_type == 'cartesian':
                self.velocity_coordinate_type = 'cylindrical'
                self.position_coordinate_type = 'cylindrical'
            else:
                self.velocity_coordinate_type = 'cartesian'
                self.position_coordinate_type = 'cartesian'

            self.set_data()
            self.go_plot()

    def set_interactivity(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

    def show(self):
        self.fig.suptitle(f'bird name: {self.current_bird_name}\n '
                          f'Iteration: {self.current_iteration}  Step: {self.current_self.step} - '
                          f'{self.self.step_label[self.current_self.step]}')

        self.fig.tight_layout()
        plt.show(block=False)

    def go_plot(self):
        #self.plot_3d()
        self.plot_histograms()
        #self.plot_position_line_plots()
        self.plot_flight_line_plots()
        self.plot_bird_velocity_line_plots()
        self.plot_air_velocity_line_plots()
        self.set_interactivity()
        self.show()

