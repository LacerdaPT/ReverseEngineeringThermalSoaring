import copy

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap


def set_turbulence_cmap():
    mm = LinearSegmentedColormap.from_list('turbulence_cmap', ((0, (1,1,1)),
                                                               (0.5, (0, 0 ,1)),
                                                               (1.0,( 0,0,0.2 ))
                                                               ), N=100)
    mm_r = LinearSegmentedColormap.from_list('turbulence_cmap_r', ((0,( 0,0,0.2 )),
                                                               (0.5, (0, 0 ,1)),
                                                               (1.0, (1,1,1))
                                                                   ), N=100)
    if 'turbulence_cmap' not  in  matplotlib.colormaps:
        matplotlib.colormaps.register(mm)
        matplotlib.colormaps.register(mm_r)

def science_matplotlib_config(figsize_multiplier, save):
    import scienceplots
    matplotlib.style.use(['science'])
    if save:
        matplotlib.use('Cairo')
        matplotlib.interactive(False)


    plt.rc('axes', titlesize=figsize_multiplier * 9)  # fontsize of the axes title
    plt.rc('axes', labelsize=figsize_multiplier * 9)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=figsize_multiplier * 8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=figsize_multiplier * 8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=figsize_multiplier * 8)  # legend fontsize
    plt.rc('axes', linewidth=figsize_multiplier * 0.5)

    plt.rc('lines', linewidth=figsize_multiplier * 0.5)
    plt.rc('lines', markersize=figsize_multiplier * 2)
    plt.rc('grid', linewidth=figsize_multiplier * 0.28)
    plt.rcParams.update({
        "text.usetex":     True,
        "font.family":     "sans-serif",
        "font.sans-serif": "Helvetica",
        'xtick.minor.width': 0.28 * figsize_multiplier,
        'ytick.minor.width': 0.28 * figsize_multiplier,
        'xtick.major.width': 0.28 * figsize_multiplier,
        'ytick.major.width': 0.28 * figsize_multiplier
    })
    # plt.rc('figure',dpi=320)


stork_dataset_renaming_dict ={'b010_0.1': 'R0',
                              'b023_0.1': 'R1',
                              'b023_1.1': 'R1.1',
                              'b072_0.1': 'R2',
                              'b077_0.1': 'R3',
                              'b112_0.2': 'R4',
                              'b121_0.1': 'R5'}
parameter_parser = {'rotation': lambda p: {'realization':      p.parents[4].name,
                                           'parameter1_name':  p.parents[6].name.split('=')[0],
                                           'parameter1_value': float(p.parents[6].name.split('=')[1]),
                                           'parameter2_name':  p.parents[5].name.split('=')[0],
                                           'parameter2_value': float(p.parents[5].name.split('=')[1]), },
                    'storks': lambda p: {'thermal': p.parents[6].name,
                                         'bin_z_size': float(p.parents[3].name.split('=')[-1])}
                    }
digital_twin_renaming_dict ={0: 'S0',
                              1: 'S1',
                              2: 'S2',
                              3: 'S3',
                              4: 'S4',
                              5: 'S5'}
turbulence_lookup_mpers = {0: 0.22,
                           1: 0.30,
                           2: 0.28,
                           3: 0.42,
                           4: 0.41,
                           5: 0.61, }

digital_twin_config_dict = {
    '0': {'zlims':                            [200,400],
                            'norm_horizontal_i' :Normalize(0,8),
                            'norm_vertical_i' : Normalize(-2,7),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':30,
                            'azim':160
                            },
               '1': {'zlims': [100,400],
                            'norm_horizontal_i' :Normalize(2.5,3.2),
                            'norm_vertical_i' : Normalize(-1,5),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':35,
                            'azim':145
                            },
               '2': {'zlims': [200,400],
                            'norm_horizontal_i' :Normalize(2.5,3),
                            'norm_vertical_i' : Normalize(-1,5),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':35,
                            'azim':200
                            },
               '3': {'zlims': [150,550],
                            'norm_horizontal_i' :Normalize(-1,1.5),
                            'norm_vertical_i' : Normalize(-1,4.5),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':35,
                            'azim':50
                            },
               '4': {'zlims': [300,500],
                            'norm_horizontal_i' :Normalize(-0.5,2),
                            'norm_vertical_i' : Normalize(-1,5),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':35,
                            'azim':350
                            },
               '5': {'zlims': [280,480],
                            'norm_horizontal_i' :Normalize(3,5),
                            'norm_vertical_i' : Normalize(-1,5.5),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':25,
                            'azim':190
                            },
               }


storks_config_dict = {'b010_0.1': {'zlims':                     [950, 1200],
                            'norm_horizontal_i' :Normalize(0,8),
                            'norm_vertical_i' : Normalize(-2,7),
                            'negative_boundary': -2,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':40,
                            'azim':160
                            },
               # 'b023_0.1': {'zlims': None,
               #                  'quiver_kwargs':         copy.deepcopy(quiver_kwargs_template),
               #              'norm_horizontal_i' :Normalize(-1,7),
               #              'norm_vertical_i' : Normalize(-2,6),
               #              'negative_boundary': -2,
               #              'boundary': 1,
               #              'inset_bounds':[0.65, 0.05, 0.3, 0.3],
               #              'elev':30,
               #              'azim':150
               #              },
               # 'b023_1.1': {'zlims': None,
               #                  'quiver_kwargs':         copy.deepcopy(quiver_kwargs_template),
               #              'norm_horizontal_i' :Normalize(-1,7),
               #              'norm_vertical_i' : Normalize(-2,6),
               #              'negative_boundary': -2,
               #              'boundary': 1,
               #              'inset_bounds':[0.65, 0.05, 0.3, 0.3],
               #              'elev':30,
               #              'azim':150
               #              },
               'b072_0.1': {'zlims': None,
                            'norm_horizontal_i':        Normalize(1,3),
                            'norm_vertical_i':           Normalize(-2,6),
                            'negative_boundary': -2,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':30,
                            'azim':200
                            },
               'b077_0.1': {'zlims':  [1300, 1477],
                            'norm_horizontal_i' :Normalize(0,2),
                            'norm_vertical_i' : Normalize(-1,5.2),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':30,
                            'azim':50
                            },
               'b112_0.2': {'zlims':[1370, 1500],
                            'norm_horizontal_i' : Normalize(5.5,6),
                            'norm_vertical_i' : Normalize(-1,5.5),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':25,
                            'azim':52
                            },
               'b121_0.1': {'zlims': [1400,1650],
                            'norm_horizontal_i' :Normalize(3.5,5.5),
                            'norm_vertical_i' : Normalize(-1,6),
                            'negative_boundary': -1,
                            'boundary': 1,
                            'inset_bounds':[0.65, 0.05, 0.3, 0.3],
                            'elev':30,
                            'azim':90
                            },
               }