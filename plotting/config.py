import matplotlib
from matplotlib.colors import LinearSegmentedColormap

mm = LinearSegmentedColormap.from_list('turbulence_cmap', [(0, (1,1,1)),
                                                           (0.5, (0, 0 ,1)),
                                                           (1.0,( 0,0,0.2 ))
                                                           ], N=100)

if 'turbulence_cmap' not  in  matplotlib.colormaps:
    matplotlib.colormaps.register(mm)

mm = LinearSegmentedColormap.from_list('thermal_cmap',
                                       [((-2 + 2) / 7,(0xa5 / 255.0, 0x54/ 255.0, 0x97/ 255.0)), # Purple
                                                ((1.0 + 2) / 7,(0xc6 / 255.0, 0xdf/ 255.0, 0xf8/ 255.0)), # Sky Blue
                                                #((3.5 + 2) / 7,(0xe8 / 255.0, 0xf2/ 255.0, 0x87/ 255.0)), # Greenish
                                                ((2.0 + 2) / 7,(0xff / 255.0, 0xd2/ 255.0, 0x3b/ 255.0)), # Yellow
                                                ((4.0 + 2) / 7,(0xe1 / 255.0, 0x23/ 255.0, 0x23/ 255.0)), # Red
                                                ((5 + 2) / 7,(0xa0 / 255.0, 0x02/ 255.0, 0x02/ 255.0)) # Inferno Red
], N=100)
# (5 + 2) / 7,
# (4 + 2) / 7,
# (2 + 2) / 7,
# (1 + 2) / 7,
# (0 + 2) / 7,
# (-2 + 2) / 7,
if 'thermal_cmap' not  in  matplotlib.colormaps:
    matplotlib.colormaps.register(mm)