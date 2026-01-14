import os.path
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd



from calc.analysis.turbulence import get_single_sweep_autocorrelation, get_air_autocorrelation
from object.air import ReconstructedAirVelocityField

parser = ArgumentParser()
parser.add_argument('-i', '--input-folder', dest='path_to_decomposition', type=str)
parser.add_argument('-o', '--output-folder', dest='save_folder', type=str, default=None)
parser.add_argument('-gs', '--grid-size', dest='grid_size', type=float, default=2)
parser.add_argument('-lx', '--size-x', dest='lx', type=float, default=50)
parser.add_argument('-ly', '--size-y', dest='ly', type=float, default=50)
parser.add_argument('-lz', '--size-z', dest='lz', type=float, default=20)
parser.add_argument('-ns', '--not-synthetic', action='store_false', dest='is_synthetic')
parser.add_argument('-dr', '--dry-run', action='store_false', dest='save')

def main():
    args = parser.parse_args()

    path_to_decomposition = args.path_to_decomposition
    save_folder = args.save_folder
    save = args.save
    is_synthetic = args.is_synthetic
    grid_size=args.grid_size
    lx = args.lx
    ly = args.ly
    lz = args.lz

    if save_folder is None:
        save_folder = os.path.join(path_to_decomposition, 'results')
    if save:
        os.makedirs(save_folder, exist_ok=True)

    df = pd.DataFrame()
    df_all_centers = pd.DataFrame()

    if is_synthetic:
        list_of_datatypes = ['dec', 'gt']
    else:
        list_of_datatypes = ['dec']
    for datatype in list_of_datatypes:
        if datatype == 'gt':
            path_to_decomposition = os.path.join(path_to_decomposition, 'ground_truth_reconstructed')

        print(path_to_decomposition)
        decomposed_avf = ReconstructedAirVelocityField.from_path(path_to_files=path_to_decomposition, max_extrapolated_distance=0)

        current_datatype_avg, current_datatype_all_centers = get_air_autocorrelation(decomposed_avf,
                                                                                     lz=lz,
                                                                                     grid_size=grid_size, lx=lx, ly=ly,
                                                                                     average_removed=False,
                                                                                     air_component='thermal')

        current_datatype_avg['size'] = lz
        current_datatype_avg['datatype'] = datatype
        df = pd.concat([df, current_datatype_avg])
        current_datatype_all_centers['size'] = lz
        current_datatype_all_centers['datatype'] = datatype
        df_all_centers = pd.concat([df_all_centers, current_datatype_all_centers])
    if save:
        df.to_csv(os.path.join(save_folder, 'autocorrelation.csv'), index=False)
        df_all_centers.to_csv(os.path.join(save_folder, 'autocorrelation_all_centers.csv'), index=False)


if __name__ == '__main__':

    main()

