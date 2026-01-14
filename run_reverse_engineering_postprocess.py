import argparse
import copy

from calc.post_processing.air_velocity_field import full_postprocessing_pipeline

default_inter_kwargs = {'smoothing':0.0,
                'kernel':'thin_plate_spline',
                }
default_final_inter_kwargs = {'smoothing':0.0,
                      'kernel':'thin_plate_spline',
                      }
default_binning_parameters = {'n_bins':          [100, 20],
                      'method':          'adaptive',
                      'adaptive_kwargs': {'adaptive_bin_max_size': 2,
                                          'adaptive_bin_min_size': 0.1,
                                          'max_bin_count':         1},
                      }
root_path = '/home/pedro/PycharmProjects/ReverseEngineeringThermalSoaring'

parser = argparse.ArgumentParser('Post-processing of reverse engineering data')


parser.add_argument('-pd', '--path-decompostion', dest='path_to_decomposition', type=str,
                    help='path to the decomposition folder')

parser.add_argument('--mc', dest='min_occupation_number', type=int,
                    help='minimun occupation number on each bin to be considered for interpolation', default=3)

parser.add_argument('--mr', dest='rho_quantile', type=int,
                    help='percentile of the points distribution in the radial direction to be used as the '
                         'maximum distance for the valid region. Number between 0 and 100.', default=90)


parser.add_argument('-bz', '--n-bins-z', dest='n_bins_z', type=int, default=100,
                    help='number of bins in the Z direction')

parser.add_argument('-bphi', '--n-bins-phi', dest='n_bins_phi', type=int, default=20,
                    help='number of bins in the phi direction')



def main():
    args = parser.parse_args().__dict__
    path_to_decomposition = args['path_to_decomposition']
    min_occupation_number = args['min_occupation_number']
    rho_quantile = args['rho_quantile']
    inter_kwargs = copy.deepcopy(default_inter_kwargs)
    final_inter_kwargs = copy.deepcopy(default_final_inter_kwargs)
    binning_parameters = copy.deepcopy(default_binning_parameters)
    binning_parameters = binning_parameters | {'n_bins': [args['n_bins_z'], args['n_bins_phi']]}


    full_postprocessing_pipeline(path_to_decomposition,
                                 min_occupation_number=min_occupation_number,
                                 rho_quantile=rho_quantile,
                                 inter_kwargs=inter_kwargs,
                                 final_inter_kwargs=final_inter_kwargs,
                                 binning_parameters=binning_parameters,
                                 save=True)

if __name__ == '__main__':
    main()
