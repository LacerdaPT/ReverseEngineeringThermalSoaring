# Reverse Engineering of Thermal Soaring

## Brief project description.
This repository contains the code used in the project "Soaring, aerodynamics and complex flows in thermals". 

## Installation instructions.
### Clone repository
### Dependencies
1. python 3.9
2. pip
### Create a virtual environment which uses Python 3.9
Open a command line, navigate to the project root and run:
`python3.9 -m venv VENV_DIR`
### Activate the virtual environment
Open the command line, navigate to the project root and run the following command:

On Linux and MacOS: 
`source VENV_DIR/bin/activate`

On Windows: 
`VENV_DIR\Scripts\activate.bat`

Note: for further details please refer to https://docs.python.org/3.9/library/venv.html
### Install dependencies
`pip install -r requirements.txt`

## Usage examples.
### Run Flock Simulation
`python run_flock_simulation.py config/examples/bird_generate.yaml`

Check `config/default/simulate_datasets.md` for details.
### Run reverse-engineering procedure
`python run_reverse_engineer.py -y config/examples/decomposition_parameters.yaml`

Check `config/default/decomposition_parameters.default.yaml` for details.
### Run reverse-engineering procedure and optimzation
`python run_reverse_engineering_and_optimization.py -y config/examples/simulated_annealing_args.yaml`

Check `config/default/simulated_annealing_args.default.yaml` and `config/default/decomposition_parameters.default.yaml` 
for details.
### Post-Processing
#### On reverse-engineering output data 
`python run_reverse_engineering_postprocess.py example/decomposition`

For details run: `python run_reverse_engineering_postprocess.py --help`

`python run_further_decomposition.py example/decomposition`

For details run: `python run_further_decomposition.py --help`
#### On optimization procedure
`python run_optimization_postprocessing.py config/examples/simulated_annealing_args.yaml`

For details run: `python run_optimization_postprocessing.py --help`

### Analyses
`python run_air_autocorrelation_estimation.py example/decomposition`

For details run: `python run_air_autocorrelation_estimation.py --help`

`python run_air_fluctuations_calculation.py example/decomposition`

## Contact information.
Please get in touch at p.v.lacerda(at)gmail.com
