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
### Generate synthetic datasets
`python run_flock_simulation.py config/examples/bird_generate.yaml`
### Run reverse-engineering procedure
`python run_reverse_engineer.py -y config/examples/decomposition_parameters.yaml`
### Run reverse-engineering procedure and optimzation
`python run_reverse_engineering_and_optimization.py -y config/examples/simulated_annealing_args.yaml`

### Post-Processing
#### On reverse-engineering output data 
`python run_reverse_engineering_postprocess.py example/decomposition`

`python run_further_decomposition.py example/decomposition`
#### On optimization procedure
`python run_optimization_postprocessing.py config/examples/simulated_annealing_args.yaml`

### Analyses
`python run_air_autocorrelation_estimation.py example/decomposition`

`python run_air_fluctuations_calculation.py example/decomposition`

## Contact information.
Please get in touch at p.v.lacerda(at)gmail.com
