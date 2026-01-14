# Reverse Engineering of Thermal Soaring

## Simulate Datasets

    python run_flock_simulation.py glider_generate.yaml

The yaml configuration file comprises three parts: run, bird and air. Please check `config/default/bird_generate.yaml` for the explanation of all parameters

### run
This section contains parameters necessary for the simulation itself to run, such as, number of gliders in the flock,
time increment between iteration  or where to save the files.

### bird
This section determines the physical/aerodynamic and behavioural characterists of each glider as well as their 
initial conditions. Each of these values can be set using one of three way:
1. Constant Value: if a constant value is given, this value will be shared by all gliders
   1. For example, if `CL: 1.5` is given then all gliders will have the lift coefficient equal to 1.5
2. List: if a list is given for a certain quantity, these values will be used in the given order to determine the 
gliders characteristics. The length of this list must match the number of gliders to be simulated, as determined on the run_parameters parts
   1. For example, if `exploration_exploitation_ratio: 0.4, 0.5, 0.6` is given, the first simulated glider will hold the value 0.4, the second glider will hold 0.5 and the three (and last) will hold 0.6
3. Randomly Sampled: if a struture as the one below, n_gliders samples will be drawn by the indicated distribution with the parameters provided.
    ```
    distribution: 'uniform'
    parameters:
      low: 25
      high: 35
   ```
   In this example, n_glider samples will be drawn from a uniform distribution with bounds 25 and 35. 

### air
This section determines the air velocity field to be generated and it comprised three components: wind, thermal and 
turbulence; thermal is further divided into profile (thermal vertical velocity) and rotation (thermal horizontal velocity).

#### wind
The wind is defined, in m/s, as a 2-dimensional field depending on the three components of space and time. 
It may be defined as:
1. Constant value: if a constant list of size 2 is given, the wind will be constant in time and in space in the x and y direction.
   1. For example, `[1,2]` will result in a constant wind of 1 m/s along the x-direction and 2 m/s in the y-direction. 
2. Function: a function may be defined on a separate file where the correspondence between the space and time and the 
 wind is determined. This function must accept the position (a list of size 3) as a first argument, and the time as 
second argument; further arguments may be defined after these two. These function must return a list of size 2, 
corresponding to the x and y directions of the wind. For instance, a rotating wind may be defined by creating the following 
function with name `rotating_wind` on a file called `my_functions.py`   
   ```
   def rotating_wind(X, t, magnitude, period):
       import numpy as np
       z = X[-1]
       return [magnitude * np.cos(2 * np.pi * z / period),
               magnitude * np.sin(2 * np.pi * z / period)]
      ```
   To use this function, the wind section of the yaml configuration file would look like:
   ```
      from_function:
        file: 'my_functions.py'
        function: rotating_wind
        parameters:
          magnitude: 4
          period: 100
   ```
3. Randomly generated velocities: two methods may be used: 
   1. `random` - A distribution (determined by `distribution` and `parameters` keywords) will be used to sample the wind 
   velocities in a grid (defined with the keywords `n_steps` and `limits`). In the following example a gaussian with
   mean $\mu=0$ and $\sigma=0.4$ will be used to sample the 2 components of the wind in a 10x10 grid spanning the region
   $x \in [-1000, 1000]$ and $z \in [0, 1000]$.
   ```
   generate:
     method: random
      distribution: 'normal'
      parameters:
        loc: 0
        scale: 0.4
      n_steps: 10
      limits:
        x: [-1000, 1000]
        z: [0, 1000]
   ```
   2. `random_walk` - A gaussian multidimensional random walk will define each component of the wind velocity 
   indepedently in a grid (defined with the keywords `n_steps` and `limits`). In the following example a 2-dimensional gaussian random walk with mean $\mu=1$ and $\sigma=0.2$ will be used to determine each component of the wind in a 10x10 grid spanning the region
   $x \in [-1000, 1000]$ and $z \in [0, 1000]$.
   ```
   generate:
    method: random_walk
    mean: 1
    std: 0.2
      n_steps: 10
      limits:
        x: [-1000, 1000]
        z: [0, 1000]
   ```
4. From data: one can define the wind directly from data by giving a list of positions and time (`XYZT_values`)and a list 
of wind values (`values`). Both list must have the same length. One last parameter (`used_vars`)  must be given to 
indicate which coordinates were used in `XYZT_values`. These lists are then used to defined a function by linear interpolation.
This is added to the yaml configuration file with the structure
in the following example:
   ```
    from_data:
      'XYZT_values': [[-1000,0],
                      [1000, 0],
                      [-1000,1000] ,
                      [1000,1000]]
      'values': [[0,0],
                 [0,0],
                 [1,2],
                 [2,4]]
      'used_vars': 'XZ'
   ```
   In this example the wind depends only on the X and Z coordinates and is defined in the region 
$x \in [-1000,1000]$ and $z \in [0,1000]$. It always vanishes at ground level (first two line in the `values`) 
and grows in altitude to $[1,2]$ at $(x,z)=(-1000, 1000)$ and doubles in intensity after two kilometers at $(x,z)=(1000, 1000)$    
   
   
