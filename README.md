<div align="center">

# QES: Quick Environmental Simulations

<!-- Badges -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7098280.svg)](https://doi.org/10.5281/zenodo.7098280)
<!-- Badges -->

</div>

The Quick Environmental Simulation (***QES***) code is a low-computational-cost framework designed to compute high-resolution wind and concentration fields in complex atmospheric-boundary-layer environments. QES is written in C++ and NVIDIA's CUDA for Graphics Processing Unit (GPU) acceleration. The code uses NVIDIA's dynamic parallelism API to substantially accelerate simulations. ***QES requires a NVIDIA GPU with Compute Capability of 7.0 (or higher)***.


### QES-Winds

QES-Winds is a fast-response 3D diagnostic urban wind model using a mass-conserving wind-field solver. QES-Winds uses a variational analysis technique to ensure the conservation of mass rather than slower yet more physics-based solvers that include the conservation of momentum. QES-Winds minimizes the difference between an initial wind field that is specified using empirical parameterizations and the final wind field. This method requires the solution of a Poisson equation for Lagrange multipliers. The Poisson equation is solved using the Successive Over-Relaxation (SOR) method (an iterative solver), which is a variant of the Gauss-Seidel method with more rapid convergence. 

> B. Bozorgmehr et al., “Utilizing dynamic parallelism in CUDA to accelerate a 3D red-black successive over relaxation wind-field solver,” *Environ Modell Softw*, vol. 137, p. 104958, 2021, doi: [10.1016/j.envsoft.2021.104958](https://doi.org/10.1016/j.envsoft.2021.104958).

### QES-Turb

QES-Turb is a turbulence model based on Prandtl’s mixing-length and Boussinesq eddy-viscosity hypotheses. QES-Turb computes the stress tensor using local velocity gradients and some emprical non-local parameterizations.

### QES-Plume

QES-Plume is a stochastic Lagrangian dispersion model using QES-Winds mean wind field and QES-Turb turbulence fields. QES-Plume solves the generalized Langevin equations to compute the fluctuations of the particle in the turbulent flow fluid. A time-implicit integration scheme is used to solve the Langevin equation, eliminating 'rogue' trajectories. The particles are advanced using a forward Euler scheme. QES-Plume is also a stand-alone dispersion model that can run using fields from diverses sources such as RANS or LES models. 

> F. Margairaz et al, "QES-Plume: QES-Plume v1.0: A Lagrangian dispersion model," *Geosci Model Dev*, SUBMITTED

### QES-Fire

QES-Fire is a microscale wildfire model coupling the fire front to microscale winds. The model consists of a simplified physics rate of spread model, a kinematic plume-rise model, and a mass-consistent wind solver. The QES-Fire module is currently not publicly available. 

> M. J. Moody et al., “QES-Fire: a dynamically coupled fast-response wildfire model,” *Int J Wildland Fire*, vol. 31, no. 3, pp. 306–325, 2022, doi: [10.1071/wf21057](https://doi.org/https://doi.org/10.1071/WF21057).

## Package Requirements

***QES requires the CUDA library and a NVIDIA GPU with Compute Capability of 7.0 (or higher).***

**Note:** a compliation option for configuration without GPU will be added in the future.

On a general Linux system, such as Ubuntu 18.04 or 20.04, the following packages need to be installed:
* libgdal-dev
* libnetcdf-c++4-dev
* libnetcdf-cxx-legacy-dev
* libnetcdf-dev
* netcdf-bin
* libboost-all-dev
* cmake
* cmake-curses-gui

If the system uses ```apt```, the packages can be installed using the following command:
```
apt install libgdal-dev libnetcdf-c++4-dev  libnetcdf-cxx-legacy-dev libnetcdf-dev netcdf-bin libboost-all-dev cmake cmake-curses-gui
```

To build the code and to use the GPU system, you will need a NVIDIA GPU with the CUDA library installed.  The code has been tested with CUDA 8.0, 10.0, 10.1, and 10.2. If your version of CUDA is installed in a non-uniform location, you will need to remember the path to the cuda install directory.

Additionally, the code can use NVIDIA's OptiX to accelerate various computations. Our OptiX code has been built to use version 7.0 or higher.

## Building the Code

On the public repository, the most recent released version of the code is available in the *main* branch. 

On the private repository, the most recent stable version of code is available in the *main* branch. The most active development occurs in the *workingBranch*. We suggest you use the main branch for production and the workingBranch for the most recent feature. You can checkout this branch with the following git command:
```
git checkout workingBranch
```
If you are unsure about which branch you are on, the ``` git status ``` command can provide you with this information.


### Building on General Linux System

We separate the build 
```
mkdir build
cd build
cmake ..
```
You can then build the source:
```
make
```

### Building on CHPC Cluster (University of Utah)

The code does run on the CHPC cluster. You need to make sure the correct set of modules are loaded.  Currently, we have tested a few configurations that use
- GCC 5.4.0 and CUDA 8.0
- CCC 8.1.0 and CUDA 10.1 (10.2)
- GCC 8.5.0 and CUDA 11.4
If you build with OptiX support, you will need to use CUDA 10.2 or newer configuration. Any builds (with or without OptiX) with CUDA 11.4 are preferred if you don't know which to use. Older configurations are provided in `CHPC/oldBuilds.md`.

After logging into your CHPC account, you will need to load specific modules. In the following sections, we outline the modules that need to be loaded along with the various cmake command-line calls that specify the exact locations of module installs on the CHPC system.  

#### CUDA 11.4 Based Builds with NVIDIA OptiX Support

*This is the preferred build setup on CHPC*

To build with GCC 8.5.0, CUDA 11.4, and OptiX 7.1.0 on CHPC.
Please use the following modules:
```
module load cuda/11.4
module load cmake/3.21.4
module load gcc/8.5.0
module load boost/1.77.0
module load intel-oneapi-mpi/2021.4.0
module load gdal/3.3.3
module load netcdf-c/4.8.1
module load netcdf-cxx/4.2
```
Or use the provided load script.
```
source CHPC/loadmodules_QES.sh
```
After completing the above module loads, the following modules are reported from `module list`:
```
Currently Loaded Modules:
  1) cuda/11.4    (g)   3) gcc/8.5.0      5) intel-oneapi-mpi/2021.4.0   7) netcdf-c/4.8.1
  2) cmake/3.21.4       4) boost/1.77.0   6) gdal/3.3.3                  8) netcdf-cxx/4.2
```
After the modules are loaded, you can create the Makefiles with cmake.  We keep our builds separate from the source and contain our builds within their own folders.  For example, 
```
mkdir build
cd build
cmake -DCUDA_TOOLKIT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/11.4.0 -DCUDA_SDK_ROOT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/11.4.0 -DOptiX_INSTALL_DIR=/uufs/chpc.utah.edu/sys/installdir/optix/7.1.0 -DCMAKE_C_COMPILER=gcc -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```
Upon completion of the above commands, you can go about editing and building mostly as normal, and issue the `make` command in your build folder to compile the source.

After you've created the Makefiles with the cmake commands above, the code can be compiled on CHPC:
```
make
```
Note you *may* need to type make a second time due to a build bug, especially on the CUDA 8.0 build.

### Build Types

The code support several build types: *Debug*, *Release*, *RelWithDebInfo*, *MinSizeRel*. You can select the build type 
```
cmake -DCMAKE_BUILD_TYPE=Release ..
```
- *Release* is recommanded for production

## Running QES

To run QES-Winds, you can take the following slurm template and run on CHPC.  We'd suggest placing it in a ```run``` folder at the same level as your build folder.  Make sure you change the various sbatch parameters as needed for your access to CHPC.

### Running from the Command Line

QES is run from the terminal using arguments. For exmaple:
```
./qesWinds/qesWinds -q ../data/InputFiles/GaussianHill.xml -s 2 -w -o gaussianHill
```
More info about the arguments supported by QES can be display using:
```
./qesWinds/qesWinds -?
```

### slurm Template (for CUDA 11.4 build)
```
#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=qesGaussian
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.5.0
ulimit -c unlimited -s
./qesWinds/qesWinds -q ../data/InputFiles/GaussianHill.xml -s 2 -w -o gaussianHill
```

Note that if you build with a different GCC (e.g. 5.4.0), you will need to change the module load to use that version of GCC. Once the slurm file has been placed in the run folder, you can then send out the job.  For example, assuming you are in the build folder and just built the code and we saved the slurm template above as a file rGaussianHill_gpu.slurm:

```
make clean
make
cd ../run
sbatch rGaussianHill_gpu.slurm
```

This will create the various NetCDF output files in the run folder, along with any output in the init_error.log and init_out.log files.


## Testing

We are using ctest to conduct unit tests and sanity check on the code. Here are a few commands:
```
ctest			# launch all tests
ctest --verbose		# launch all tests with verbose (see commant output)
ctest -N		# get list of tests
ctest -R $testname	# launch only $testname
```
Here is a list of tests and testing option. Most test require manuel inspection of the results. Recursive testing will be implemented in the future. 

### QES-Winds Tests

Test for QES-Winds are designed to check that to code is still running under a given set of parameters. These tests do not guarentee the validity of the results. To turn on the basic QES-wind test, use:
```
cmake -DENABLE_SANITY_TESTS=ON -DENABLE_GPU_TESTS=ON ..
```
The QES-Winds sanity tests are: 
- GPU_FlatTerrain: basic empty domain test
- GPU_GaussianHill: basic terrain test
- GPU_OklahomaCity: coarse resolution shapefile reader (without parameterization)
- GPU_MultiSensors: test of multiple sensor and multiple timesteps
- GPU_SaltLakeCity: test of high resolution urban setup with parameterizations
- GPU_RxCADRE: test of high resolution and complex terrain (DEM)

### QES-Turb Tests

There currently is no automated test available for QES-Turb. 

### QES-Plume Tests

There currently is no automated test available for QES-Plume. The following test cases are available
- testing well-mixed condition: Sinusoidal3D Channel3D BaileyLES
- testing against analitical soluation: UniformFlow_ContRelease PowerLawBLFlow_ContRelease
- testing against wind-tunnel data: EPA_7x11array  
         
### Unit Tests
Unit tests can be enable by settong the flag `ENABLE_UNITTESTS` to `ON`. 
```
cmake -DENABLE_UNITTESTS=ON ..
```

## Tips and Tricks

In case things don't go as planned with these instructions, here are some tips for correcting some build or run issues:

## Building the Documentation via Doxygen

After the build is configured the Doxygen documentation can be built. The output from this process is the updating of the _html_ and _latex_ folders in the top-level _docs_ folders.

```
make windsdoc
```

## Continuous Integration

We were running continuous integration on Travis-CI but this is no longer functional...

[Basic Concepts for Travis Continuous Integration](https://docs.travis-ci.com/user/for-beginners/)

## Published QES Papers

1. B. Bozorgmehr et al., “Utilizing dynamic parallelism in CUDA to accelerate a 3D red-black successive over relaxation wind-field solver,” *Environ Modell Softw*, vol. 137, p. 104958, 2021, doi: [10.1016/j.envsoft.2021.104958](https://doi.org/10.1016/j.envsoft.2021.104958).

2. F. Margairaz et al., “Development and evaluation of an isolated-tree flow model for neutral-stability conditions,” *Urban Clim*, vol. 42, p. 101083, 2022, doi: [10.1016/j.uclim.2022.101083](https://doi.org/10.1016/j.uclim.2022.101083).

3. M. J. Moody et al., “QES-Fire: a dynamically coupled fast-response wildfire model,” *Int J Wildland Fire*, vol. 31, no. 3, pp. 306–325, 2022, doi: [10.1071/wf21057](https://doi.org/https://doi.org/10.1071/WF21057).


