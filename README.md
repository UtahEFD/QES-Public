<div align="center">

# QES: Quick Environmental Simulations

<!-- Badges -->
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7098280.svg)](https://doi.org/10.5281/zenodo.7098280)
<!-- Badges -->

</div>

## QES-Winds

QES-Winds is a fast response 3D diagnostic urban wind model written in
C++ and uses NVIDIA's CUDA framework to accelerate a mass-conserving
wind-field solver. QES-Winds uses a variational analysis technique to
ensure the conservation of mass rather than slower yet more
physics-based solvers that include conservation of momentum. QES-Winds
minimizes the difference between an initial wind field that is
specified using empirical parameterizations and thefinal wind field.
This method requires the solution of a Poisson equation for Lagrange
multipliers. The Poisson equation is solved using the Successive
Over-Relaxation (SOR) method (an iterative solver), which is a variant
of the Gauss-Seidel method with more rapid convergence. QES-Winds
utilizes the concept of dynamic parallelism in NVIDIAs parallel
computing-based Graphics Processing Unit (or GPU) API, CUDA, to
substantially accelerate wind simulations.

## QES-Turb

## QES-Plume

## QES-Fire


## Package Requirements

On a general Linux system, such as Ubuntu 18.04 or 20.04 which we commonly use, you will need the following packages installed:
* libgdal-dev
* libnetcdf-c++4-dev
* libnetcdf-cxx-legacy-dev
* libnetcdf-dev
* netcdf-bin
* libboost-all-dev
* cmake
* cmake-curses-gui

If you have a system that uses apt, here's the command:
```apt install libgdal-dev libnetcdf-c++4-dev  libnetcdf-cxx-legacy-dev libnetcdf-dev netcdf-bin libboost-all-dev cmake cmake-curses-gui```

To use the GPU system (and even build the code) you will need a NVIDIA
GPU with the CUDA library installed.  We have tested with CUDA 8.0, 10.0, 10.1, and 10.2. 
If your version of CUDA is installed in a non-uniform location, you
will need to remember the path to the cuda install directory.

Additionally, the code can use NVIDIA's OptiX to accelerate various computations. Our OptiX code has been built to use version 7.0 or higher.

## Building the Code

The most active development occurs in the *workingBranch*. We suggest you use that branch at this time.  You can checkout this branch with the following git command:

```
git checkout workingBranch
```

If you are unsure about which branch you are on, the ``` git status ``` command can provide you with this information.


### Building on general Linux system

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


## Instructions for CHPC Cluster

The code does run on the CHPC cluster. You need to make sure the correct set of modules are loaded.  Currently, we have tested a few configurations that use
- GCC 5.4.0 and CUDA 8.0
- CCC 8.1.0 and CUDA 10.1 (10.2)
- GCC 8.5.0 and CUDA 11.4
If you build with OptiX support, you will need to use CUDA 10.2 or newer configuration. Any builds (with or without OptiX) with CUDA 10.2 are preferred if you don't know which to use. Older configurations are provided in `CHPC/oldBuilds.md`.

After logging into your CHPC account, you will need to load specific modules. In the following sections, we outline the modules that need to be loaded along with the various cmake command-line calls that specify the exact locations of module installs on the CHPC system.  

### CUDA 11.4 Based Builds with NVIDIA OptiX Support

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

## Compiling the Code and Running on CHPC

After you've created the Makefiles with the cmake commands above, the code can be compiled on CHPC:

```
make
```
Note you *may* need to type make a second time due to a build bug, especially on the CUDA 8.0 build.

To run QES-Winds, you can take the following slurm template and run on CHPC.  We'd suggest placing it in a ```run``` folder at the same level as your build folder.  Make sure you change the various sbatch parameters as needed for your access to CHPC.

### slurm Template (for CUDA 10.1 build)
```
#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=moser395
#SBATCH --nodes=1
#SBATCH --mem=15G
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.1.0
ulimit -c unlimited -s
./qesWinds/qesWinds -q ../data/InputFiles/GaussianHill.xml -s 2 -w -o gaussianHill
```

Note that if you build with a different GCC (i.e. 5.4.0), you will need to change the module load to use that version of GCC. Once the slurm file has been placed in the run folder, you can then send out the job.  For example, assuming you are in the build folder and just built the code and we saved the slurm template above as a file rGaussianHill_gpu.slurm:

```
make clean
make
cd ../run
sbatch rGaussianHill_gpu.slurm
```

This will create the various NetCDF output files in the run folder, along with any output in the init_error.log and init_out.log files.


## Tips and Tricks

In case things don't go as planned with these instructions, here are some tips for correcting some build or run issues:

## Building the Documentation via Doxygen

After the build is configured the Doxygen documentation can be built. The output from this process is the updating of the _html_ and _latex_ folders in the top-level _docs_ folders.

```
make windsdoc
```


### Continuous Integration

We were running continuous integration on Travis-CI but this is no longer functional...

[Basic Concepts for Travis Continuous Integration](https://docs.travis-ci.com/user/for-beginners/)


## Testing

We are using ctest to conduct unit tests and sanity check on the code. Here are a few commands:
```
ctest			# launch all tests
ctest --verbose		# launch all tests with verbose (see commant output)
ctest -N		# get list of tests
ctest -R $testname	# launch only $testname
```
List of tests and testing option will be added here.
