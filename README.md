# QES-Winds

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

The code does run on the CHPC cluster. You need to make sure the
correct set of modules are loaded.  Currently, we have tested a few 
configurations that use CUDA 8.0 and GCC 5.4 and another that
uses CUDA 10.1 or 10.2 and GCC 8.1.0. If you build with OptiX support, you will need to use the CUDA 10.2 configuration. Any builds (with or without OptiX) with CUDA 10.2 are preferred if you don't know which to use.

After logging into your CHPC account, you will need to load specific
modules. In the following sections, we outline the modules that need to be loaded along with the various cmake command-line calls that specify the exact locations of module installs on the CHPC system.  

### CUDA 10.2 Based Builds with NVIDIA OptiX Support

*This is the preferred build setup on CHPC*

To build with CUDA 10.2 and OptiX 7.1.0 on CHPC, please use the following

```
module load cuda/10.2
module load gcc/8.1.0
module load cmake/3.15.3
module load gdal/3.0.1
module load boost/1.69.0
```

After completing the above module loads, the following modules are reported from `module list`:

```
Currently Loaded Modules:
  1) chpc/1.0 (S)   2) cuda/10.2 (g)   3) gcc/8.1.0   4) cmake/3.15.3   5) gdal/3.0.1   6) boost/1.69.0
```

After the modules are loaded, you can create the Makefiles with cmake.  We keep our builds separate from the source and contain our builds within their own folders.  For example, 

```
mkdir build
cd build

cmake -DCUDA_TOOLKIT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.2.89 -DCUDA_SDK_ROOT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.2.89 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include -DOptiX_INSTALL_DIR=/uufs/chpc.utah.edu/sys/installdir/optix/7.1.0/ -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH="/uufs/chpc.utah.edu/sys/installdir/gdal/3.0.1;/uufs/chpc.utah.edu/sys/installdir/hdf5/1.8.17-c7" ..
```

Upon completion of the above commands, you can go about editing and building mostly as normal, and issue the ```make``` command in your build folder to compile the source.   The instructions for other configurations are very similar with the primary exception being the modules that can be loaded.


### CUDA 10.2 Based Builds

To build with CUDA 10.1 without any OptiX GPU acceleration, please use the following modules:

```
module load cuda/10.2
module load gcc/8.1.0
module load cmake/3.11.2 
module load gdal/3.0.1
module load boost/1.69.0
```

After completing the above module loads, the following modules are reported from `module list`:

```
Currently Loaded Modules:
  1) chpc/1.0 (S)   2) cuda/10.2 (g)   3) gcc/8.1.0   4) cmake/3.15.3   5) gdal/3.0.1   6) boost/1.69.0
```

After the modules are loaded, you can create the Makefiles with cmake.  We keep our builds separate from the source and contain our builds within their own folders.  For example, 

```
mkdir build
cd build

cmake -DCUDA_TOOLKIT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.2.89 -DCUDA_SDK_ROOT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.2.89 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include -DOptiX_INSTALL_DIR=/uufs/chpc.utah.edu/sys/installdir/optix/7.1.0/ -DCMAKE_C_COMPILER=gcc -DCMAKE_PREFIX_PATH="/uufs/chpc.utah.edu/sys/installdir/gdal/3.0.1;/uufs/chpc.utah.edu/sys/installdir/hdf5/1.8.17-c7" ..
```

### CUDA 10.1 Based Builds

Using CUDA 10.1:

```
module load cuda/10.1
module load gcc/8.1.0
module load cmake/3.11.2 
module load gdal/2.4.0
module load boost/1.69.0
ml netcdf-cxx
```

After completing the above module loads, the following modules are reported from `module list`:

```
Currently Loaded Modules:
  1) chpc/1.0     (S)   3) gdal/2.4.0        5) netcdf-c/4.4.1     7) cuda/10.1 (g)   9) boost/1.69.0
  2) cmake/3.11.2       4) hdf5/1.8.17 (H)   6) netcdf-cxx/4.3.0   8) gcc/8.1.0
```

To construct the Makefiles, you can run cmake in this way from your build directory.  For example, 
```
mkdir build
cd build
cmake -DCUDA_TOOLKIT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.1.168 -DCUDA_SDK_ROOT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.1.168 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.4.0 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```

### CUDA 9.1

For CUDA 9.1, 

```
module load cuda/9.1
module load gcc/5.4.0
module load cmake/3.11.2
module load gdal/2.4.0
module load boost/1.68.0
```

```
cmake -DCUDA_TOOLKIT_DIR=/usr/local/cuda-9.1 -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-9.1 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.4.0 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```


### CUDA 8.0

And, CUDA 8.0:

```
module load cuda/8.0
module load gcc/5.4.0
module load cmake/3.11.2 
module load gdal/2.3.1
module load boost/1.66.0
ml netcdf-cxx
```

```
module list

Currently Loaded Modules:
  1) chpc/1.0   2) cuda/8.0 (g)   3) gcc/5.4.0   4) cmake/3.11.2   5) gdal/2.3.1   6) hdf5/1.8.17   7) netcdf-c/4.4.1   8) netcdf-cxx/4.3.0
```

```
cmake -DCUDA_TOOLKIT_DIR=/usr/local/cuda-8.0 -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.1.3-c7 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```

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

