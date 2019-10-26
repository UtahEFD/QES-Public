GPU QUIC
--------

This code contains the GPU versions of Urb and Plume that were started
by Andrew Larson (urb), Alex (Gai) Geng (plume), Balwinder Singh
(plume), Pete Willemsen (urb, plume) and Eric Pardyjak (urb,
plume). These versions of the code are done in CUDA.

The code has been updated by Loren Atwood to match methods given by
Brian Bailey to completely eliminate rogue trajectories.

## GPU-Plume (3D GLE Model)

This plume model uses Balwinder Singh's 3D GLE model from his
dissertation. This code currently relies on CUDA 8.0. The code
requires a recent Linux distribution (Ubuntu 16.04) and a recent
NVIDIA graphics card and somewhat recent NVIDIA Linux drivers.

The model is still the 3D GLE, but now the method of time integration is a lot simpler.
Follows Brian Bailey's Lagrangian Implicit method for eliminating rogue trajectories.

### Building the Source

To compile plume, first
```
  mkdir build_plume
  cd build_plume
```
Note that if you installed CUDA or Boost in non-standard places, you
will need to run cmake interactively to manually type in the locations
of the Boost and CUDA libraries. To run cmake interactively, you use
the following cmake command:
```
   cmake .. -i
```
Alternatively, if you know where you installed CUDA and libsivelab,
you can run cmake with command line options that will set up the
project correctly. For instance:
```
  cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/home/cuda_8.0 -DCUDA_SDK_ROOT_DIR=/home/cuda_8.0
```
Once cmake has been configured, the GPU plume code can be compiled.
```
  make
```
The process is a bit tricker for compiling the code on chpc. On kingspeak where
boost can be loaded in separate from the boost libraries, and where the default gcc
is 4.8.5 (you probably need gcc 5.4.0 or higher), do the following before running
cmake from the clean build directory:
```
	module load cuda/8.0
	module load gcc/5.4.0
	module load cmake/3.11.2 
	module load gdal/2.3.1
	module load boost/1.66.0
	ml netcdf-cxx
```
Where just to keep things safe with overkill, all the expected libraries for when Plume, Urb, and Turb are finished are loaded as well as the current minimum requirements. Now the cmake command needs to be something like the following:
```    
cmake -DCUDA_TOOLKIT_DIR=/usr/local/cuda-8.0 -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.1.3-c7 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```
or
```    
	cmake ../CUDA-Plume -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOTDIR -DCUDA_SDK_ROOT_DIR=$CUDA_ROOTDIR -DCUDA_SDK_ROOT_DIR=$CUDA_ROOTDIR -DBOOST_ROOT=$BOOST_DIR
```
Then can run make the same as normal and it should work. Note that notchpeak
may not have boost libraries that can be loaded in with module load.

Note that the first of the cmake commands seems to be more stable right now.

