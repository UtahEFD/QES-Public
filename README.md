GPU QUIC
--------

This code contains the GPU versions of Urb and Plume that were started
by Andrew Larson (urb), Alex (Gai) Geng (plume), Balwinder Singh
(plume), Pete Willemsen (urb, plume) and Eric Pardyjak (urb,
plume). These versions of the code are done in CUDA.

## GPU-Plume (3D GLE Model)

This plume model uses Balwinder Singh's 3D GLE model from his
dissertation. This code currently relies on CUDA 8.0. The code
requires a recent Linux distribution (Ubuntu 16.04) and a recent
NVIDIA graphics card and somewhat recent NVIDIA Linux drivers.

### Building the Source

To compile plume, first

  mkdir build
  cd build

Note that if you installed CUDA or Boost in non-standard places, you
will need to run cmake interactively to manually type in the locations
of the Boost and CUDA libraries. To run cmake interactively, you use
the following cmake command:

   cmake .. -i

Alternatively, if you know where you installed CUDA and libsivelab,
you can run cmake with command line options that will set up the
project correctly. For instance:

  cmake .. -DCUDA_TOOLKIT_ROOT_DIR=/home/cuda_8.0 -DCUDA_SDK_ROOT_DIR=/home/cuda_8.0

Once cmake has been configured, the GPU plume code can be compiled.

  make



The process is a bit tricker for compiling the code on chpc. On kingspeak where
boost can be loaded in separate from the boost libraries, and where the default gcc
is 4.8.5 (you probably need gcc 5.4.0 or higher), do the following before running
cmake from the clean build directory:

	module load cuda/8.0
	module load boost
	module load gcc/5.4.0

now the cmake command needs to be something like the following:

	cmake ../CUDA-Plume -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOTDIR -DCUDA_SDK_ROOT_DIR=$CUDA_ROOTDIR -DCUDA_SDK_ROOT_DIR=$CUDA_ROOTDIR -DBOOST_ROOT=$BOOST_DIR

Then can run make the same as normal and it should work. Note that notchpeak
may not have boost libraries that can be loaded in with module load.


