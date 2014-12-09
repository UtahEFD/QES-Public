GPU QUIC
--------

This code contains the GPU versions of Urb and Plume that were started by Andrew Larson (urb) and Alex (Gai) Geng and Pete Willemsen (plume). These versions of the code are done in CUDA.

## GPU-Plume (3D GLE Model)

This plume model uses Balwinder Singh's 3D GLE model from his
dissertation. This code currently relies on CUDA 4.2. Instructions to
compile and run it are designed for a Ubuntu 12.04 system with an
NVIDIA graphics card and somewhat recent NVIDIA Linux drivers.

To compile the 3D GLE model of gpuPlume, you will need cmake version
2.8 or greater, Boost, and the NVIDIA CUDA Libraries installed. As of
this writing the code has been tested with CUDA 4.2 only.

You will also need the libsivelab library. It can be checked out with
the following command:

git clone https://envsim.d.umn.edu/genusis/libsivelab.git

This will create a directory called libsivelab. You will need to go
into that directory and build the libsivelab source. To do this,
follow these steps:

cd libsivelab
mkdir build
cd build
cmake ..
make
cd ..

At this point, the libsivelab source should have compiled.

### Building the Plume Source

To compile plume, first

  cd plume
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

  cmake .. -DLIBSIVELAB_PATH=/home/cs/willemsn/Coding/libsivelab/trunk -DCUDA_TOOLKIT_ROOT_DIR=/home/cs/software/sivelab/cuda_4.2/cuda -DCUDA_SDK_ROOT_DIR=/home/cs/software/sivelab/cuda_4.2/sdk

Once cmake has been configured, the GPU plume code can be compiled.

  make

### Running the Plume test case

To run the hard-coded test (I think its Balwinder's test case), just
type

./plume

Buildings are not drawn in this test, but they do exist. Once the
window pops up, just press 'b' to show the buildings.

