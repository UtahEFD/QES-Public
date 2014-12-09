GPU QUIC
--------

This code contains the GPU versions of Urb and Plume that were started by Andrew Larson (urb) and Alex (Gai) Geng and Pete Willemsen (plume). These versions of the code are done in CUDA.


GPU-Plume (3D GLE Model)

To compile the 3D GLE model of gpuPlume, you will need cmake version
2.8 or greater, Boost, and the NVIDIA CUDA Libraries installed. As of
this writing the code has been tested with CUDA 4.2 only.

You will also need the libsivelab library. It can be checked out with
the following command:

svn co https://wind.d.umn.edu/svn/libsivelab/trunk libsivelab

This will create a directory called libsivelab. You will need to go
into that directory and build the libsivelab source. To do this,
follow these steps:

cd libsivelab
mkdir build
cd build
cmake ..
make
cd ..

At this point, the libsivelab source should have compiled. Note that
if you installed CUDA or Boost in non-standard places, you will need
to run cmake interactively to manually type in the locations of the
Boost and CUDA libraries.  To run cmake interactively, you use the
following cmake command:

   cmake .. -i

To compile gpuPlume, go back to the plume directory. From within the
plume directory, you will notice a CMakeLists.txt file. This is the
main file that CMake uses to create the Makefiles that will build this
project. To run cmake and compile the plume code, do the following
from the same level as the CMakeLists.txt file in the plume directory:

mkdir build
cd build
cmake .. -DLIBSIVELAB_PATH=<YourPathToWhereThelibsivelabBuildDirectoryExists>
make

To run the hard-coded test (I think its Balwinder's test case), just
type

./plume

Buildings are not drawn in this test, but they do exist. Once the
window pops up, just press 'b' to show the buildings.

