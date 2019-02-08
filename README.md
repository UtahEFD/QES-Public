# CUDA-URB

## Configuring the Build

```
mkdir build
cd build
cmake ..
```

You can then build the source:

```
make
```

## Building the Documentation via Doxygen

After the build is configured the Doxygen documentation can be built. The output from this process is the updating of the _html_ and _latex_ folders in the top-level _docs_ folders.

```
make doc
```

## Instructions for CHPC Cluster

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

  Where:
   g:  built for GPU
```

```
cmake -DCUDA_TOOLKIT_DIR=/usr/local/cuda-8.0 -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.1.3-c7 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```

### Continuous Integration

We are running continuous integration on Travis-CI.

[Basic Concepts for Travis Continuous Integration](https://docs.travis-ci.com/user/for-beginners/)


