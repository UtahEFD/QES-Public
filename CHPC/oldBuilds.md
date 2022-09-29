## Instructions for CHPC Cluster

This file is provided as reference to older build on CHPC.

NOTE: these commandes where vaild on CentOS7. Now CHPC cluster run on Rocky 8. Most commands are no longer valid!!

### CUDA 10.2 Based Builds with NVIDIA OptiX Support

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