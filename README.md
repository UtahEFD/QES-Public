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

To compile plume so the test cases run easily, make sure you are in the top directory, then create a new build directory
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


The process is a bit tricker for compiling the code on chpc. On notchpeak and kingspeak, do the following before running
cmake from inside the clean build directory:
```
module load cuda/8.0
module load gcc/5.4.0
module load cmake/3.11.2 
module load gdal/2.3.1
module load boost/1.66.0
ml netcdf-cxx

```
Not all these modules are used as of yet, but they will be once Urb, Turb, and Plume get close to completion. The cmake command needs to be something like the following:
```
cmake -DCUDA_TOOLKIT_DIR=/usr/local/cuda-8.0 -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.1.3-c7 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```
or
```
cmake ../CUDA-Plume -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOTDIR -DCUDA_SDK_ROOT_DIR=$CUDA_ROOTDIR -DCUDA_SDK_ROOT_DIR=$CUDA_ROOTDIR -DBOOST_ROOT=$BOOST_DIR
```
Then run make the same as normal and it should work
```
make
```

Note that the first of the cmake commands seems to be more stable right now.


### Running the test cases on chpc

After building the code from source, enter the build directory. Then run sbatch on the batch file of the desired test case. For example, for the FlatTerrain single point test case do the following:
```
cd build_plume
sbatch ../testCases/FlatTerrain/a_plumeRunScripts/runPlume_FlatTerrain_singlePoint.sh
squeue -u <uID>
```

Note that this will not work unless you copy the required urb and turb netcdf files to the testCases/FlatTerrain/b_plumeInputs directory. These files are generated output files from CUDA-URB and CUDA-Turb.


The testCases directory holds each of the current test cases. The top level directory for each test case gives the test case name and contains three main folders: a_plumeRunScripts, b_plumeInputs, and c_plumeOutputs. The a_plumeRunScripts folder contains the .xml and .sh batch scripts for each currently available variation of the test case. The b_plumeInputs folder contains common urb and turb netcdf files required for each variation of the test case. The c_plumeOutputs folder is where CUDA-Plume will place outputs for each available variation of the test cases.

The Bailey test cases are a bit different from the other test cases, in that there are additional folders d_matlabPlotRunScripts and e_matlabPlotOutput in each Bailey test case. The idea is that you can run sbatch on the batch scripts found in the d_matlabPlotRunScripts folder after generating all the data for each variation of the test case in the c_plumeOutputs folder. The batch scripts in the d_matlabPlotRunScripts folder use the CUDA-PlumePlotting MATLAB utility functions found in the /util/MATLAB/c_CUDA-PlumePlotting directory to generate the plots from Bailey's 2017 Rogue Trajectory elimination paper. The plots are output to the e_matlabPlotOutput folder for each Bailey test case.

If you wish to make your own test cases, study each of the files for each of the current test cases, paying particular attention to how the paths are setup.

It may also be a good idea to eventually add visitPlotRunScripts and visitPlotOutput folders to each of the test cases if utility scripts to automate some of the visit plotting are ever added to the util directory.


Be aware that while the .xml and .sh batch files are tracked for each test case, the urb and turb netcdf files are only tracked for the Bailey test cases. You will need to generate your own urb and turb files for the other test cases before running CUDA-Plume on those cases.




