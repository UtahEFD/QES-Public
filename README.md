# QES-Plume

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

## Instructions for CHPC Cluster

The code does run on the CHPC cluster. You need to make sure the correct set of modules are loaded.  
Currently, we have tested a few configurations that use CUDA 8.0 and GCC 5.4 and another that uses CUDA 10.2 and GCC 8.1.0. 
For QES-Plume, OptiX support is not required for QES-Plume.

### CUDA 10.2 Based Builds

To build with CUDA 10.1 without any OptiX GPU acceleration, please use the following modules:

```
module load cuda/10.2
module load gcc/8.1.0
module load cmake/3.11.2 
module load gdal/3.0.1
module load boost/1.69.0
module load netcdf-cxx
```

After completing the above module loads, the following modules are reported from `module list`:

```
Currently Loaded Modules:
  1) chpc/1.0     (S)   3) gdal/3.0.1        5) netcdf-c/4.4.1     7) cuda/10.2 (g)   9) boost/1.69.0
  2) cmake/3.11.2       4) hdf5/1.8.17 (H)   6) netcdf-cxx/4.3.0   8) gcc/8.1.0
```

After the modules are loaded, you can create the Makefiles with cmake.  We keep our builds separate from the source and contain our builds within their own folders.  For example, 

```
mkdir build
cd build
cmake -DCUDA_TOOLKIT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.2.89 -DCUDA_SDK_ROOT_DIR=/uufs/chpc.utah.edu/sys/installdir/cuda/10.2.89 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/3.0.1 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```

### CUDA 8.0

To build with CUDA 8.0, please use the following modules:
```
module load cuda/8.0
module load gcc/5.4.0
module load cmake/3.11.2 
module load gdal/2.3.1
module load boost/1.66.0
module load netcdf-cxx
```

```
module list

Currently Loaded Modules:
  1) chpc/1.0   2) cuda/8.0 (g)   3) gcc/5.4.0   4) cmake/3.11.2   5) gdal/2.3.1   6) hdf5/1.8.17   7) netcdf-c/4.4.1   8) netcdf-cxx/4.3.0
```
After the modules are loaded, you can create the Makefiles with cmake. We keep our builds separate from the source and contain our builds within their own folders. For example,
```
mkdir build
cd build
cmake -DCUDA_TOOLKIT_DIR=/usr/local/cuda-8.0 -DCUDA_SDK_ROOT_DIR=/usr/local/cuda-8.0 -DCMAKE_PREFIX_PATH=/uufs/chpc.utah.edu/sys/installdir/gdal/2.1.3-c7 -DNETCDF_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-c/4.4.1-c7/include -DNETCDF_CXX_DIR=/uufs/chpc.utah.edu/sys/installdir/netcdf-cxx/4.3.0-5.4.0g/include ..
```

### Compiling the Code 

After you've created the Makefiles with the cmake commands above, the code can be compiled on CHPC:
```
make
```
Note you may need to type make a second time due to a build bug, especially on the CUDA 8.0 build.


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


\
\
An extra set of plot tools has just been started in the /util/MATLAB/c_CUDA-PlumePlotting/a_runscripts/c_particlePlots directory. The idea is that these scripts can be a starting point for visualizing lagrangian particle data. Once the plotting methods get more generalized, they should be wrapped up into functions, but for now the idea is to do whatever you need to do to load in data from a simulation, and start plotting particle information. This example also shows how to create videos of the plot output in matlab.



### Useful chpc information

Normally you run scripts by finding the created script executable (in this case plume) and running:
```
./plume
```
But on chpc, you need to create a proper batch script (in this case runscript.sh), then call the batch script with:
```
sbatch runscript.sh
```
You can then query the run by calling one of these two lines of code:

```
squeue -u uID
or
squeue -A account
```
You can then cancel the run early or let it run to completion. Once it starts to run, it produces a slurm file with the results of the run. To cancel the run, use squeue to find the processID, then call:

```
scancel processID
```

\
\
Modifying the runscript file to get the right account, partition, and qos can be tricky. Use the following command to see what is available to you:
```
sacctmgr -p show assoc user=uID
```
You can paste the result in excel and use Data/text-to-column with the "|" char to get it easy to read. It doesn't list the partition, but qos is the same thing as the partition. You can find more information about all available accounts and partitions at: https://www.chpc.utah.edu/documentation/policies/2.1GeneralHPCClusterPolicies.php


Getting the right type of gpu is also tricky. Basically different accounts have access to different resources. It's kind of trial and error looking up information, or trying things out, but this website at least tells you some of the possible combinations: https://www.chpc.utah.edu/documentation/guides/gpus-accelerators.php


The example runscript files may or may not work for you, but they are the starting spot to figure out combinations that should work for you. If they have kp in the name, they are for kingspeak. If np, they are for notchpeak. If they have cpu, they are for cpu only computations. If they have gpu in the name, they are for a mix of cpu and gpu computations. Note even if the code has been pre-built, you will need to repeat the module loads as if compiling the script the first time you run the code from a new login session.


\
\
It can be tricky to transfer files back and forth with chpc. To access kingspeak or notchpeak, do one of these two lines of code, putting in your uID:
```
ssh uID@kingspeak.chpc.utah.edu
or
ssh uID@notchpeak.chpc.utah.edu
```
After you figure out where you want to send files on the chpc server, and you are in the directory of the folder you want to send (in this case one folder above CUDA-Plume), do the following:
```
scp -r ./CUDA-Plume/ uID@kingspeak.chpc.utah.edu:CUDA-Plume/
```






