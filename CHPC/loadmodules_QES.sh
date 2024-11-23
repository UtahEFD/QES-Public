#!/bin/bash

echo '--------------------------------------------------------------------------------'
echo 'Load module for QES on UTAH CHPC cluster'
#module --force purge
module load cuda/11.4
module load cmake/3.21.4
module load gcc/8.5.0
module load boost/1.77.0
module load intel-oneapi-mpi/2021.4.0
module load gdal/3.3.3
module load netcdf-c/4.8.1
module load netcdf-cxx/4.2
module list
echo '--------------------------------------------------------------------------------'
