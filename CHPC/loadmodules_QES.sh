#!/bin/bash

echo '--------------------------------------------------------------------------------'
echo 'Load module for QES on UTAH CHPC cluster'
module --force purge
module load cuda/11.8
module load cmake/3.21.4
module load gcc/11.2.0
module load boost/1.83.0
module load gdal/3.8.5
module load netcdf-c/4.9.2
module load netcdf-cxx/4.2
module list
echo '--------------------------------------------------------------------------------'
