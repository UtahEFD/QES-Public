#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=QES-plume-ContRelease_PowerLawBLFlow
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH -o slurm_%x-%j-%N.log

### UNCOMMENT FOR NODE SELECTION 
##SBATCH --nodelist=notch055
##SBATCH --exclude=notch055


# Prologue

echo '****** PROLOGUE ******'
echo '----------------------------------------------------------------------------'
echo 'Hi, jobID: '$SLURM_JOBID
date
sacct -j $SLURM_JOBID
echo '----------------------------------------------------------------------------'
echo 'setting environment'
module load cuda/11.4
module load cmake/3.21.4
module load gcc/8.5.0
module load boost/1.77.0
module load intel-oneapi-mpi/2021.4.0
module load gdal/3.3.3
module load netcdf-c/4.8.1
module load netcdf-cxx/4.2
module load matlab
ulimit -c unlimited -s
module list
echo '****** START OF JOB ******'

cd MATLAB

matlab -nodisplay -nosplash -nodesktop -r "run('PowerLawBLFlow_xDir_inputFiles.m'); run('PowerLawBLFlow_yDir_inputFiles.m'); exit;"

cd ../../../build

./qesPlume/qesPlume -q ../testCases/PowerLawBLFlow_ContRelease/QES-files/PowerLawBLFlow_xDir_ContRelease.xml -w ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_xDir_windsWk.nc -t ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_xDir_turbOut.nc -o ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_xDir_ContRelease

./qesPlume/qesPlume -q ../testCases/PowerLawBLFlow_ContRelease/QES-files/PowerLawBLFlow_yDir_ContRelease.xml -w ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_yDir_windsWk.nc -t ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_yDir_turbOut.nc -o ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_yDir_ContRelease

cd -

matlab -nodisplay -nosplash -nodesktop -r "run('PowerLawBLFlow_xDir_mainPlumeResults.m'); clear; run('PowerLawBLFlow_yDir_mainPlumeResults.m'); exit;"

cd ..

echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
