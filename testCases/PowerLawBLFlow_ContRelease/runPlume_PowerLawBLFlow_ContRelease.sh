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
module load cuda/10.2
module load gcc/8.1.0
module load cmake/3.15.3
module load gdal/3.0.1
module load boost/1.69.0
module load netcdf-cxx
ulimit -c unlimited -s
module list
echo '****** START OF JOB ******'

./qesPlume/qesPlume -q ../testCases/PowerLawBLFlow_ContRelease/QES-files/PowerLawBLFlow_xDir_ContRelease.xml -u ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_xDir_windsWk.nc -t ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_xDir_turbOut.nc -o ../testCases/PowerLawBLFlow_ContRelease/QES-data/ -b ContRelease_xDir -l

./qesPlume/qesPlume -q ../testCases/PowerLawBLFlow_ContRelease/QES-files/PowerLawBLFlow_yDir_ContRelease.xml -u ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_yDir_windsWk.nc -t ../testCases/PowerLawBLFlow_ContRelease/QES-data/PowerLawBLFlow_yDir_turbOut.nc -o ../testCases/PowerLawBLFlow_ContRelease/QES-data/ -b ContRelease_yDir -l


echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
