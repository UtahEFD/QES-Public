#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=QES-plume-UniformFlow_ContRelease
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH -o slurm_%x.log

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

./qesPlume/qesPlume -q ../testCases/UniformFlow_ContRelease/QES-files/UniformFlow_ContRelease.xml -u ../testCases/UniformFlow_ContRelease/QES-data/UniformFlow_windsWk.nc -t ../testCases/UniformFlow_ContRelease/QES-data/UniformFlow_turbOut.nc -o ../testCases/UniformFlow_ContRelease/QES-data/ -b ContRelease -l

echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
