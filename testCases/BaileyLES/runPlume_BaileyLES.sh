#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=QES-plume-BaileyLES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH -o slurm_BaileyLES-%N.log

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

./cudaPlume/cudaPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_22.2_222.xml -u ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/ -b BaileyLES_22.2_222 -e -l

./cudaPlume/cudaPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_2.22_222.xml -u ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/ -b BaileyLES_2.22_222 -e -l;

./cudaPlume/cudaPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_0.222_222.xml -u ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/ -b BaileyLES_0.222_222 -e -l;

./cudaPlume/cudaPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_0.0222_222.xml -u ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/ -b BaileyLES_0.0222_222 -e -l;

echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
