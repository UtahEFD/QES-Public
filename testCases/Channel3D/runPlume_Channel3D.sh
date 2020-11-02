#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=QES-plume-Channel3D
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH -o slurm_Channel3D-%N.log

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

./cudaPlume/cudaPlume -q ../testCases/Channel3D/QES-files/Channel3D_1.83_18.3.xml -u ../testCases/Channel3D/QES-data/Channel3D_windsWk.nc -t ../testCases/Channel3D/QES-data/Channel3D_turbOut.nc -o ../testCases/Channel3D/QES-data/ -b Channel3D_1.183_18.3 -e -l

./cudaPlume/cudaPlume -q ../testCases/Channel3D/QES-files/Channel3D_0.183_18.3.xml -u ../testCases/Channel3D/QES-data/Channel3D_windsWk.nc -t ../testCases/Channel3D/QES-data/Channel3D_turbOut.nc -o ../testCases/Channel3D/QES-data/ -b Channel3D_0.183_18.3 -e -l

./cudaPlume/cudaPlume -q ../testCases/Channel3D/QES-files/Channel3D_0.00183_18.3.xml -u ../testCases/Channel3D/QES-data/Channel3D_windsWk.nc -t ../testCases/Channel3D/QES-data/Channel3D_turbOut.nc -o ../testCases/Channel3D/QES-data/ -b Channel3D_0.00183_18.3 -e -l 

echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
