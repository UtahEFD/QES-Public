#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=QES-plume-Sinusoidal3D
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH -o slurm_Sinusoidal3D-%j-%N.log

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

./qesPlume/qesPlume -q ../testCases/Sinusoidal3D/QES-files/Sinusoidal3D_4_12.xml -u ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_windsWk.nc -t ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_turbOut.nc -o ../testCases/Sinusoidal3D/QES-data/ -b Sinusoidal3D_4_12 -e -l

./qesPlume/qesPlume -q ../testCases/Sinusoidal3D/QES-files/Sinusoidal3D_0.1_12.xml -u ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_windsWk.nc -t ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_turbOut.nc -o ../testCases/Sinusoidal3D/QES-data/ -b Sinusoidal3D_0.1_12 -e -l

./qesPlume/qesPlume -q ../testCases/Sinusoidal3D/QES-files/Sinusoidal3D_0.05_12.xml -u ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_windsWk.nc -t ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_turbOut.nc -o ../testCases/Sinusoidal3D/QES-data/ -b Sinusoidal3D_0.05_12 -e -l

./qesPlume/qesPlume -q ../testCases/Sinusoidal3D/QES-files/Sinusoidal3D_0.01_12.xml -u ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_windsWk.nc -t ../testCases/Sinusoidal3D/QES-data/Sinusoidal3D_turbOut.nc -o ../testCases/Sinusoidal3D/QES-data/ -b Sinusoidal3D_0.01_12 -e -l

echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
