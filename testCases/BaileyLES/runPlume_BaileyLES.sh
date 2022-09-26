#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=QES-plume-BaileyLES
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=5G
#SBATCH --time=05:00:00
#SBATCH -o slurm_BaileyLES-%j-%N.log

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

matlab -nodisplay -nosplash -nodesktop -r "run('BaileyLES_testcase.m');exit;"

cd ../../../build/

./qesPlume/qesPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_22.2_222.xml -w ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/BaileyLES_22.2_222 -l

./qesPlume/qesPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_2.22_222.xml -w ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/BaileyLES_2.22_222 -l;

./qesPlume/qesPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_0.222_222.xml -w ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/BaileyLES_0.222_222 -l;

./qesPlume/qesPlume -q ../testCases/BaileyLES/QES-files/BaileyLES_0.0222_222.xml -w ../testCases/BaileyLES/QES-data/BaileyLES_windsWk.nc -t ../testCases/BaileyLES/QES-data/BaileyLES_turbOut.nc -o ../testCases/BaileyLES/QES-data/BaileyLES_0.0222_222 -l;

cd - 

matlab -nodisplay -nosplash -nodesktop -r "run('plotPlumeResults_BaileyLES.m');exit;"

cd ..

echo '****** END OF JOB ****** '

echo '****** EPILOGUE ******'
echo '----------------------------------------------------------------------------'
sacct -j $SLURM_JOBID
sacct -j $SLURM_JOBID --format=Elapsed,NodeList
echo '----------------------------------------------------------------------------'
