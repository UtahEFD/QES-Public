#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_oneBuilding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --exclude=notch055
#SBATCH -e plume_error_oneBuilding.log
#SBATCH -o plume_slurm_oneBuilding.log

module load gcc/5.4.0
ulimit -c unlimited -s


./cudaPlume/cudaPlume -q ../testCases/oneBuilding/stairStep_coarse/a_plumeRunScripts/oneBuilding_stairStep_coarse.xml -u ../testCases/oneBuilding/stairStep_coarse/b_plumeInputs/oneBuilding_stairStep_coarse_urb.nc -t ../testCases/oneBuilding/stairStep_coarse/b_plumeInputs/oneBuilding_stairStep_coarse_turb.nc -o ../testCases/oneBuilding/stairStep_coarse/c_plumeOutputs/ -b oneBuilding_stairStep_coarse -e -l -s -d

