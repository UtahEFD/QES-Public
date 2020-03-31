#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_twoBuilding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --exclude=notch055
#SBATCH -e plume_error_twoBuilding.log
#SBATCH -o plume_slurm_twoBuilding.log

module load gcc/5.4.0
ulimit -c unlimited -s


./cudaPlume/cudaPlume -q ../testCases/twoBuilding/twoCellGap_stairStep/a_plumeRunScripts/twoBuilding_twoCellGap_stairStep.xml -u ../testCases/twoBuilding/twoCellGap_stairStep/b_plumeInputs/twoBuilding_twoCellGap_stairStep_urb.nc -t ../testCases/twoBuilding/twoCellGap_stairStep/b_plumeInputs/twoBuilding_twoCellGap_stairStep_turb.nc -o ../testCases/twoBuilding/twoCellGap_stairStep/c_plumeOutputs/ -b twoBuilding_twoCellGap_stairStep -e -l -s -d

