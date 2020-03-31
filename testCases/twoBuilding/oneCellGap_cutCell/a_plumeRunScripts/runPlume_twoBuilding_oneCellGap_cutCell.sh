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


./cudaPlume/cudaPlume -q ../testCases/twoBuilding/oneCellGap_cutCell/a_plumeRunScripts/twoBuilding_oneCellGap_cutCell.xml -u ../testCases/twoBuilding/oneCellGap_cutCell/b_plumeInputs/twoBuilding_oneCellGap_cutCell_urb.nc -t ../testCases/twoBuilding/oneCellGap_cutCell/b_plumeInputs/twoBuilding_oneCellGap_cutCell_turb.nc -o ../testCases/twoBuilding/oneCellGap_cutCell/c_plumeOutputs/ -b twoBuilding_oneCellGap_cutCell -e -l -s -d

