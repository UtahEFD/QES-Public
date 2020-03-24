#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileySinewave_0o1_10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --exclude=notch055
#SBATCH -e plume_error_BaileySinewave.log
#SBATCH -o plume_slurm_BaileySinewave.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testCases/Bailey/a_sinewave/a_plumeRunScripts/BaileySinewave_0o1_10.xml -u ../testCases/Bailey/a_sinewave/b_plumeInputs/BaileySinewave_urb.nc -t ../testCases/Bailey/a_sinewave/b_plumeInputs/BaileySinewave_turb.nc -o ../testCases/Bailey/a_sinewave/c_plumeOutputs/c_0o1_10/ -b sinewave -e -l -s -d

