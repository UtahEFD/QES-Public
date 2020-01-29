#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileyLES_2o22_222
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error_BaileyLES.log
#SBATCH -o plume_slurm_BaileyLES.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testCases/Bailey/c_LES/a_plumeRunScripts/BaileyLES_2o22_222.xml -u ../testCases/Bailey/c_LES/b_plumeInputs/BaileyLES_urb.nc -t ../testCases/Bailey/c_LES/b_plumeInputs/BaileyLES_turb.nc -o ../testCases/Bailey/c_LES/c_plumeOutputs/c_2o22_222/BaileyLES_plume_2o22_222.nc -d ../testCases/Bailey/c_LES/c_plumeOutputs/c_2o22_222

