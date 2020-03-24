#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileyLES_0o222_222
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=01:30:00
#SBATCH --exclude=notch055
#SBATCH -e plume_error_BaileyLES.log
#SBATCH -o plume_slurm_BaileyLES.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testCases/Bailey/c_LES/a_plumeRunScripts/BaileyLES_0o222_222.xml -u ../testCases/Bailey/c_LES/b_plumeInputs/BaileyLES_urb.nc -t ../testCases/Bailey/c_LES/b_plumeInputs/BaileyLES_turb.nc -o ../testCases/Bailey/c_LES/c_plumeOutputs/b_0o222_222/ -b LES -e -l -s -d

