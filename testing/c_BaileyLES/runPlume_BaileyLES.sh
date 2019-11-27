#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileyLES
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error.log
#SBATCH -o plume_slurm.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testing/c_BaileyLES/BaileyLES.xml -u ../testing/c_BaileyLES/BaileyLES_urb.nc -t ../testing/c_BaileyLES/BaileyLES_turb.nc -o BaileyLES_plume.nc


