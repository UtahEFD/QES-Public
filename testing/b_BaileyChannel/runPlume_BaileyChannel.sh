#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileyChannel
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error.log
#SBATCH -o plume_slurm.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testing/b_BaileyChannel/BaileyChannel.xml -u ../testing/b_BaileyChannel/BaileyChannel_urb.nc -t ../testing/b_BaileyChannel/BaileyChannel_turb.nc -o BaileyChannel_plume.nc


