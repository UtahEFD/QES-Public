#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileyChannel_1o83_18o3
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error_BaileyChannel.log
#SBATCH -o plume_slurm_BaileyChannel.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testCases/Bailey/b_channel/a_plumeRunScripts/BaileyChannel_1o83_18o3.xml -u ../testCases/Bailey/b_channel/b_plumeInputs/BaileyChannel_urb.nc -t ../testCases/Bailey/b_channel/b_plumeInputs/BaileyChannel_turb.nc -o ../testCases/Bailey/b_channel/c_plumeOutputs/c_1o83_18o3/ -b channel -e -l -s -d

