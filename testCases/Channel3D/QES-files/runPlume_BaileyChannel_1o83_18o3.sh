#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileyChannel_1o83_18o3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --exclude=notch055
#SBATCH -e plume_error_BaileyChannel.log
#SBATCH -o plume_slurm_BaileyChannel.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testCases/Channel3D/QES-files/BaileyChannel_1o83_18o3.xml -u ../testCases/Channel3D/MATLAB-inputData/channel3D_windsWk.nc -t ../testCases/Channel3D/MATLAB-inputData/channel3D_turbOut.nc -o ../testCases/Channel3D/QES-data/ -b BaileyChannel_1o83_18o3 -e -l -s -d

