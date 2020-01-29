#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=matlabPlot_resultPlots_BaileyChannel
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH -e plume_error_matlabPlot_resultPlots_BaileyChannel.log
#SBATCH -o plume_slurm_matlabPlot_resultPlots_BaileyChannel.log
ulimit -c unlimited -s

module load matlab

cd ../util/MATLAB/c_CUDA-PlumePlotting/a_runscripts/e_resultPlots
matlab -nodisplay -r b_resultPlots_channel -logfile b_resultPlots_channel.log

echo "finished batch"
exit

cd ../../../../../build_plume

