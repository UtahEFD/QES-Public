#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=matlabPlots
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --exclude=notch055
#SBATCH --export=ALL
#SBATCH -e plume_error_matlabPlots.log
#SBATCH -o plume_slurm_matlabPlots.log
ulimit -c unlimited -s

module load matlab

cd ../util/MATLAB/c_CUDA-PlumePlotting/a_runscripts/a_inputPlots
matlab -nodisplay -r b_inputPlots_channel -logfile b_inputPlots_channel.log

echo "finished batch"
exit

cd ../../../../../build_plume

