#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=matlabPlot_inputPlots_BaileySinewave
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH -e plume_error_matlabPlot_inputPlots_BaileySinewave.log
#SBATCH -o plume_slurm_matlabPlot_inputPlots_BaileySinewave.log
ulimit -c unlimited -s

module load matlab

cd ../util/MATLAB/c_CUDA-PlumePlotting/a_runscripts/a_inputPlots
matlab -nodisplay -r a_inputPlots_sinewave -logfile a_inputPlots_sinewave.log

echo "finished batch"
exit

cd ../../../../../build_plume

