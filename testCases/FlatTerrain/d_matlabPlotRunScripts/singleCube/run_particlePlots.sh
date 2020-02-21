#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=matlabPlot_particlePlots_FlatTerrain
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH -e plume_error_matlabPlot_particlePlots_FlatTerrain.log
#SBATCH -o plume_slurm_matlabPlot_particlePlots_FlatTerrain.log
ulimit -c unlimited -s

module load matlab

cd ../util/MATLAB/c_CUDA-PlumePlotting/a_runscripts/c_particlePlots
matlab -nodisplay -r f_particlePlots_FlatTerrain_singleCube -logfile f_particlePlots_FlatTerrain_singleCube.log

echo "finished batch"
exit

cd ../../../../../build_plume

