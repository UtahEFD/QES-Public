#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_BaileySinewave
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error.log
#SBATCH -o plume_slurm.log

module load gcc/5.4.0
ulimit -c unlimited -s

./cudaPlume/cudaPlume -q ../testing/a_BaileySinewave/BaileySinewave.xml -u ../testing/a_BaileySinewave/BaileySinewave_urb.nc -t ../testing/a_BaileySinewave/BaileySinewave_turb.nc -o ../util/MATLAB/c_CUDA-PlumePlotting/c_plumeOutput/a_sinewave/c_HeteroAnisoExplicitTurb_0o1_10/BaileySinewave_plume.nc -d ../util/MATLAB/c_CUDA-PlumePlotting/c_plumeOutput/a_sinewave/c_HeteroAnisoExplicitTurb_0o1_10


