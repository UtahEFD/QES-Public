#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_FlatTerrain_singleLine
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error_FlatTerrain.log
#SBATCH -o plume_slurm_FlatTerrain.log

module load gcc/5.4.0
ulimit -c unlimited -s


./cudaPlume/cudaPlume -q ../testCases/FlatTerrain/a_plumeRunScripts/FlatTerrain_singleLine.xml -u ../testCases/FlatTerrain/b_plumeInputs/FlatTerrain_urb.nc -t ../testCases/FlatTerrain/b_plumeInputs/FlatTerrain_turb.nc -o ../testCases/FlatTerrain/c_plumeOutputs/singleLine/ -b FlatTerrain_singleLine -d

