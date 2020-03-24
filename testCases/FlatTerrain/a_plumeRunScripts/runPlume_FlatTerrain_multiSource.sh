#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_FlatTerrain_multiSource
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=2G
#SBATCH --time=01:30:00
#SBATCH --exclude=notch055
#SBATCH -e plume_error_FlatTerrain.log
#SBATCH -o plume_slurm_FlatTerrain.log

module load gcc/5.4.0
ulimit -c unlimited -s


./cudaPlume/cudaPlume -q ../testCases/FlatTerrain/a_plumeRunScripts/FlatTerrain_multiSource.xml -u ../testCases/FlatTerrain/b_plumeInputs/FlatTerrain_urb.nc -t ../testCases/FlatTerrain/b_plumeInputs/FlatTerrain_turb.nc -o ../testCases/FlatTerrain/c_plumeOutputs/multiSource/ -b FlatTerrain_multiSource -e -l -s -d

