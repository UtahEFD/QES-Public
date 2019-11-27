#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-np
#SBATCH --qos=efd-np
#SBATCH --job-name=plume_FlatTerrain
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH -e plume_error.log
#SBATCH -o plume_slurm.log

module load gcc/5.4.0
ulimit -c unlimited -s

### run the debugger
###gdb ./cudaPlume/cudaPlume

### without debugger this used to be the following:
###./cudaPlume/cudaPlume -q ../data/FlatTerrain.xml -u ./FlatTerrain_urb.nc -t ./FlatTerrain_turb.nc -o FlatTerrain_plume.nc
### now it is the following:
###r -q ./FlatTerrain.xml -u ../data/FlatTerrain_urb.nc -t ./FlatTerrain_turb.nc -o FlatTerrain_plume.nc


./cudaPlume/cudaPlume -q ../data/FlatTerrain.xml -u ./FlatTerrain_urb.nc -t ./FlatTerrain_turb.nc -o FlatTerrain_plume.nc


