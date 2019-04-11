#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=moser395
#SBATCH --nodes=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/5.4.0
ulimit -c unlimited -s
./cudaUrb/cudaUrb -q ../data/QU_Files/SHPSeoul_gangnam.xml -o Gangnam_DEM_SHP.nc -s 2
