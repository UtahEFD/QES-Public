#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=QES-fire
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.5.0
ulimit -c unlimited -s
./../build/qesFire/qesFire -q ../data/InputFiles/BMRA_5.xml -s 3 -o ~/../stoll-groupbig/moody/BMRA/fuel5 -b
