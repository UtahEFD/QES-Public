#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=QES-fire
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.5.0
ulimit -c unlimited -s
./../build/qesFire/qesFire -q ../data/InputFiles/fire.xml -s 3 -o ~/../stoll-groupbig/moody/QUIC/cure120 -b
