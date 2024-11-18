#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=TestFire
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --gres=gpu:3090:1
#SBATCH --time=36:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.5.0
ulimit -c unlimited -s
./qesFire/qesFire -q ../data/InputFiles/fire.xml -s 3 -o test -b
