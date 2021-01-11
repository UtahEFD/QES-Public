#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=QES-fire
#SBATCH --nodes=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:titanv:1
#SBATCH --time=01:00:00
#SBATCH -e init_error.log
#SBATCH -o init_out.log
module load gcc/8.1.0
ulimit -c unlimited -s
./qesFire/qesFire -q ../data/InputFiles/fireIncPlane.xml -s 3 -o IncPlane

