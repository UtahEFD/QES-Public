#!/bin/bash
#SBATCH --account=efd-np
#SBATCH --partition=efd-shared-np
#SBATCH --job-name=MarshallFire
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=36:00:00
#SBATCH -e mar_error.log
#SBATCH -o mar_out.log
module load gcc/8.5.0
ulimit -c unlimited -s
./../build/qesFire/qesFire -q ../data/InputFiles/fireMarshallTime.xml -s 3 -o ~/../stoll-groupbig/moody/Marshall/Force -b
