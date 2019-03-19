#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="baseChar_net"
#SBATCH --time=140:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
echo "loading"
module load python/3.6.1
module load tensorflow/1.12.0-py36-gpu
echo "loaded"

python ../base_withCharEmb/learn.py --lr $1 --reg $2 --char_mod $3
