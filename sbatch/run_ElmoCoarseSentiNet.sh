#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="elmo"
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
echo "loading"
module load python/3.6.1
module load tensorflow/1.12.0-py36-gpu
echo "loaded"

python ../learn/LEARN_ElmoCoarseSentiNet.py --num $1 --epoch $2