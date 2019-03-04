#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="merge review"
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
echo "loading"
module load python/3.6.1
echo "loaded"

python ../data_process/merge_review.py
