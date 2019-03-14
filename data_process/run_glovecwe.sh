#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="CWEP"
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load python/3.6.1
python gendata_from_GloVeCWE_Emb.py