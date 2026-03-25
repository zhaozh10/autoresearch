#!/bin/bash
#SBATCH --partition=radcluster
#SBATCH --job-name=auto
#SBATCH --time=24:00:00
#SBATCH --nodelist=hawking
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=120G

nvidia-smi
sleep infinity
# source ~/llama/bin/activate

# python ~/care/agent_script.py --model_name medvlm --output_path medvlm.csv

