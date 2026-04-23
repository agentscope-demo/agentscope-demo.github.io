#!/bin/bash
#SBATCH --job-name=mas-tom
#SBATCH --partition=normal_q
#SBATCH --account=jiaming_cui
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/kavanav/mas-tom/logs/slurm_%j.out
#SBATCH --error=/home/kavanav/mas-tom/logs/slurm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kavanav@vt.edu

cd /home/kavanav/mas-tom
source /home/kavanav/mas-tom/venv/bin/activate
export PYTHONPATH=/home/kavanav/mas-tom/src

python run_all.py --skip-existing