#!/bin/bash
#
# You should only work under the /scratch/users/<username> directory.
#
# Example job submission script
#
# TODO:
#   - Set name of the job below changing "Test" value.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter.
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority.
#     - long  : For jobs that have maximum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-h$
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input and output file names below.
#   - If you do not want mail please remove the line that has --mail-type
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch examle_submit.sh

# -= Resources =-
#
#SBATCH --job-name=proje_411
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=mid
#SBATCH --gres=gpu:tesla_v100
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-0
#SBATCH --output=test-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brizai18@ku.edu.tr


# Set stack size to unlimited
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

echo "Running Example Job...!"
echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."

echo "Loading cuda"
module load cuda/11.8.0
module load cudnn/8.2.2/cuda-11.4 

echo "Loading conda env"
module load anaconda/3.6

conda init bash

conda activate proje411

source activate proje411
echo "activated env"
conda list
conda run -n proje411 python train.py --dataroot /datasets/CycleGan/summer2winter_yosemite --model attention_gan --gpu_ids 0 --max_dataset_size 150
