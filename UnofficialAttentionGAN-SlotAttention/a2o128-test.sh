#!/bin/bash
#
# -= Resources =-
#
#SBATCH --job-name=comp_411a2o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --partition=mid
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1-0
#SBATCH --output=train-a2o-128-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ebostanci18@ku.edu.tr


# Set stack size to unlimited

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

echo "==============================================================================="
# Command 1 for matrix
echo "Running Python script..."

echo "Loading cuda"
module load cuda/11.8.0
module load cudnn/8.2.2/cuda-11.4

echo "Loading conda env"
module load anaconda/3.6

conda init bash

conda activate comp411

source activate comp411
echo "activated env"

conda run -n comp411 python test-apple2orange.py
