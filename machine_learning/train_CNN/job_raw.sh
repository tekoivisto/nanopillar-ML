#!/bin/bash
#SBATCH -J CNN
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --mem=32768
#SBATCH --array=1-5

module purge
module load Miniconda3
source activate tensorflow-env-paper

mkdir -p "checkpoints_$SLURM_ARRAY_TASK_ID"

num_epochs=2000
patience=50
filter_divider=2
batch_size=16
grad_cam_batch_size=8
learning_rate=0.005

gt_file="$1"
resolution="$2"
descriptor="$3"

srun python3 "../../../train_NG_CNN.py" --seed $SLURM_ARRAY_TASK_ID --ground_truth $gt_file --descriptors $descriptor --resolution $resolution --num_epochs $num_epochs --index 0 --patience $patience --learning_rate $learning_rate --filter_divider $filter_divider --batch_size $batch_size --grad_cam_batch_size $grad_cam_batch_size
