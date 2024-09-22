#!/bin/bash
#SBATCH -J LAMMPS
#SBATCH --time=4:37:00
#SBATCH --partition=large
#SBATCH --mem-per-cpu=90
#SBATCH --nodes=2
#SBATCH --ntasks=80
#SBATCH --array=1-500 ## change these values to deform all nanopillars from 1 to 5000. (All jobs cannot be submitted at once due to job limit in cluster.)

module purge
module load gcc/11.3.0 cuda openmpi intel-oneapi-mkl/2022.1.0

rate=0.002
temp=4
pillar_num=$(($SLURM_ARRAY_TASK_ID+0))

srun lmp -var temperature $temp -var compression_rate $rate -var pillar_num $pillar_num -in simulate.in
