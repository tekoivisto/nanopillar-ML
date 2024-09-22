#!/bin/bash
#SBATCH -J LAMMPS
#SBATCH --time=01:15:00
#SBATCH --partition=small
#SBATCH --mem-per-cpu=10
#SBATCH --nodes=1
#SBATCH --ntasks=1

module purge
module load gcc/11.3.0 cuda openmpi intel-oneapi-mkl/2022.1.0


for seed in {1..5000}
do
    srun nanocrystal_generator --filename "polycrystal_${seed}.lmp" --lattice_type bcc --box 100 100 200 --atom_type Ta --grain_amount 8 --lattice_constant 3.300101 --seed $seed -n
done

