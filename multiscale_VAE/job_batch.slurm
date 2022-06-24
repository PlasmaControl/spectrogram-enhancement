#!/bin/bash
#SBATCH --job-name=spec_pipeline     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --array=0-9              # job array with indices inclusive
#SBATCH --output=/scratch/gpfs/ar0535/error_files/slurm_%j.out
#SBATCH --gres=gpu:1

echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on machine:" $(hostname)

module purge
module load anaconda3/2020.11
conda activate tf2-gpu

python /home/ar0535/spectrogram-enhancement/multiscale_VAE/train_MSAE.py
