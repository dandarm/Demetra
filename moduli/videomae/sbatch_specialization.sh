#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --error=specialization.err
#SBATCH --output=specialization.out
#SBATCH --account=iscrc_same-d2


module load profile/deeplrn
module load cineca-ai/4.3.0
source $HOME/videomae/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340

export PYTHONWARNINGS=ignore
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1

mpirun --map-by socket:PE=4 --report-bindings --tag-output python -u specialization.py \
    --on leonardo \
    
