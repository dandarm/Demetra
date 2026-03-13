#!/bin/bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=5
#SBATCH --time=08:20:00
#SBATCH --error=specialization.err
#SBATCH --output=specialization.out
#SBATCH --account=iscrc_same-d2
#SBATCH --job-name=specialization_videomae

module load profile/deeplrn
module load cineca-ai/4.3.0
source $WORK/videomae2/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12340

export PYTHONWARNINGS=ignore
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

# A/B rapido compile:
#   sbatch --export=ALL,COMPILE_OVERRIDE=off sbatch_specialization.sh
#   sbatch --export=ALL,COMPILE_OVERRIDE=on  sbatch_specialization.sh
COMPILE_OVERRIDE=${COMPILE_OVERRIDE:-auto}

mpirun --map-by socket:PE=5 --report-bindings --tag-output python -u specialization.py --on leonardo --compile "${COMPILE_OVERRIDE}"
    
