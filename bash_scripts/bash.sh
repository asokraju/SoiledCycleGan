#!/bin/bash -i

LAMBDA=$1
LAMBDA_DIR="${LAMBDA}"
DIR=$PWD
PARENT_DIR="$(dirname $PWD)"
mkdir -p ./$WEIGHT_DIR
cd ./$WEIGHT_DIR

source activate tfgpu
PATH="C:\Users\kkosara\OneDrive - Clemson University\Documents\GitHub\SoiledCycleGan\run.py"
export run_exec=$PATH #python script that we want to run
export run_flags="--env_weight=${WEIGHT} --summary_dir=$PWD"   #flags for the script
C:\\ProgramData\\Anaconda3\\envs\\tfgpu\\python.exe $run_exec $run_flags