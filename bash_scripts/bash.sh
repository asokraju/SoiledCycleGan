#!/bin/bash -i

LAMBDA=$1
LAMBDA_DIR="lambda=${LAMBDA}"
DIR=$PWD
PARENT_DIR="$(dirname $PWD)"
mkdir -p ./$LAMBDA_DIR
cd ./$LAMBDA_DIR

source activate tfgpu
PATH="C:\\Users\\kkosara\\SoiledCycleGan\\run.py"
export run_exec=$PATH #python script that we want to run
export run_flags="--LAMBDA=${LAMBDA} --summary_dir=$PWD"   #flags for the script
C:\\ProgramData\\Anaconda3\\envs\\tfgpu\\python.exe $run_exec $run_flags