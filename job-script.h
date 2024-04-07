#!/bin/bash

#SBATCH --nodes=1 ##Number of nodes I want to use

#SBATCH --mem=1024  ##Memory I want to use in MB

#SBATCH --time=12:00:00 ## time it will take to complete job

#SBATCH --partition=gpu ##Partition I want to use

#SBATCH --gres=gpu:1

#SBATCH --ntasks=1 ##Number of task

#SBATCH --job-name=ks ## Name of job

#SBATCH --output=/active/debruinz_project/hongha/test-job.%j.out ##Name of output file

module load ml-python/nightly
export PYTHONPATH=/usr/bin/python:/active/debruinz_project/hongha/MMVAE_Architecture_Team
python /active/debruinz_project/hongha/MMVAE_Architecture_Team/get-started/Arch_Main.py

