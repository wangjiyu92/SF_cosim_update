#!/bin/bash

#SBATCH --account=salmon
#SBATCH --time=3:00:00
#SBATCH --job-name=sf_test
#SBATCH --mail-user=Jiyu.Wang@nrel.gov
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL
#SBATCH --output=sf_test.%j.out
#SBATCH --qos=high

source bin/hpc_run/run_cosim_job.sh "$1"
