#!/bin/bash

#SBATCH --account=estcpmadrl
#SBATCH --time=5:00:00
#SBATCH --job-name=sf_test
#SBATCH --nodes=5
#SBATCH --mail-user=jwang4@nrel.gov
#SBATCH --mail-type=FAIL
#SBATCH --output=sf_test.%j.out
#SBATCH --qos=normal

source bin/hpc_run/run_cosim_job.sh "$1"
