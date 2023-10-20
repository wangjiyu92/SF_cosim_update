#!/bin/bash

#SBATCH --account=fastderms
#SBATCH --time=3:00:00
#SBATCH --job-name=test40
#SBATCH --nodes=2
#SBATCH --mail-user=pmunanka@nrel.gov
#SBATCH --mail-type=FAIL
#SBATCH --output=test40.%j.out
#SBATCH --qos=high

source bin/scenario_test_40.sh
    
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES

python bin/make_config_file.py eagle
echo "Simulation Complete"

cd /projects/fastderms/cosim_share/fastderms_cosim/outputs
zip -r test40.zip $SCENARIO_NAME
echo "All done. Checking results:"