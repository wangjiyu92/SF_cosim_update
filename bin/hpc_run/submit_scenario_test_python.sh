#!/bin/bash

#SBATCH --account=reorg1
#SBATCH --time=48:00:00
#SBATCH --job-name=python_test
#SBATCH --nodes=1
#SBATCH --mail-user=Yansong.Pei@nrel.gov
#SBATCH --mail-type=FAIL
#SBATCH --output=sf_test.%j.out
#SBATCH --error=sf_test.%j-error.out
#SBATCH --qos=normal
#SBATCH --partition=standard

module purge
module load conda
source activate /projects/salmon/pmunanka_test/SALMON_Cosim_Jiyu
which python
which -a python
conda env list
conda list foresee
python SF_v1_load_data_tt.py