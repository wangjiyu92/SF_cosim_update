module purge
module load anaconda3
source activate /projects/salmon/envs/jiyu_salmon
which python
which -a python
conda env list
conda list foresee
export PYTHONPATH=$(pwd)
export SCENARIO_NAME="$1"
export NO_OF_HOMES="10"
export DER_PENETRATION='BAU'
export NO_OF_DAYS="1"
export START_DATE="1"
export MONTH="1"
export HOUSE="True"
export FEEDER="True"
export HEMS="True"
export AGGREGATOR="True"
export BUILDING_MODEL='test_sf'

