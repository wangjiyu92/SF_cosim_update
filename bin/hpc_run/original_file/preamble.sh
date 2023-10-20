module purge
module load conda
source activate /projects/salmon/pmunanka_test/SALMON_Cosim_Jiyu
which python
which -a python
conda env list
conda list foresee
export PYTHONPATH=$(pwd)
export SCENARIO_NAME="$1"
export NO_OF_HOMES="5"
export DER_PENETRATION='BAU'
export NO_OF_DAYS="1"
export START_DATE="1"
export MONTH="1"
export HOUSE="True"
export FEEDER="True"
export HEMS="False"
export AGGREGATOR="False"
export BUILDING_MODEL='test_sf'

