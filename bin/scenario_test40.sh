module purge
module load conda
source activate fastderms_develop
export BASE_DIRECTORY=/projects/fastderms/cosim_share/fastderms_cosim
export PYTHONPATH=$PYTHONPATH:$BASE_DIRECTORY
which python
export SCENARIO_NAME="test40"
export NO_OF_HOMES="100"
export DER_PENETRATION='BAU'
export NO_OF_DAYS="7"
export START_DATE="1"
export MONTH="1"
export HOUSE="True"
export FEEDER="True"
export BUILDING_MODEL='sd_ca'