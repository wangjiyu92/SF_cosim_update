conda activate fastderms_develop
export BASE_DIRECTORY=$(pwd)
export PYTHONPATH=$PYTHONPATH:$BASE_DIRECTORY
export SCENARIO_NAME="test_ieee123"
export NO_OF_HOMES="1"
export DER_PENETRATION='BAU'
export NO_OF_DAYS="365"
export START_DATE="1"
export MONTH="1"
export HOUSE="True"
export FEEDER="True"
export BUILDING_MODEL='sd_ca'