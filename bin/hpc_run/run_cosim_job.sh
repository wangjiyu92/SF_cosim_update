source bin/hpc_run/preamble.sh "$1"

echo "Running Cosim"
python bin/make_config_file.py eagle
echo "Cosim complete, running analysis"