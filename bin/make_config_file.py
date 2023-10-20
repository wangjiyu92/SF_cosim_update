import shutil
import sys
import os
import math
import json
import socket
import subprocess
import cProfile
from copy import copy
from datetime import datetime
try:
    from constants import *
except ImportError:
    import sys
    import os

    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)
    from constants import *

# Get the host type 
host = sys.argv[1]
print("Base path:", base_path)
houses_per_broker = 40

# Set paths
config_file = os.path.join(base_path, "bin", "config.json")
federates_directory = os.path.join(base_path, 'agents')

# config skeleton
config = {"federates": [],
          "name": scenario_name,
          }
federate = {"directory": federates_directory,
            "host": "localhost",
            }

def get_ip_address():
    ip_addr = socket.gethostbyname(socket.gethostname())

    # Other methods:
    # print('IP address 2:', socket.gethostbyname_ex(socket.gethostname())[-1])
    # print('IP address 3:', socket.gethostbyname(socket.getfqdn()))
    return ip_addr


# List the nodes allocated on job for ssh
def get_nodes_info():
    nodes_count = int(os.environ['SLURM_NNODES'])
    
    # nodes = os.environ['SLURM_JOB_NODELIST']
    nodes = subprocess.run(['scontrol', 'show', 'hostnames'], capture_output=True).stdout.decode()
    nodes = nodes.split('\n')
    if nodes[-1] == '':
        nodes = nodes[:-1]
    print('Hostnames (i.e. node names):', nodes)
    assert len(nodes) == nodes_count
    
    return nodes_count, nodes

# Execute commands by ssh-ing into the node
def ssh_nodes(cmd_list):
    ssh = subprocess.Popen(cmd_list,
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           preexec_fn=os.setsid)
    print("process running", cmd_list)
    result = ssh.stdout.readlines()
    if result == []:
        error = ssh.stderr.readlines()
        print("SSH error", error)
    else:
        result = result[0].rstrip()
        print(result)
    return ssh, result


# Create multiple config files to run on multiple nodes
def construct_configs():
    config_files = []  # Config files for HELICS
    config_outfiles = []  # Capture the stdout on the node
    config_errfiles = []  # Capture the stderr on the node

    # Determine the number of config files needed
    federates = config["federates"]
    fed_len = len(federates)
    no_of_config_file = math.floor(fed_len / houses_per_broker)

    if len(federates) % houses_per_broker > 0:
        no_of_config_file += 1

    # creating config file
    for i in range(1, no_of_config_file + 1):
        with open("{}/config_{}.json".format(output_path, i), "w+") as f1:
            data = {
                # "broker": "false",
                "broker": False,
                "name": config["name"]
            }
            federates_new_config = federates[(i - 1) * houses_per_broker:i * houses_per_broker]
            if len(federates_new_config) > 0:
                data["federates"] = federates_new_config

            if data != None:
                f1.write(json.dumps(data))
                # outfile = open(os.path.join(output_path, "config_{}_outfile.txt".format(i)), 'a')
                # errfile = open(os.path.join(output_path, "config_{}_errfile.txt".format(i)), 'a')
                config_files.append("config_{}.json".format(i))
                # config_errfiles.append(errfile)
                # config_outfiles.append(outfile)

    return (fed_len, config_files, config_outfiles, config_errfiles)

def populate_configs(config, federate, ip_addr): 
    # Creating the feeder fed
    if include_feeder:
        feeder = copy(federate)
        feeder['name'] = "feeder"
        feeder['exec'] = "python Feeder.py {}".format(ip_addr)
        config['federates'].append(feeder)
        
    if include_aggregator:
        aggregator = copy(federate)
        aggregator['name'] = "aggregator"
        aggregator['exec'] = "python Aggregator.py {}".format(ip_addr)
        config['federates'].append(aggregator)

    # Add houses, hems, and brokers
    for i, load in enumerate(house_ids):
        # add house and hems
        house = copy(federate)
        house['exec'] = "python House.py {} {}".format(load, ip_addr)
        house['name'] = "house_{}".format(load)
        config['federates'].append(house)

        if include_hems:
            hems = copy(federate)
            hems['exec'] = "python Hems.py {} {}".format(load, ip_addr)
            hems['name'] = "hems_{}".format(load)
            config['federates'].append(hems)

    # Broker federate
    if host != "eagle": 
        broker = copy(federate)
        broker['name'] = "broker"
        broker['exec'] = 'helics_broker -f {}'.format(len(config['federates']))
        config['federates'].append(broker)
    
    return config

# Getting the broker's network address
if host != "eagle":
    ip_addr = '0.0.0.0:4545'
    config = populate_configs(config, federate, ip_addr)

# Create the output directory for the scenario
if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)
os.makedirs(hems_results_path, exist_ok=True)
os.makedirs(house_results_path, exist_ok=True)
os.makedirs(feeder_results_path, exist_ok=True)

# Record the start time of the simulation
start_time = datetime.now()

# save config to the main config file
if host == "localhost":
    print(start_time)
    with open(config_file, 'w+') as f:
        f.write(json.dumps(config, indent=4, sort_keys=True, separators=(",", ":")))
        cmd = "helics run --path bin/config.json --broker-loglevel=2"
        print(cmd)

elif host == "eagle":
    # nodes_count, node_list = get_nodes_info()

    # Get the broker's network address
    ip_address = get_ip_address()
    print('Broker IP address:', ip_address)

    # Get node count and names
    n_nodes, node_list = get_nodes_info()
    print(f'No. of Nodes:{n_nodes} and Nodes:{node_list}')

    config = populate_configs(config, federate, ip_address)
    (fed_len, config_files, out_files, err_files) = construct_configs()
    print("Number of federates", fed_len)
    print("config files", config_files)

    # Start the helics broker
    broker_out_file = open(os.path.join(output_path, 'broker_outfile.txt'), 'w')
    loglevel = 'debug'
    # broker_address = ip_address + ':4545'  # use this to specify broker port
    # cmd = f'helics_broker -t zmq_ss -f {fed_len} --ipv4 --loglevel={loglevel}'  # --broker_address={ip_address}'
    cmd = f'helics_broker -f {fed_len} --ipv4 --loglevel={loglevel}'
    cmd += ' --terminate_on_error'
    print(cmd)
    p_broker = subprocess.Popen(cmd.split(' '), stdout=broker_out_file)

    # ssh into each node and run the config files
    for i, (node, config_file) in enumerate(zip(node_list, config_files)):
        print(f'Node {node} running: {config_file}')

        # open error and output files
        # err_file = open(os.path.join(output_path, 'config_{}_errfile.txt'.format(i)), 'a')
        out_file = open(os.path.join(output_path, 'config_{}_outfile.txt'.format(i)), 'w')

        # ssh into node, set up environment with preamble code, and run the config file
        # cmd = 'cd {}; source bin/hpc_run/preamble.sh {}; python {}/bin/run_agents_from_helicsrunner.py {}'\
        #     .format(base_path, scenario_name, base_path, config_file)
        ssh_cmd = f'ssh -f {node} cd {base_path}; source bin/hpc_run/preamble.sh {scenario_name}; helics run --path {output_path}/{config_file}'
        # ssh_cmd = f'source bin/hpc_run/preamble.sh; helics run --path {output_path}{config_file}'
        print(ssh_cmd)
        p_ssh = subprocess.Popen(ssh_cmd.split(' '), stdout=out_file)
        # p_ssh = subprocess.Popen(ssh_cmd.split(' '), stdout=out_file, stderr=err_file)

    p_broker.wait()
    print("Time taken to finish the simulation: ", datetime.now() - start_time)
