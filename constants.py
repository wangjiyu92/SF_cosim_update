import os
import pandas as pd
from datetime import datetime, timedelta

# Scenario Name (used for finding Master Spreadsheet)
scenario_name = os.environ['SCENARIO_NAME'] if 'SCENARIO_NAME' in os.environ else 'test40'
no_of_homes = int(os.environ['NO_OF_HOMES']) if 'NO_OF_HOMES' in os.environ else 10
der_penetration_pc = os.environ['DER_PENETRATION'] if 'DER_PENETRATION' in os.environ else 'BAU'  # 'BAU', '50p', '100p'
building_model = os.environ['BUILDING_MODEL'] if 'BUILDING_MODEL' in os.environ else 'test_SF'
debug = True

# Simulation interval
start_date = int(os.environ['START_DATE']) if 'START_DATE' in os.environ else 1
days = int(os.environ['NO_OF_DAYS']) if 'NO_OF_DAYS' in os.environ else 1
month = int(os.environ['MONTH']) if 'MONTH' in os.environ else 1
# Runs from scenario name: 
month=1

year = 2021
start_time = datetime(year, month, start_date, 0, 0)  # (Year, Month, Day, Hour, Min)
duration = timedelta(days=days)
# start_time = datetime(year, 3, 2, 0, 0)  # (Year, Month, Day, Hour, Min)
# duration = timedelta(days=7)
time_step = timedelta(seconds=60)
end_time = start_time + duration
times = pd.date_range(start=start_time, end=end_time, freq=time_step)[:-1]

# Agents to run
include_house = os.environ['HOUSE'] == 'True' if 'HOUSE' in os.environ else True
include_feeder = os.environ['FEEDER'] == 'True' if 'FEEDER' in os.environ else True
include_hems = os.environ['HEMS'] == 'True' if 'HEMS' in os.environ else False
include_aggregator = os.environ['AGGREGATOR'] == 'True' if 'AGGREGATOR' in os.environ else False
#include_aggregator = True

# Frequency of Updates
freq_house = timedelta(minutes=1)
freq_hems = timedelta(minutes=15)
freq_feeder = timedelta(minutes=1)
freq_aggregator = timedelta(minutes=15)
freq_save_results = timedelta(hours=1)

ev_aggregator=False

# Foresee variables 
hems_horizon = timedelta(hours = 1)

# Time offsets for communication order
offset_house_run = timedelta(seconds=0)
offset_feeder_run = timedelta(seconds=0)
offset_hems_to_agg = timedelta(seconds=0)
# offset_hems_run = timedelta(seconds=60)# timedelta(seconds=20)
offset_aggregator_send= timedelta(seconds=0)
offset_aggregator_receive= timedelta(seconds=10)
offset_hems_to_house = timedelta(seconds=20)# timedelta(seconds=30)
offset_house_to_hems = timedelta(seconds=25)# timedelta(seconds=40)

#offset_aggregator_receive= timedelta(seconds=0)

offset_save_results = timedelta(0)

aggregator_horizon = timedelta(hours=1)

# Input/Output file paths
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
base_path = r'/projects/reorg1/SF_cosim_update/'
input_path = os.path.join(base_path, "inputs")
output_path = os.path.join(base_path, "outputs", scenario_name)
baseline_output_path = os.path.join(base_path,"outputs", "Test_Feeder_House_10house_baseline")
print('Output path:', output_path)
feeder_input_path = os.path.join(input_path, "opendss")
doom_input_path = os.path.join(input_path, "house")
resstock_input_path = os.path.join(input_path, "resstock_schedules")
foresee_input_path = os.path.join(input_path, "foresee")

# Input file locations
# TODO: update
print('SCENARIO NAME:', scenario_name)
if scenario_name == 'test':
    master_dss_file = os.path.join(feeder_input_path, "Secondary_p2udt2338_p2udt2338lv", "Master.dss")
elif scenario_name == 'test40':
    # master_dss_file = os.path.join(feeder_input_path, "test_40_houses", "Master.dss")
    # master_dss_file = os.path.join(feeder_input_path, "UseCase5", "Master_snap.dss")
    master_dss_file = os.path.join(feeder_input_path, "UseCase5", "Master.dss")
else:
    # TODO: Update once we have original feeder
    master_dss_file = os.path.join(feeder_input_path, "model_SF", "Master_qsts.dss")
print("MASTER DSS FILE:", master_dss_file)
epw_weather_file_name = os.path.join(input_path, 'weather', 'G4100510.epw')
print('Weather file:', epw_weather_file_name)
# pv_profile = os.path.join(input_path, 'solar pv', 'pv_profile_sd_ca.csv')


PVshape_file = os.path.join(feeder_input_path, "model_SF", 'PVShape.csv')
pvinfo_filename = os.path.join(feeder_input_path, 'Pv_info.csv')
tflag=0

# Output file locations
house_results_path = os.path.join(output_path, 'Ochre')
print("House result path:", house_results_path)
hems_results_path = os.path.join(output_path, 'Foresee')
feeder_results_path = os.path.join(output_path, 'Feeder')

# processing master spreadsheet
# UPDATED THE NAME OF MS once we have final version
# ms_file = os.path.join(input_path, "MS", "Main_spreadsheet_test40.xlsx")
# ms_file = os.path.join(input_path, "MS", "Main_spreadsheet_SALMON_test_scaling.xlsx")
# ms_file = os.path.join(input_path, "MS", "Main_spreadsheet_SALMON_with_buildings_updated3.xlsx")
ms_file = os.path.join(input_path, "MS", "Main_spreadsheet_SF_with_foresee_test.xlsx")


print("MS file:", ms_file)
master_df = pd.read_excel(ms_file, index_col='House_ID')[:no_of_homes]
# house_no_start = 320
# master_df = pd.read_excel(ms_file, index_col='House_ID')[house_no_start:house_no_start+no_of_homes]
house_ids = master_df.index.to_list()
feeder_loads = dict(zip(house_ids, master_df['Load_name']))
print(house_ids)

# Additional Configuration for foresee
HIL_sim = False
BTO_sim = False
hems_scenario = 'SETO'
resilience_mode = False
