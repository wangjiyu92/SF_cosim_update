import numpy as np

from constants import *
from agents import Agent

from ochre import Dwelling
# from dwelling_model import Dwelling

HOUSE_DEFAULT_ARGS = {
    # Timing parameters
    'start_time': start_time,
    'time_res': freq_house,
    'duration': duration,
    'times': times,
    'initialization_time': timedelta(days=7),

    # Input and Output Files
    # 'input_path': doom_input_path,
    'output_path': house_results_path,
    'weather_file': epw_weather_file_name,

    # 'envelope_model_name': 'Env',
    'assume_equipment': True,
    # 'uncontrolled_equipment': ['Lighting', 'Exterior Lighting', 'Range', 'Dishwasher', 'Refrigerator',
    #                            'Clothes Washer', 'Clothes Dryer', 'MELs', 'EV Scheduled'],  # List of uncontrolled loads
    'uncontrolled_equipment': ['Indoor Lighting', 'Exterior Lighting', 'Range', 'Dishwasher', 'Refrigerator',
                               'Clothes Washer', 'Clothes Dryer', 'MELs', 'EV Scheduled'],  # List of uncontrolled loads
    'save_results': True,
    'verbosity': 7,
    'ext_time_res': freq_hems,
    'timezone':None,
}

equipment = {

}


def update_house_args(house_id):
    house_args = HOUSE_DEFAULT_ARGS
    print('MS DF:', master_df)
    # Load Master Spreadsheet and get house information
    house_row = master_df.loc[house_id]
    house_args.update(house_row)

    if 'PV (kW)' in house_args: 
        if house_args['PV (kW)'] > 0:
            scale_factor = house_args['PV (kW)']
            equipment['PV'] = {
                'equipment_schedule_file': pv_profile,
                'schedule_scale_factor': -1 * scale_factor,  # flip sign of PV power
                'val_col': str(house_args['PV_Profile_ID'])
                # 'capacity': house_args['PV (kW)'],
                # 'inverter_capacity': house_args['PV (kW)'] * 0.9
            }

    if 'EV' in house_args: 
        print("House Args", house_args['EV'], type(house_args['EV']))
        if house_args['EV'] != 'None':
            ev_type, ev_mileage = tuple(house_args['EV'].split('-'))
            equipment['EV'] = {
                'vehicle_type': ev_type,
                'charging_level': 'Level 1' if ev_type == 'PHEV' else 'Level 2',
                'mileage': float(ev_mileage),
            }

    if 'Battery (kWh)' in house_args:
        if house_args['Battery (kWh)'] > 0:
            if 'Battery Control' in house_args.keys():
                if house_args['Battery Control'] == 'schedule':
                    equipment['Battery'] = {
                    'capacity_kwh': house_args['Battery (kWh)'],
                    'capacity_kw': house_args['Battery (kWh)'] / 2,
                    'control_type': 'Schedule',
                    'charge_start_hour': 22,
                    'charge_power': 1,
                    'discharge_start_hour': 17,
                    'discharge_power': 2,
                    # 'soc_init': house_args['Initial SOC'],
                    }
            else: 
                equipment['Battery'] = {
                'capacity_kwh': house_args['Battery (kWh)'],
                'capacity_kw': house_args['Battery (kWh)'] / 2,
                'control_type': 'Self-Consumption',
                'charge_any_solar': True,
                # 'enable_schedule': False
                }

    # Properties file
    # prop_file_path = os.path.join(doom_input_path, 'test_scaling', house_args['DOOM Input File'], 'run')
    prop_file_path = os.path.join(doom_input_path, house_args['DOOM Input File'], 'run')
    hpxml_file = 'in.xml'
    house_args['properties_file'] = os.path.join(prop_file_path, hpxml_file)
    house_args['hpxml_file'] = os.path.join(prop_file_path, hpxml_file)

    # Schedule file
    # resstock_id = house_args['ResStock ID']
    # occupants = house_args['Num_Occupant']
    # resstock_path = os.path.join(resstock_input_path, f'schedules_{int(occupants)}', f'bldg{int(resstock_id):07d}')
    house_args['schedule_file'] = os.path.join(prop_file_path, 'schedules.csv')
    house_args['schedule_input_file'] = os.path.join(prop_file_path, 'schedules.csv')
    # house_args['max_values_type'] = str(resstock_id)
    # house_args['max_values_file'] = os.path.join(resstock_path, 'max_schedules.csv')

    # Output path
    house_args['output_path'] = os.path.join(house_results_path, 'House_{}'.format(house_id)) 
    return house_args


class House(Agent):
    def __init__(self, house_id, **kwargs):
        print("HOUSE MODEL CALLED")
        self.house_id = house_id
        print('_house_id')
        print(self.house_id)

        self.hems_controls = None
        self.status = {}
        self.house = None
        self.utility_battery = False
        kwargs['result_path'] = os.path.join(house_results_path, 'House_{}'.format(house_id)) 
        super().__init__('House_' + house_id, **kwargs)

    def initialize(self):
        house_args = update_house_args(self.house_id)
        self.print_log(house_args)
        self.print_log(equipment)
        house_args['Equipment'] = equipment
        self.house = Dwelling(self.house_id, **house_args)

        # if 'Battery (kWh)' in house_args:
        #     if house_args['Battery (kWh)'] > 0:
        #         if 'Battery Control' in house_args.keys():
        #             if house_args['Battery Control'] == 'schedule': 
        #                 battery = self.house.equipment_by_end_use['Battery'][0]
        #                 battery.soc = house_args['Initial SOC']

        # # Utility Battery
        # if 'Battery (kWh)' in house_args:
        #     if house_args['Battery (kWh)'] > 0 and (house_args['Utility Battery'] == 'TRUE' or house_args['Utility Battery'] == 1): 
        #         self.utility_battery = True
        #     else: 
        #         self.utility_battery = False

    def setup_pub_sub(self):
        topic_to_feeder_load = "house_{}_power_to_feeder".format(self.house_id)
        self.register_pub("power", topic_to_feeder_load, "String")

        topic_from_feeder = "feeder_vtg_to_house_{}".format(self.house_id)
        self.register_sub("feeder", topic_from_feeder, default={})

        if include_hems:
            topic_to_ctrlr_status = "house_status_to_ctrlr_{}".format(self.house_id)
            self.register_pub("status", topic_to_ctrlr_status, "String")

            topic_from_ctrlr = "ctrlr_controls_to_house_{}".format(self.house_id)
            self.register_sub("controls", topic_from_ctrlr)

    def setup_actions(self):
        # Note: order matters!
        if include_hems:
            self.add_action(self.get_hems_controls, 'Get HEMS Controls', freq_hems, offset_house_to_hems)
        self.add_action(self.run_house, 'Run House', freq_house, offset_house_run)
        if include_hems:
            self.add_action(self.send_status_to_hems, 'Send House Status', freq_hems, offset_house_run)
        self.add_action(self.save_results, 'Save Results', freq_save_results, offset_save_results)

    def get_voltage_from_feeder(self):
        voltage = self.fetch_subscription("feeder")
        return 1 if voltage is None else voltage.get('voltage')

    def send_powers_to_feeder(self, power_to_dss):
        self.publish_to_topic("power", power_to_dss)

    def send_status_to_hems(self):
        # convert times to string
        if 'Time' in self.status.keys():
            self.status['Time'] = str(self.status['Time'])
            if 'EV Start Time' in self.status:
                self.status['EV Start Time'] = str(self.status['EV Start Time'])
                self.status['EV End Time'] = str(self.status['EV End Time'])

        self.publish_to_topic("status", self.status)

    def get_hems_controls(self):
        self.hems_controls = self.fetch_subscription("controls")
        print('Get_hems_control:', self.hems_controls)

    def run_house(self):
        voltage = self.get_voltage_from_feeder()

        # run simulator
        # if include_hems and self.hems_controls and voltage:
        #     # results = self.house.update(voltage=voltage, from_ext_control=self.hems_controls)
        #     results = self.house.update_inputs(schedule_inputs={'Voltage (-)': voltage}, from_ext_control=self.hems_controls)
        #     to_dss = {
        #         'P Total': results['Total Electric Power (kW)'],
        #         'Q Total': results['Total Reactive Power (kVAR)'],
        #     }
        # else:
        #     self.print_log("Did not receive house controls: {} and voltage {}".format(self.hems_controls, voltage))
        #     if voltage == 'None' or voltage == None: 
        #         # results = self.house.update(ext_model_args={'Voltage (-)': 1})
        #         self.house.update_inputs(schedule_inputs={'Voltage (-)': 1})
        #         results = self.house.generate_results()
        #         pass
        #     else:
        #         # results = self.house.update(ext_model_args={'Voltage (-)': voltage})
        #         self.house.update_inputs(schedule_inputs={'Voltage (-)': voltage})
        #         results = self.house.generate_results()
        #     self.print_log("OCHRE results:", results)
        schedule_inputs = {'Voltage (-)': voltage} if voltage is not None else None
        control_signal = self.hems_controls if include_hems else None
        results = self.house.update(control_signal, schedule_inputs)
        # self.print_log("OCHRE results:", results)
        self.print_log("House controls: {} and voltage {}".format(self.hems_controls, voltage))
        to_dss = {
            'P Total': results['Total Electric Power (kW)'],
            'Q Total': results['Total Reactive Power (kVAR)'],
        }
        
        # Utility Battery
        if self.utility_battery: 
            to_dss['P Total'] = to_dss['P Battery']
            to_dss['Q Total'] = to_dss['Q Battery']

        self.send_powers_to_feeder(to_dss)

        # Save simulator outputs for online controller and HEMS
        if include_hems:
            self.status = results

        # self.status = {}
        # for k, v in to_ext_control.items():
        #     if isinstance(v, np.ndarray):
        #         self.status[k] = v[0]
        #     else:
        #         self.status[k] = v
        # self.print_log(self.status)
        
    def save_results(self):
        super().save_results()
        self.house.export_results()

    def finalize(self):
        self.house.finalize()
        # self.house.calculate_metrics()

        super().finalize()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        house_id = str(sys.argv[1])
        addr = str(sys.argv[2])
        agent = House(house_id, broker_addr=addr)
    elif len(sys.argv) == 2:
        house_id = str(sys.argv[1])
        agent = House(house_id, debug=True, run_helics=False)
    else:
        raise Exception('Must specify House ID')

    agent.simulate()
