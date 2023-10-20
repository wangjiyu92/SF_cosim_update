from constants import *
import time
import pytz
import tzlocal
from agents import Agent

from ochre import Analysis
# from foresee import hems_optimizer_reverse, hems_optimizer_with_band
from foresee.hems_optimizer import Foresee

# from dav_kafka_python.consumer import PythonConsumer
# from dav_kafka_python.producer import PythonProducer

DEFAULT_HEMS_ARGS = {
    'util_feedin_frac': 1, # 1 for net metering, 0 for no export credit
    'Device_list' : ['Noncontrollable'],
    'resilience_device_list' : ['Noncontrollable'], # Separate device list in case of resilience..conceptualized as devices connected to critical panel 
    'PV': {
        'S_pv': 10, 
        'pf': 0.9,
        'allow_nighttime_pv': False,
    },
    'Battery': {
        'Home_Battery_Nominal_Capacity__kWh': 6,
        'pf': 1,
    },

    # HVAC template
    'Air Source Heat Pump': {
        'AC_Rated_Capacity__kW': 5, #TODO: Generalized HT and AC (capacity and COP) rather than having seperate keys
        'HT_Rated_Capacity__kW': 7,
        'Supplemental Heating Capacity (kW)' : 5,
        'AC_COP': 5.4365,
        'HT_COP': 4.8340,
        'hvac_control': 'setpoint',
        'hvac_setpoint': 'variable_setpoint',
    },
    'Mini Split Heat Pump': {
        'AC_Rated_Capacity__kW': 5, #TODO: Generalized HT and AC (capacity and COP) rather than having seperate keys
        'HT_Rated_Capacity__kW': 7,
        'Supplemental Heating Capacity (kW)' : 5,
        'AC_COP': 5.4365,
        'HT_COP': 4.8340,
        'hvac_control': 'setpoint',
    },
    'Air Conditioning': {
        'AC_Rated_Capacity__kW': 5.53, # AC
        'AC_COP': 3.95,
        'hvac_control': 'setpoint',
        'hvac_setpoint': 'variable_setpoint',
    },
    'Room Air Conditioning':{
        'AC_Rated_Capacity__kW': 5.53, # AC
        'AC_COP': 3.95,
        'space_fraction': 0.2,
        'hvac_control': 'setpoint',
        'hvac_setpoint': 'variable_setpoint',
    },
    'Electric Furnace':{ 
        'hvac_control': 'setpoint',
        'hvac_setpoint': 'variable_setpoint',       
    },
    'Electric Baseboard':{
        'hvac_control': 'setpoint',
        'hvac_setpoint': 'variable_setpoint',
    },
    'Furnace':{
        'hvac_control': 'setpoint',
        'hvac_setpoint': 'variable_setpoint',
    },

    # WH template
    'Heat Pump Water Heater':{
        'Lower_Tank_Volume__litres': 85,
        'Upper_Tank_Volume__litres': 85,
        'Water_Heater_Electrical_Element_Lower__kW': 4.5,
        'Water_Heater_Electrical_Element_Upper__kW': 4.5,
        "Water_heater_Heat_Pump_COP": 2.4,
        "Water_Heater_Heat_Pump__kW": 0.4,
        "Water_Heater_UA__kW_sec": 0.0021,
        "hpwh control": 'both',
    }, 
    'Electric Resistance Water Heater':{
        'Lower_Tank_Volume__litres': 85,
        'Upper_Tank_Volume__litres': 85,
        'Water_Heater_Electrical_Element_Lower__kW': 4.5,
        'Water_Heater_Electrical_Element_Upper__kW': 4.5,
        "Water_Heater_UA__kW_sec": 0.0021,
    },
    'Electric Tankless Water Heater':{        
    },

    'Noncontrollable':{
    },
    'EV':{},
    'EV Uncontrollable':{
    },
    'user_params': None, 
    'Market': True,
    'Debug': True, 
    'save_results': True,
    'properties_file': None, 
    'input_path': None,
    'risk_tolerance': False,
}

season_mode = {
    'Jan': 'Heat',
    'May': 'Auto',
    'Aug': 'Cool'
}


def update_hems_args(house_id):
    hems_args = DEFAULT_HEMS_ARGS
    # Load Master Spreadsheet and get house information
    house_row = master_df.loc[house_id].to_dict()
    print('HOUSE ROW:', house_row)
    hems_args.update(house_row)

    # HVAC
    # if 'ASHP' in [hems_args['Space Heating'], hems_args['Space Cooling']] or \
    #         'MSHP' in [hems_args['Space Heating'], hems_args['Space Cooling']]:
    #     hems_args['Device_list'].append('Air Source Heat Pump')
    #     # Resilience device list
    #     hems_args['resilience_device_list'].append('Air Source Heat Pump')
    #     # hems_args['Air Source Heat Pump']['HT_Rated_Capacity__kW'] = hems_args['Nominal Heating Capacity (W)'] / 1000
    #     # hems_args['Air Source Heat Pump']['AC_Rated_Capacity__kW'] = hems_args['Nominal Cooling Capacity (W)'] / 1000
    #     # hems_args['Air Source Heat Pump']['Supplemental Heating Capacity (kW)'] = hems_args[
    #     #                                                                               'Supplemental Heating Capacity (W)'] / 1000
    #     # hems_args['Air Source Heat Pump']['hvac control'] = 'setpoint'
    #     # hems_args['Air Source Heat Pump']['AC_T_sp_desired'] = 22.222

    # if hems_args['Space Heating'] == 'Furnace':
    #     hems_args['Device_list'].append('Furnace')

    # if hems_args['Space Cooling'] == 'AC':
    #     # Resilience device list
    #     hems_args['resilience_device_list'].append('Air Conditioning')
    #     hems_args['Device_list'].append('Air Conditioning')
    #     hems_args['Air Conditioning']['AC_Rated_Capacity__kW'] = hems_args['Nominal Cooling Capacity (W)'] / 1000

    # # Water Heating
    # if hems_args['Water Heating'] == 'ERWH':
    #     hems_args['Device_list'].append('Electric Resistance Water Heater')
    # if hems_args['Water Heating'] == 'HPWH':
    #     hems_args['Device_list'].append('Heat Pump Water Heater')  # ERWH, HPWH, Gas Tankless

    # # PV
    # if hems_args['PV_Profile_ID'] not in ["", 'Flat']:
    #     hems_args['Device_list'].append('PV')
    #     # Resilience device list
    #     hems_args['resilience_device_list'].append('PV')
    #     if not BTO_sim:
    #         hems_args['PV']['S_pv'] = hems_args['PV Size (kW)']
    #     if allow_nighttime_pv:
    #         hems_args['PV']['allow_nighttime_pv'] = True

    # # Battery
    if hems_args['Battery (kWh)'] > 0:
        hems_args['Device_list'].append('Battery')
        # Resilience device list
        hems_args['resilience_device_list'].append('Battery')
        hems_args['Battery']['Home_Battery_Nominal_Capacity__kWh'] = hems_args['Battery (kWh)']

    # # EV
    # if hems_args['EV'] != 'None':
    #     ev_type, ev_mileage = tuple(hems_args['EV'].split('-'))
    #     hems_args['Device_list'].append('EV')
    #     hems_args['EV'] = {
    #         'vehicle_type': ev_type,
    #         'charging_level': 'Level1',
    #         'mileage': float(ev_mileage),
    #     }

    # User Preferences
    if 'Preference Scenario' in hems_args.keys():
        if hems_args['Preference Scenario'] != 'None':
            preference_df = pd.read_csv(os.path.join(input_path, 'foresee', 'user_preference.csv'), index_col=0)
            hems_args['user_params'] = preference_df[house_row['Preference Scenario']].to_dict()

    # house model properties file
    # if basalt_sim or boston_sim or dallas_sim or sacramento_sim:
    #     rc_file = str(hems_args['OCHRE Properties File']) + '_rc_model.properties'
    # else:
    #     rc_file = hems_args['DOOM_filename']
    # hems_args['properties_file'] = os.path.join(ochre_scenario_path, 'n'+str(hems_args['OCHRE Properties File']) , rc_file)


    # if basalt_sim or boston_sim or dallas_sim or sacramento_sim:
    #     resstock_id = hems_args['ResStock ID']
    #     occupants = hems_args['Num_Occupant']
    #     resstock_path = os.path.join(resstock_input_path, f'schedules_{occupants}', f'bldg{int(resstock_id):07d}')
    #     hems_args['schedule_file'] = os.path.join(resstock_path, 'schedules.csv')

    return hems_args


class HEMS(Agent):
    def __init__(self, house_id, **kwargs):
        self.house_id = house_id
        self.horizon = hems_horizon
        self.freq = freq_hems

        self.hems_args = None
        self.foresee = None
        self.forecast_data = None

        self.dispatch = None
        self.forecast = {'forecast_data_flag': True}
        self.previous_timestamp = 0
        self.status = None
        self.usage = {}
        self.band = {}

        super().__init__('HEMS_' + house_id, **kwargs)

    def initialize(self):
        self.hems_args = update_hems_args(self.house_id)
        horizon_steps = int(self.horizon // self.freq)
        freq_minutes = int(self.freq // timedelta(minutes=1))
        self.foresee = Foresee(horizon_steps, freq_minutes, **self.hems_args)
        self.forecast_data = self.setup_forecast(**self.hems_args)

    def setup_forecast(self, **kwargs):
        year_start = datetime(start_time.year, 1, 1)
        # currently only works at 15 min intervals
        assert self.freq == timedelta(minutes=15)

        # load OCHRE output and schedule files from baseline simulation
        baseline_ochre_folder = os.path.join(baseline_output_path, 'Ochre', 'House_' + self.house_id)
        df_ochre, _, _ = Analysis.load_ochre(baseline_ochre_folder, self.house_id, combine_schedule=True, remove_tz=False)
        print(df_ochre)

        keep_cols = ['Total Electric Power (kW)', 'Total Reactive Power (kVAR)'
                     'Other Electric Power (kW)', 'Other Reactive Power (kVAR)', 'PV Electric Power (kW)',
                     'EV SOC (-)', 'EV Start Time', 'EV End Time',
                     'Temperature - Outdoor (C)', 'GHI (W/m^2)', 'Showers (L/min)', 'Sinks (L/min)',
                     'Clothes washer (L/min)', 'Dishwasher (L/min)', 'Baths (L/min)', 'Mains Temperature (C)', 'Occupancy (Persons)'
                     ]
        df_ochre = df_ochre.loc[:, [col for col in keep_cols if col in df_ochre.columns]]

        # TODO:HARDCODING FOR Battery AGGREGATOR
        df_ochre['Other Electric Power (kW)'] = df_ochre['Total Electric Power (kW)']
        # df_ochre['Other Reactive Power (kVAR)'] = df_ochre['Total Reactive Power (kVAR)']

        # PV
        missing = pd.Series([0] * len(df_ochre), index=df_ochre.index)
        df_ochre['PV Electric Power (kW)'] = df_ochre.get('PV Electric Power (kW)', missing).abs()

        # EV
        if ev_aggregator:
            if 'EV' in self.hems_args and self.hems_args['EV'] != 'None' and len(self.hems_args['EV'])>0:
                def ev_availability(df_row):
                    if pd.to_datetime(df_row['EV Start Time']) <= df_row.name <= pd.to_datetime(df_row['EV End Time']):
                        return 1
                    else:
                        return 0

                df_ochre['EV_availability'] = df_ochre.apply(lambda x: ev_availability(x), axis=1)

                def initial_soc(df_row, df):
                    return df.loc[df_row['EV Start Time'], 'EV SOC (-)']

                df_ochre['EV initial SOC'] = df_ochre.apply(lambda x: initial_soc(x, df_ochre), axis=1)

            if 'EV' in self.hems_args and self.hems_args['EV'] != 'None' and len(self.hems_args['EV'])>0:
                df['EV_availability'][df['EV_availability'] < 1] = 0

        # combine and resample to 15 min
        df_ochre.to_csv(os.path.join(self.result_path,'df_ochre.csv'))
        df_ochre.index = pd.to_datetime(df_ochre.index)
        df = df_ochre.resample(freq_hems).mean()
        


        # # house model coeff
        # house_model_name = kwargs['OCHRE Properties File']
        # coeff_file = os.path.join(input_path, 'foresee', 'house_models', location, f'{house_model_name}_coef.csv')
        # coeff = pd.read_csv(coeff_file, names=[1, 2, 3, 4, 5, 6])
        # coeff = coeff.append(pd.Series(coeff.iloc[len(coeff) - 1]), ignore_index=True)
        # coeff.index = pd.date_range(start=year_start, periods=len(coeff), freq='MS')
        # coeff["all_coeff"] = [coeff.iloc[i].values for i in range(len(coeff))]
        # coeff = coeff.drop(labels=[1, 2, 3, 4, 5, 6], axis=1)
        # coeff = coeff.resample(freq_hems).pad()
        # df['foresee_house_model_coef'] = coeff.loc[df.index, 'all_coeff']

        # # house model disturbance
        # disturbance_file = os.path.join(input_path, 'foresee', 'house_models', location, f'{house_model_name}_dist.csv')
        # dist = pd.read_csv(disturbance_file, usecols=['disturbance'])['disturbance']
        # dist.index = pd.date_range(start=year_start, periods=len(dist), freq=freq_hems)
        # df['foresee_house_model_dist'] = dist

        # load TOU data
        tou_file = os.path.join(input_path, 'foresee', "pge_utility_rate.csv")
        #tou = pd.read_excel(tou_file, sheet_name=location, usecols=["Utility Rate"])["Utility Rate"]
        tou = pd.read_csv(tou_file)
        tou = tou['util_rate__USD']
        tou.index = pd.date_range(start=year_start, periods=len(tou), freq='H')
        df['util_rate__USD'] = tou.resample(freq_hems).pad()
        # df['util_rate__USD'] = 0.1

        # for facilitating annual simulation
        extra_day_df = pd.DataFrame(df[:96].values, columns=df.columns)
        extra_day_df.index = pd.date_range(start=end_time, periods=len(extra_day_df), freq=freq_hems)
        # df = df.append(extra_day_df)
        df = pd.concat([df, extra_day_df])

        # save forecast data
        # if debug:
        df.to_csv(os.path.join(self.result_path, 'forecast.csv'))

        # remove data before start date and add 1 day after end date
        df = df.loc[start_time:(end_time + timedelta(days=1)), :]
        df.to_csv(os.path.join(self.result_path,'forecast_resilience.csv'))
        return df

    def setup_pub_sub(self):
        topic_to_house = "ctrlr_controls_to_house_{}".format(self.house_id)
        self.register_pub("controls", topic_to_house, "String")

        topic_from_house_status = "house_status_to_ctrlr_{}".format(self.house_id)
        self.register_sub("status", topic_from_house_status)

        if include_aggregator:
            topic_to_aggre_flex = "ctrlr_{}_band_to_aggre".format(self.house_id)
            self.register_pub("flex_band_to_agg", topic_to_aggre_flex, "String")

            topic_from_aggre_dispatch = "aggre_dispatch_to_ctrlr_{}".format(self.house_id)
            self.register_sub("dispatch", topic_from_aggre_dispatch)

        # if include_utility:
        #     topic_from_utility = "tou_pricing"
        #     self.register_sub("tou", topic_from_utility)

    def setup_actions(self):
        if include_aggregator:
            self.add_action(self.make_flex_band, 'Make Flex Band', freq_hems, offset_hems_to_agg)
        self.add_action(self.send_controls, 'Send Controls', freq_hems, offset_hems_to_house)
        self.add_action(self.save_results, 'Save Results', freq_save_results, offset_save_results)

    def get_dispatch(self):
        if include_aggregator:
            self.dispatch = self.fetch_subscription("dispatch")

    def get_house_status(self):
        self.status = self.fetch_subscription("status")
        print("House status:", self.status)

    def update_forecast(self, horizon, offset):
        # Note: pandas datetime slicing is inclusive!
        start = self.current_time - offset
        end = start + horizon - freq_hems

        forecast = self.forecast_data.loc[start: end]


        self.forecast = {col: series.to_list() for col, series in forecast.items()}
        self.forecast['forecast_data_flag'] = True
        #  self.forecast['foresee_house_model_coef'] = self.forecast['foresee_house_model_coef'][0]
        self.forecast['current_time'] = self.current_time.hour + self.current_time.minute / 60 + 1
        # self.forecast['hvac_mode'] = season_mode[month] if not annual_sim else 'Auto'

        # self.print_log(self.forecast)

    def make_flex_band(self):
        # kafka message exchange for lab house
        if HIL_sim and self.house_id == 'n47_c_1':
            # get flexband from lab foresee
            default_band = {
                'Pg': [],
                'Qg': [],
                'Pg_low': [],
                'Qg_low': [],
                'Pg_high': [],
                'Qg_high': [],
            }
            band = self.receive_kafka_message(default_band)
            # forward the band to aggregator
            self.publish_to_topic("flex_band_to_agg", band)
            self.received_hil_flexband[self.current_time] = band

        else:
            self.get_house_status()
            if self.status is not None:
                self.update_forecast(self.horizon, offset_hems_to_agg)
                band_start_time = datetime.now()
                if include_aggregator:
                    if resilience_mode: # and self.current_time in critical_times:
                        self.print_log("RESILIENCE FLEXBAND MODE")
                        band = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                      opt_mode='resilience_flexband', scenario=hems_scenario)
                    else:
                        band = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                      opt_mode='flexband', scenario=hems_scenario)
                    self.print_log("timetaken for band optimi", datetime.now() - band_start_time)
                    self.publish_to_topic("flex_band_to_agg", band)
            else:
                self.print_log("No status received from house, skipping optimization")

    def send_controls(self):
        print("IN SENDING CONTROLS")
        self.get_dispatch()

        if include_aggregator and HIL_sim and self.house_id == 'n47_c_1':
            # get dispatch signal from aggregator and forward to lab foresee via kafka
            local_timezone = tzlocal.get_localzone()
            data = {
                "timestamp": time.time(),
                "time": datetime.utcfromtimestamp(time.time()).replace(tzinfo=pytz.utc) \
                    .astimezone(local_timezone).strftime('%Y-%m-%d %H:%M:%S'),
                "dispatch": self.dispatch
            }
            key = {"key": "aggregator_request"}
            self.send_kafka_messge(key, data)
        else:
            if not include_aggregator:
                # only get house status if it hasn't been received in make_flex_bands
                self.get_house_status()
            if self.status is not None:
                self.update_forecast(aggregator_horizon, offset_hems_to_house)
                if include_aggregator and self.dispatch is not None:
                    print("AGGREGATOR")
                    rev_start_time = datetime.now()
                    # Reverse Optimization of foresee with Aggregator

                    if resilience_mode: # and self.current_time in critical_times:
                        # Seperate mode for resilience as foresee needs seperate flag for resilience use case
                        self.print_log("RESILIENCE REVERSE MODE")
                        out = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                     opt_mode='resilience_reverse',
                                                     IP_FORESEE_OP_AGGREGATOR=self.dispatch, scenario=hems_scenario)
                    else:
                        out = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                     opt_mode='reverse',
                                                     IP_FORESEE_OP_AGGREGATOR=self.dispatch, scenario=hems_scenario)
                        self.print_log("timetaken for rev optimi", datetime.now() - rev_start_time)
                else:
                    # Normal Optimization of foresee
                    if BTO_sim:
                        occupancy = True
                        out = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                     opt_mode='normal',
                                                     occupancy=occupancy, scenario='BTO')
                    else:
                        if resilience_mode:
                            out = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                         opt_mode='resilience_normal', scenario=hems_scenario)
                            self.print_log('Controls before:', out[0])
                            if 'Battery' in out[0].keys(): 
                                self.print_log("Removing Battery Controls FROM FORESEE")
                                out[0]['Battery'] = {}
                            if 'HVAC Heating' not in out[0].keys(): 
                                self.print_log("Adding Heating setpoint for resilience operation for gas equip")
                                out[0]['HVAC Heating'] = {}
                                out[0]['HVAC Heating']['Setpoint'] = (self.forecast_data['HT_T_Setpoint'].unique()[0] - 2.778)
                            self.print_log('Controls after:', out[0])

                        else:
                            out = self.foresee.Optimizer(self.status, IP_FORESEE_OP_FORECAST=self.forecast,
                                                         opt_mode='normal',
                                                         scenario=hems_scenario)

                controls = out[0]
                print("CONTROLS IN HEMS:", controls)
                self.publish_to_topic("controls", controls)

    def save_results(self):
        self.foresee.export_results(self.house_id, self.result_path)
        if HIL_sim and self.house_id == 'n47_c_1':
            received_hil_flexband_df = pd.DataFrame(self.received_hil_flexband.items())
            received_hil_flexband_df.to_csv(os.path.join(output_path, 'Foresee', "flexband_47c_1.csv"))

    def send_sim_control_to_lab(self, control):
        # sends start and stop signal to lab foresee for co-ordination
        if HIL_sim and self.house_id == 'n47_c_1':
            local_timezone = tzlocal.get_localzone()
            data = {
                "timestamp": time.time(),
                "time": datetime.utcfromtimestamp(time.time()).replace(tzinfo=pytz.utc) \
                    .astimezone(local_timezone).strftime('%Y-%m-%d %H:%M:%S'),
                'simulation_start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'action': control

            }
            key = {"key": "action"}
            self.send_kafka_messge(key, data)

    def finalize(self):
        if HIL_sim and self.house_id == 'n47_c_1':
            self.send_sim_control_to_lab('stop')
            self.save_results()
            # self.python_producer.flush()
            self.python_consumer.disconnect()
        else:
            self.save_results()

        super().finalize()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        house_id = str(sys.argv[1])
        addr = str(sys.argv[2])
        agent = HEMS(house_id, broker_addr=addr)
    elif len(sys.argv) == 2:
        house_id = str(sys.argv[1])
        agent = HEMS(house_id, debug=True, run_helics=False)
    else:
        raise Exception('Must specify House ID')

    agent.simulate()
