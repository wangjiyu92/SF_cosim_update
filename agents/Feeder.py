try:
    from constants import *
except ImportError:
    import sys
    import os

    path = os.path.join(os.path.dirname(__file__), os.pardir)
    sys.path.append(path)
    from constants import *
from agents import Agent
#from opendss_wrapper import OpenDSS

from core.OpenDSS import OpenDSS, save_linear_power_flow
import numpy as np

class Feeder(Agent):
    def __init__(self, **kwargs):
        # Set up houses
        self.houses = feeder_loads
        kwargs['result_path'] = feeder_results_path
        print('_feeder loads')
        #print(feeder_loads)
        print('_self.houses')
        #print(self.houses)

        self.feeder = None
        self.df_loadshapes = None
        self.df_pvshapes = None
        self.all_load_names = []
        self.all_bus_names = []
        self.utility_pv = {'PV_utility_1': {'p': 1, 'q': 2},
                'PV_utility_2': {'p': 2, 'q': 4},
         }   
        
        self.counter = 1
        global tflag
        tflag=0
        

        super().__init__('Feeder', **kwargs)

    def initialize(self):
        # create feeder instance
        redirects = [master_dss_file]
        #print(redirects)
        #print(str(redirects))
        #print("Compile " + redirects)
        self.feeder = OpenDSS(redirects[0], freq_house, start_time)

        self.all_load_names = self.feeder.get_all_elements(element='Load').index
        self.all_bus_names = self.feeder.get_all_buses()
        self.master_df_all = pd.read_excel(ms_file)
        global PS_all
        PS_all={}
        global pvinfo
        PS_all['PV']=pd.read_csv(PVshape_file,header=None)[0]
        pvinfo=pd.read_csv(pvinfo_filename)
        
        
        global LS_all
        LS_all={}
        for i in range(0,len(self.all_load_names)):

            ls_path=os.path.join(feeder_input_path,'model_SF', "LS" ,str(self.all_load_names[i]).lower()+'.csv')
            LS_all[self.all_load_names[i].upper()]=pd.read_csv(ls_path,header=None)[0]
        
        print("check A")
        # update 1
        # self.dfloadshapes_sel = pd.read_csv(loadshapes_sel_file)
        # loadshapes_sel = self.dfloadshapes_sel.iloc[:, 2].tolist()
        # loadshape_vars = dict((key, 0.0) for key in loadshapes_sel)
        # self.df_loadshapes = pd.DataFrame(columns=loadshape_vars.keys())
        # for key, val in loadshape_vars.items():
        #     key_loc_mid=list(loadshape_vars).index(key)
        #     #print(key)
        #     filename = os.path.join(feeder_input_path, "UseCase5",self.dfloadshapes_sel['Type'][key_loc_mid], str(key) + ".csv")
        #     self.df_loadshapes[str(key)] = pd.read_csv(filename, header=None).values.tolist()
        # #1124 lsname = os.path.join(input_path, 'df_loadshapes.csv')
        # #self.loadinfo = pd.read_csv(loadinfo_filename)
        # self.pvinfo = pd.read_csv(pvinfo_filename)
        
        # pvshapes_sel = self.pvinfo.iloc[:, 2].tolist()
        # pvshape_vars = dict((key, 0.0) for key in pvshapes_sel)
        # self.df_pvshapes = pd.DataFrame(columns=pvshape_vars.keys())
        # for key, val in pvshape_vars.items():
        #     key_loc_mid=list(pvshape_vars).index(key)
        #     #print(key)
        #     filename = os.path.join(feeder_input_path, "UseCase5", "PVShape.csv")
        #     self.df_pvshapes[str(key)] = pd.read_csv(filename, header=None).values.tolist()

        # set up results files
        self.initialize_results('community_P')
        self.initialize_results('community_Q')
        self.initialize_results('community_V')
        self.initialize_results('summary')
        self.initialize_results('all_voltages')
        self.initialize_results('all_bus_voltages')
        self.initialize_results('all_powers')
        self.initialize_results('fh_voltage')

    def setup_pub_sub(self):
        print('_DOOM subscriptions')
        #print(self.houses)
        
        self.register_pub('feeder_to_aggregator')
        self.register_pub('feeder_to_aggregator_PV')
        self.register_pub('feeder_to_aggregator_Cap')
        
        for house in self.houses:
            topic_from_house = "house_{}_power_to_feeder".format(house)
            self.register_sub(house, topic_from_house)

            topic_to_house = "feeder_vtg_to_house_{}".format(house)
            self.register_pub(house, topic_to_house)

    def setup_actions(self):
        print('Check Setup')
        self.add_action(self.run_feeder, 'Run Feeder', freq_house, offset_feeder_run)
        self.add_action(self.save_results, 'Save Results', freq_save_results, offset_save_results)

    def get_house_power(self, house):
        data = self.fetch_subscription(house)
        if data is not None:
            p = data['P Total']
            q = data['Q Total']
            return p, q
        else:
            return None, None

    def send_voltage_to_houses(self):
        house_voltages = {}
        for house, load in self.houses.items():
            # v = self.feeder.get_voltage(load, average=True)
            if load!=0:
                #v = self.feeder.get_voltage(load)
                #if isinstance(v, list) or isinstance(v, tuple): 
                    #print('Type of V:', type(v))
                #    v = np.mean(v)
                # v = self.feeder.get_all_complex(load, element='Load')['VoltagesMagAng'][0]
                v=1
                house_voltages[house] = v

                # Send voltage to house
                self.publish_to_topic(house, {"voltage": v})
        return house_voltages

    def add_community_results(self, house_powers, house_voltages):
        # save powers and voltages to results files

        p = {house: house_powers.get(house, (None, None))[0] for house in self.houses}
        self.add_to_results('community_P', p, remove_seconds=True)

        q = {house: house_powers.get(house, (None, None))[1] for house in self.houses}
        self.add_to_results('community_Q', q, remove_seconds=True)

        v = {house: house_voltages.get(house, 0) for house in self.houses}
        self.add_to_results('community_V', v, remove_seconds=True)

        # return total p and q
        try:
            return sum(p.values()), sum(q.values())
        except TypeError:
            return 0, 0

    def run_feeder(self):
        house_powers = {}
        # Get house powers and update feeder
        print('_houses')
        #print(self.houses)
        

        global tflag
        global LS_all

        
        
        print('Start read csv')
        print(datetime.now())
        

        global PS_all
        global pvinfo
        to_aggregator_PV={}

        to_aggregator_Cap={}


            
        print(datetime.now())        

        tflag=tflag+1
        print(tflag)

        
        #print('_utility pvs')
        #print(self.utility_pv)

        self.counter += 1
        # if self.counter == 6540:

        house_voltages = self.send_voltage_to_houses()
        print(house_voltages)
        if house_voltages:
            #print(house_voltages.values())
            v_max = max(house_voltages.values())
            v_min = min(house_voltages.values())
            self.print_log('Community Voltage Range: {} - {}'.format(v_min, v_max))
        
        
            # Add results for each house
        p_total, q_total = self.add_community_results(house_powers, house_voltages)



if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        addr = str(sys.argv[1])
        agent = Feeder(broker_addr=addr)
    else:
        #agent = Feeder(debug=True, run_helics=False)
        agent = Feeder(run_helics=False)

    agent.simulate()
