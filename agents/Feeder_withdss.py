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
                v = self.feeder.get_voltage(load)
                if isinstance(v, list) or isinstance(v, tuple): 
                    #print('Type of V:', type(v))
                    v = np.mean(v)
                # v = self.feeder.get_all_complex(load, element='Load')['VoltagesMagAng'][0]

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
        
        ld_pw={}
        ld_qw={}
        for i in range(0,len(self.all_load_names)):
            ld_pw[self.all_load_names[i].upper()]=0
            ld_qw[self.all_load_names[i].upper()]=0
            #print(ld_pw)
        
        for house, load in self.houses.items():
            if load!=0:
                p,q=self.get_house_power(house)
                house_powers[house] = p, q
                if p is not None:
                    ld_pw[load.upper()]=ld_pw[load.upper()]+p
                    ld_qw[load.upper()]=ld_pw[load.upper()]+q
        
        global tflag
        global LS_all

        
        #t_day=[0,31,59,90,120,151,181,212,243,273,304,334]
        

        
        print('Start read csv')
        print(datetime.now())
        
        to_aggregator={}
        
        #for i in range(no_of_homes,len(self.master_df_all)):
            
        #    ln_mid=self.master_df_all['Load_name'][i]
            
        #    P_agg_mid=[]
        #    Q_agg_mid=[]
            
        #    for jj in range(0,4):
        #        P_agg_mid.append(LS_all[ln_mid.upper()][t_day[month-1]*24+(start_date-1)*24+(tflag+jj)*15//60]/self.master_df_all['Ld_rpt'][i])
        #        Q_agg_mid.append(LS_all[ln_mid.upper()][t_day[month-1]*24+(start_date-1)*24+(tflag+jj)*15//60]*0.329/self.master_df_all['Ld_rpt'][i])
                
        #    Write_feeder_agg_dic={}
        #    Write_feeder_agg_dic['Pg']=P_agg_mid
        #    Write_feeder_agg_dic['Qg']=Q_agg_mid
        #    Write_feeder_agg_dic['Pg_low']=P_agg_mid
        #    Write_feeder_agg_dic['Qg_low']=Q_agg_mid
        #    Write_feeder_agg_dic['Pg_high']=P_agg_mid
        #    Write_feeder_agg_dic['Qg_high']=Q_agg_mid
            
        #    to_aggregator[self.master_df_all['House_ID'][i].lower()]=Write_feeder_agg_dic
  
        
        for i in range(0,len(self.all_load_names)):
        
            #if self.all_load_names[i].upper() not in list(self.master_df_all['Load_name']):
        
            #    P_agg_mid=[]
            #    Q_agg_mid=[]
                
            #    for jj in range(0,4):
            #        P_agg_mid.append(LS_all[self.all_load_names[i].upper()][t_day[month-1]*24+(start_date-1)*24+(tflag+jj)*15//60])
            #        Q_agg_mid.append(LS_all[self.all_load_names[i].upper()][t_day[month-1]*24+(start_date-1)*24+(tflag+jj)*15//60]*0.329)    
                
            #    Write_feeder_agg_dic={}
            #    Write_feeder_agg_dic['Pg']=P_agg_mid
            #    Write_feeder_agg_dic['Qg']=Q_agg_mid
            #    Write_feeder_agg_dic['Pg_low']=P_agg_mid
            #    Write_feeder_agg_dic['Qg_low']=Q_agg_mid
            #    Write_feeder_agg_dic['Pg_high']=P_agg_mid
            #    Write_feeder_agg_dic['Qg_high']=Q_agg_mid
                
                #to_aggregator['P_'+self.all_load_names[i].upper()]=P_agg_mid
                #to_aggregator['Q_'+self.all_load_names[i].upper()]=Q_agg_mid
                
            #    to_aggregator[self.all_load_names[i].lower()]=Write_feeder_agg_dic
            
            
            
            if ld_pw[self.all_load_names[i].upper()]==0:
                #ls_path=os.path.join(feeder_input_path, "Model_SALMON_Cosim", "LS",str(self.all_load_names[i]).upper()+'.csv')
                #ld_pw[self.all_load_names[i].upper()]=pd.read_csv(ls_path,header=None)[0][tflag]
                #ld_pw[self.all_load_names[i].upper()]=LS_all[self.all_load_names[i].upper()][t_day[month-1]*24+(start_date-1)*24+tflag*15//60]
                ld_pw[self.all_load_names[i].upper()]=LS_all[self.all_load_names[i].upper()][tflag]
            #print(self.all_load_names[i].upper())
            #print(ld_pw)
            #self.feeder.set_power(self.all_load_names[i].upper(), ld_pw[self.all_load_names[i].upper()], ld_qw[self.all_load_names[i].upper()])
            self.feeder.run_command('edit load.'+self.all_load_names[i].upper()+' kW='+str(ld_pw[self.all_load_names[i].upper()])+' kVAR='+str(ld_qw[self.all_load_names[i].upper()]))
        
        global PS_all
        global pvinfo
        to_aggregator_PV={}
        for i in range(0,len(pvinfo)):
            #self.feeder.run_command('edit PVSystem.'+pvinfo['PN'][i].upper()+' pmpp='+str(PS_all['PV'][t_day[month-1]*24+(start_date-1)*24+tflag*15//60]*pvinfo['P'][i]))
            self.feeder.run_command('edit PVSystem.'+pvinfo['PN'][i].upper()+' pmpp='+str(PS_all['PV'][tflag]*pvinfo['P'][i]))
            #agg_PV_mid=[]
            #for jj in range(0,4):
            #    agg_PV_mid.append(PS_all['PV'][t_day[month-1]*24+(start_date-1)*24+(tflag+jj)*15//60]*pvinfo['P'][i]*-1)
                
            #to_aggregator_PV[pvinfo['Bus'][i].lower()]=agg_PV_mid
            
        to_aggregator_Cap={}
        #to_aggregator_Cap['7643815_335_695711_174']=-600
        #to_aggregator_Cap['7643815_335_695711_174']=[-200,-200,-200,-200]
        #to_aggregator_Cap['7643815_335_695711_174']=[-200,-200,-200,-200]
        #to_aggregator_Cap['pprk_cap']=-3000
        #to_aggregator_Cap['PPRK_CAP.2']=[-1000,-1000,-1000,-1000]
        #to_aggregator_Cap['PPRK_CAP.3']=[-1000,-1000,-1000,-1000]
        #to_aggregator_Cap['391']=-3000
        #to_aggregator_Cap['391.2']=[-1000,-1000,-1000,-1000]
        #to_aggregator_Cap['391.3']=[-1000,-1000,-1000,-1000]
        
        #v_reg_mid=np.mean(self.feeder.get_bus_voltage('DELA_BUS_R55'))
        #to_aggregator_Cap['dela_bus_r55']=v_reg_mid
        #v_reg_mid=np.mean(self.feeder.get_bus_voltage('PPRK_SW_R101'))
        #to_aggregator_Cap['pprk_sw_r101']=v_reg_mid
        
        
            # if i<10:
            #     print('edit load.'+self.all_load_names[i].upper()+' kW='+str(ld_pw[self.all_load_names[i].upper()])+' kVAR='+str(ld_qw[self.all_load_names[i].upper()]))
            
            #if i==0:
            #print(str(self.all_load_names[i].upper())+' P:'+str(ld_pw[self.all_load_names[i].upper()])+' Q:'+str(ld_qw[self.all_load_names[i].upper()]))
        #print(str(self.all_load_names[i].upper())+' P:'+str(ld_pw[self.all_load_names[i].upper()])+' Q:'+str(ld_qw[self.all_load_names[i].upper()]))
        
        #for house, load in self.houses.items():
         #   p, q = self.get_house_power(house)
          #  house_powers[house] = p, q
           # print('_house powers: {}, {}'.format(p, q))
            #if p is not None:
             #   self.feeder.set_power(load, p, q)
                
              #  print(str(load)+' P:'+str(p)+' Q:'+str(q))
        #self.feeder.get_all_complex("701_167489_mc", element='Line')
            
        print(datetime.now())        
        #print('End read csv')        
        #global tflag
        tflag=tflag+1
        print(tflag)

        
        #print('_utility pvs')
        #print(self.utility_pv)

        self.counter += 1
        # if self.counter == 6540:
        #     self.feeder.run_command('export powers')

        #for pv, pow in self.utility_pv.items():
        #    self.utility_pv[pv].update({'p': pow['p']*self.counter, 'q': pow['q']*self.counter})
        
        # update 2
        # for ii in range(0,len(self.dfloadshapes_sel)):
        #     lsname=self.dfloadshapes_sel['Name'][ii]
        #     pmid=self.df_loadshapes[str(lsname)][tflag][0]
        #     #self.feeder.set_power(self.loadinfo['LN'][ii], pmid)
        #     #print(['edit load.'+self.loadinfo['LN'][ii]+' kW='+str(pmid)])
        #     self.feeder.run_command('edit load.'+str(self.dfloadshapes_sel['Load'][ii])+' kW='+str(pmid))

        # for ii in range(0,len(self.pvinfo)):
        #     pvname=self.pvinfo['ls'][ii]
        #     pmid=self.pvinfo['P'][ii]*(self.df_pvshapes[str(pvname)][tflag][0]/max(self.df_pvshapes[str(pvname)])[0])
        #     #self.feeder.set_power(self.loadinfo['LN'][ii], pmid)
        #     #print(['edit load.'+self.loadinfo['LN'][ii]+' kW='+str(pmid)])
        #     self.feeder.run_command('edit PVSystem.'+str(self.pvinfo['PN'][ii])+' Pmpp='+str(pmid))
        #     #print('edit Generator.'+str(self.pvinfo['PN'][ii])+' kW='+str(pmid))    
        
        # Run DSS
        self.feeder.run_dss()
        
        # for i in range(0,10):
            
        #     pmid=self.feeder.get_power(self.all_load_names[i].upper(), element='Load')
        #     vmid=self.feeder.get_voltage(self.all_load_names[i].upper(), element='Load')
        #     print(self.all_load_names[i].upper())
        #     print(pmid)
        #     print(vmid)
        
        # Get house voltages and send to houses
        house_voltages = self.send_voltage_to_houses()
        print(house_voltages)
        if house_voltages:
            #print(house_voltages.values())
            v_max = max(house_voltages.values())
            v_min = min(house_voltages.values())
            self.print_log('Community Voltage Range: {} - {}'.format(v_min, v_max))
        
        self.publish_to_topic('feeder_to_aggregator', to_aggregator)
        self.publish_to_topic('feeder_to_aggregator_PV', to_aggregator_PV)
        self.publish_to_topic('feeder_to_aggregator_Cap', to_aggregator_Cap)
        
            # Add results for each house
        p_total, q_total = self.add_community_results(house_powers, house_voltages)

        # Add substation results, including community Loads, PV, and Battery
        substation_data = self.feeder.get_circuit_info()
        substation_data['Community Loads P (MW)'] = p_total / 1000
        substation_data['Community Loads Q (MVAR)'] = q_total / 1000
        self.add_to_results('summary', substation_data, remove_seconds=True)

        # Add results for all load powers and voltages
        #data = self.feeder.get_all_bus_voltages(average=True)
        data = {load: self.feeder.get_voltage(load) for load in self.all_load_names}
        self.add_to_results('all_voltages', data, remove_seconds=True)
        
        data = {bus_mid: self.feeder.get_bus_voltage(bus_mid) for bus_mid in self.all_bus_names}
        self.add_to_results('all_bus_voltages', data, remove_seconds=True)
        
        Fh_v={}
        #Fh_v['Delaware A'] = self.feeder.get_voltage('dela_br_r142', element='Line')[0]
        #Fh_v['Delaware B'] = self.feeder.get_voltage('dela_br_r142', element='Line')[1]
        #Fh_v['Delaware C'] = self.feeder.get_voltage('dela_br_r142', element='Line')[2]
        #Fh_v['OckleyGreen A'] = self.feeder.get_voltage('PPRK_SW_R101', element='Line')[0]
        #Fh_v['OckleyGreen B'] = self.feeder.get_voltage('PPRK_SW_R101', element='Line')[1]
        #Fh_v['OckleyGreen C'] = self.feeder.get_voltage('PPRK_SW_R101', element='Line')[2]
        self.add_to_results('fh_voltage', Fh_v, remove_seconds=True)
        
        
        all_powers = {load: self.feeder.get_power(load, total=True) for load in self.all_load_names}
        # print('_all powers')
        # print(all_powers)
        all_powers = {load + pq: val for load, powers in all_powers.items() for pq, val in
                      zip(('_P', '_Q'), powers)}
        self.add_to_results('all_powers', all_powers, remove_seconds=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        addr = str(sys.argv[1])
        agent = Feeder(broker_addr=addr)
    else:
        #agent = Feeder(debug=True, run_helics=False)
        agent = Feeder(run_helics=False)

    agent.simulate()
