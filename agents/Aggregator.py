import numpy as np

from constants import *
from agents import Agent
#from agents/Control import Agg_Util
import agents.Control.SF_v1 as SF_control

#from ochre import Dwelling
# from dwelling_model import Dwelling

# load keys for sending data
#signals = pd.read_csv(aggregator_signal_filename)
#from_dss_keys = signals['DSS to microgrid'][0:11].to_list() ##from_dss_keys = signals['DSS to microgrid'].to_list()
#from_house_keys = signals['House to Aggregator'][0:10].to_list() ##from_dss_keys = signals['DSS to microgrid'].to_list()
#to_house_keys = signals['Aggregator to House'][0:10].to_list()


class Aggregator(Agent):
    def __init__(self, **kwargs):
    
        print('_init')
        
        self.t_receiver = None
        self.t_sender = None
        global aggregator_flag
        aggregator_flag=0
        super().__init__('Aggregator', **kwargs)
        
        

    def initialize(self):
    
        print('initialize')
        
        global file1
        
        file1 = open("Agg_record.txt","w") 
        file1.writelines(['Agg Code Class \n'])  
        file1.close()
        
        self.house_forsee=[]
        print('start initialization')
        self.controls=SF_control.Controls() #check control comment out
        print('end initialization')
        
        file1 = open("Agg_record.txt","a") 
        file1.writelines(['check 1 \n'])  
        file1.close()
        
        #for i in range(0,len(master_df)):
        print(master_df.index)
        for i in master_df.index:
            print(i)
            print(master_df['Battery (kWh)'][i])
        
            if master_df['Battery (kWh)'][i]!=0:
                #self.house_forsee.append(master_df['House_ID'][i])
                self.house_forsee.append(i)
                
        print(self.house_forsee)
        print('if include')
        print(include_aggregator)
        
        file1 = open("Agg_record.txt","a") 
        file1.writelines(['check 2 \n'])  
        file1.close()
        
        
        if include_aggregator:
            # Receiver for microgrid-to-feeder
            self.t_receiver = {}
            # Sender for feeder-to-microgrid
            self.t_sender = {}
           
        
            

    def setup_pub_sub(self):
    
        print('setup_pub_sub')
        
        self.register_sub('feeder_to_aggregator', default={})
        self.register_sub('feeder_to_aggregator_PV', default={})
        self.register_sub('feeder_to_aggregator_Cap', default={})
        
        for house in self.house_forsee:
            topic_from_house = "ctrlr_{}_band_to_aggre".format(house)
            #self.register_sub("flex_band_to_agg", topic_from_house)
            self.register_sub(house, topic_from_house)

            topic_to_house = "aggre_dispatch_to_ctrlr_{}".format(house)
            #self.register_pub("dispatch", topic_to_house)
            self.register_pub(house, topic_to_house)

    def setup_actions(self):
    
        print('setup_actions')
        self.add_action(self.send_to_aggregator, 'Send to aggregator', freq_aggregator, offset_aggregator_send)
        self.add_action(self.receive_from_aggregator, 'Receive from aggregator', freq_aggregator, offset_aggregator_receive)
        #self.add_action(self.save_results, 'Save Results', freq_save_results)

    def send_to_aggregator(self):
    
        print('send_to_aggregator')
        # Get subscriptions
        #global from_feeder
        #from_house = self.fetch_subscription('house_to_aggregator')
        #from_feeder = self.fetch_subscription('feeder_to_aggregator')
        #from_feeder_PV = self.fetch_subscription('feeder_to_aggregator_PV')
        #from_feeder_Cap = self.fetch_subscription('feeder_to_aggregator_Cap')
        #from_feeder2 = self.fetch_subscription('feeder_to_microgrid2')
        #print(from_house)
        #print(from_feeder2)
        
        global file1
        
        file1 = open("Agg_record.txt","a") 
        file1.writelines(['check 3 \n'])  
        file1.close()
        
        from_house_data={}
        
        for house in self.house_forsee:
            #data = self.fetch_subscription(house)
            topic_from_house = "ctrlr_{}_band_to_aggre".format(house)
            #data = self.fetch_subscription("flex_band_to_agg", topic_from_house)
            #data = self.fetch_subscription("flex_band_to_agg")
            data = self.fetch_subscription(house)
            
            #data = self.fetch_subscription(house)
            from_house_data[house]=data
        
        #print(from_house_data)
        
        data_input_all_agg={}
        #data_input_all_agg.update(from_feeder)
        data_input_all_agg.update(from_house_data)
        
        print(data_input_all_agg)
        data_input_all_uti={}
        #data_input_all_uti['pv']=from_feeder_PV
        #data_input_all_uti['cap']=from_feeder_Cap
        print(data_input_all_uti)
        
        
        #print(data_input_all)
        
        global aggregator_flag
        
        print('run algorithm')
        
        file1 = open("Agg_record.txt","a") 
        file1.writelines(['check 4 \n'])  
        file1.writelines([str(aggregator_flag)+' \n'])  
        file1.close()
        
        if aggregator_flag==0:
            
            print(' ')
            #print('start initialization')
            #self.controls=SF_control.Controls() #check control comment out
            #print('end initialization')
        
        if aggregator_flag>2:
        
            #self.controls.optimize(aggregator_flag,data_input_all_agg,{})
           # self.controls.optimize(cosim_time_stamp=aggregator_flag,from_houses=data_input_all_agg,uncntrl_dict=data_input_all_uti)
            self.controls.optimize(aggregator_flag*4,data_input_all_agg) #comment back for control algorithm #check control comment out
            #data_to_foresee=self.controls.hems_setpoints
            aggregator_flag=aggregator_flag+1
        
            file1 = open("Agg_record.txt","a") 
            file1.writelines(['check 5 \n'])  
            file1.close()
            
            print('algorithm done')
            print(aggregator_flag)
            
            
            for house in self.house_forsee:
                p = self.controls.hems_setpoints[house][0] #check control comment out
                q = self.controls.hems_setpoints[house][1] #check control comment out
                #p=[0]
                #q=[0]
                print(house)
                print(p)
                print(q)
                
                to_house_data={}
                to_house_data['Pagg']=list(p)
                to_house_data['Qagg']=list(q)
                
                topic_to_house = "aggre_dispatch_to_ctrlr_{}".format(house)

                self.publish_to_topic(house, to_house_data)

                print(house)
                print('to_house_data')
                print(to_house_data)
                
        else:
            
            aggregator_flag=aggregator_flag+1
            
        #Agg_Util.Agg_Util_Control
        
        #if include_aggregator:
            # Send data to microgrid-RT
            #to_aggregator = []
            #for key in from_house_keys:
            #    if key == 'Unknown':
            #        to_aggregator.append(0)
            #    else:
            #        to_aggregator.append(from_house[key])
           # print(to_aggregator)
            
            
            

        

        # Send data to ADMS, if the agent exists
        #if include_adms:
        #    self.publish_to_topic('microgrid_to_adms', from_feeder)

    def receive_from_aggregator(self):
    
        print('receive_from_aggregator')
        if include_aggregator:
            # Get data from microgrid-RT
            #self.print_log('Receiving from microgrid:', self.t_receiver.data)
            to_house=[]
            #for ii in range(0,len(to_house_keys)):

            #    self.t_receiver[to_house_keys[ii]]=0

                
                
            #to_feeder = {key: self.t_receiver.data[i] for i, key in enumerate(to_dss_keys)}
            to_house=self.t_receiver
        else:
            to_house = {}  # no data sent back to feeder

        # Send data to feeder agent
        #self.publish_to_topic('aggregator_to_house', to_house)

    def finalize(self):
    
        print('finalize')
        super().finalize()
        #if microgrid_connected:
        #    self.t_receiver.sock.close()
        #    self.t_sender.sock.close()





if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        addr = str(sys.argv[1])
        agent = Aggregator(broker_addr=addr)
    else:
        agent = Aggregator(run_helics=False)
    agent.simulate()