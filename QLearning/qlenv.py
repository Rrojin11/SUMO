import os


import os
import sys
import time
from typing import DefaultDict
import numpy as np

from xml.etree.ElementTree import parse
from collections import defaultdict

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    print(tools)
    sys.path.append(tools)
    try:
        import traci
    except ImportError:
        raise EnvironmentError("Declare SUMO_HOME environment")
else:
    sys.exit('Declare environment variable "SUMO_HOME"')

class qlEnv():
    def __init__(self, sumoBinary, net_file: str,det_file: str, cfg_file: str, alldets: list, edgelists: list, dict_connection, veh:str, endpoint: list,use_gui: bool = True,
            begin_time: int =0, num_seconds:int = 2000, max_depart_delay:int = 10000):
        
        self.sumoBinary = sumoBinary
        self.net = net_file
        self.det = det_file
        self.sumocfg = cfg_file
        self.alldets = alldets
        self.edgelists = edgelists
        self.use_gui = use_gui

        self.veh = veh
        self.endpoint = endpoint

        self.episode = 0 # # of run time 
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.max_depart_delay = max_depart_delay

        self.action_space = [0,1,2]
        self.n_actions = len(self.action_space)
        self.dict_connection = dict_connection

        self.sumo =traci
        
 
    def start_simulation(self):
        sumo_cmd = [self.sumoBinary,
            '-c', self.sumocfg,
            '--max-depart-delay', str(self.max_depart_delay)]
        
        self.sumo.start(sumo_cmd)
        '''
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            traci.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
        '''
        
        
    def reset(self):
        if self.episode!=0: 
            self.sumo.close()

        self.episode+=1
        self.start_simulation()

        #vehicle  생성
        self.sumo.route.add("rou1", ["E19", "E0", "E1","E2"]) #default route
        self.sumo.vehicle.add(self.veh, "rou1")
        self.sumo.simulationStep()
        
        curedge = self.get_curedge(self.veh)
        return curedge

    def get_curedge(self,veh):
        curlane = self.sumo.vehicle.getLaneID(veh)
        curedge = self.sumo.lane.getEdgeID(curlane)
        return curedge

    def get_done(self,curedge):
        done = False
        if curedge in self.endpoint:
            done = True
        return done
    
    def get_reward(self, nextedge):
        reward = 0
        if nextedge=='E6' or nextedge=='E2' or nextedge=='E13' or nextedge=='-E19':
            reward = -100
        elif nextedge=='E20':
            reward = 500
        return reward
    
    def get_nextedge(self,curedge, action):
        nextedge = self.dict_connection[curedge][action]
        return nextedge

    def get_vehiclecount(self):
        return self.sumo.vehicle.getIDCount()
  
    def step(self, curedge, nextedge):
        
        beforeedge = curedge #비교해서 변하면 고를려고!

        done = self.get_done(curedge)
        reward = self.get_reward(nextedge)

        if done:
            return reward, done
        
        self.sumo.vehicle.changeTarget(self.veh,nextedge) #차량 움직여!
        
        while self.sumo.simulation.getMinExpectedNumber() > 0:
            curedge = self.get_curedge(self.veh)
            done = self.get_done(curedge)
            if done:
                break  
            self.sumo.simulationStep() 
            if curedge in self.edgelists and curedge !=beforeedge : #변했네!! 그럼 이제 다음 꺼 고르러 가야지
                break

        return reward, done    

    
    
    