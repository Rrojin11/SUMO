import time

import numpy as np

import traci
import random
import string

from xml.etree.ElementTree import parse
from collections import defaultdict

class gj_dqnEnv():
    def __init__(self, sumoBinary, net_file: str, cfg_file: str, edgelists: list,alldets: list, dict_connection, veh:str, destination:list, state_size: int, action_size: int,lanedict: dict, hopadj : list, use_gui: bool = True,
            begin_time: int =0, num_seconds:int = 3600, max_depart_delay:int = 10000):
        
        self.sumoBinary = sumoBinary
        self.net = net_file
        
        self.sumocfg = cfg_file
       
        self.edgelists = edgelists
        self.alldets = alldets

        self.use_gui = use_gui

        self.veh = veh
        
        self.destination = destination

        self.episode = 0 # # of run time 
        self.begin_time = begin_time
        self.num_seconds = num_seconds
        self.max_depart_delay = max_depart_delay

        self.action_size = action_size
        self.state_size = state_size
        
        self.dict_connection = dict_connection
        self.lanedict = lanedict
        self.hopadj = hopadj

        self.sumo =traci
        
    def start_simulation(self):
        sumo_cmd = [self.sumoBinary,
            '-c', self.sumocfg,
            '--max-depart-delay', str(self.max_depart_delay),
            '--collision.action', 'none']
        
        self.sumo.start(sumo_cmd)
        #테스트중 traffic randomroute
        ## 모든 edge 별 length & speed limit 정보 얻어오기
        self.dict_edgelengths, self.list_edgelengths = self.get_edgelengths()
    
        self.dict_edgelimits = self.get_edgelimits()
        destlane = [i+'_0' for i in self.destination]
        self.destCord = self.sumo.lane.getShape(destlane[0])[0]
        #print('self.destCord: ',self.destCord)
        
     
        
    def sumo_step(self):
        self.sumo.simulationStep()

    def sumoclose(self):
        self.sumo.close()
        
    def reset(self):
        
        self.episode+=1 
        self.start_simulation()
        
        curlane = self.get_curlane(self.veh)
        while curlane=='':
            curlane = self.get_curlane(self.veh)
            self.sumo.simulationStep()

        curedge = self.get_curedge(curlane)
        
        state = self.get_state(self.veh,curedge, curlane)
            
        return state
        

    def get_curlane(self,veh):
        curlane = self.sumo.vehicle.getLaneID(veh)
        return curlane

    def get_curedge(self,curlane): ##0217목 8pm 수정 오류 확인해봐야함  curlane ''뜨는거!!!!!!
        curedge = self.sumo.lane.getEdgeID(curlane)
        return curedge

    def get_done(self,curedge):
        done = False
        if curedge in self.destination:
            done = True
        return done
    
    def get_RoadID(self, veh):
        return traci.vehicle.getRoadID(veh)

    def get_reward(self, curedge, curlane, nextedge):
        reward = 0
        #reward = traveling time of curedge
        det = 'D'+curlane
        
        num_veh = self.get_numVeh(det)
        if num_veh ==1: # 자기자신  #length/speedlimit 
            traveltime = self.dict_edgelengths[curedge] / self.dict_edgelimits[curedge]
        else:
            #print('err1 get_reward num_veh 확인용 : ',num_veh) 
            traveltime = self.sumo.edge.getTraveltime(curedge) #(length/mean speed).
       
        reward = -traveltime
        return reward
    
    def get_nextedge(self, curedge, action):
        nextedge = self.dict_connection[curedge][action]
        return nextedge

    def get_numVeh(self, det):
        num_veh = self.sumo.lanearea.getLastStepVehicleNumber(det)
        return num_veh
    
    def get_edgelengths(self):
        dict_edgelengths = defaultdict(float)
        list_edgelengths = []
        dict_edgelengths.update((k,0.0 ) for k in self.edgelists)
        for edge in self.edgelists:
            lane = edge + '_0'
            length = self.sumo.lane.getLength(lane)
            dict_edgelengths[edge]=length
            list_edgelengths.append(length)
        return dict_edgelengths, list_edgelengths

    def get_edgelimits(self):
        dict_edgelimits = defaultdict(float)
        dict_edgelimits.update((k,15.0) for k in self.edgelists)
        dict_edgelimits['E7'] = 10.0
        dict_edgelimits['-E7'] = 10.0
        return dict_edgelimits

    def get_state(self, veh, curedge, curlane): #state = [hop0's(vehicle number, avg speed, length), hop1's, hop2's, hop3's, origin x, y, destination x,y] = (16,1)
        state = []
        curidx = self.lanedict[curedge]
        det = 'D'+curlane
       
        state.append(self.get_numVeh(det)) #vehicle number
        state.append(self.sumo.edge.getLastStepMeanSpeed(curedge)) #avg speed
        state.append(self.sumo.lane.getLength(curedge+'_0')) # length
        
        for hop in range(1,4):
            avgVeh, avgLength, avgSpeed = 0,0,0
            cnt = 0
            for i in np.where(self.hopadj[curidx]==hop)[0]: 
                cnt+=1
                edge = self.lanedict[i]
                det = 'D'+curlane
                
                avgVeh+=self.get_numVeh(det) #vehicle number
                avgLength += self.sumo.lane.getLength(curedge+'_0') # length
                avgSpeed += self.sumo.edge.getLastStepMeanSpeed(edge) #avg speed
            hopFeature = [avgVeh, avgLength, avgSpeed]
            if cnt>1:
                hopFeature = [x/cnt for x in hopFeature]
            state.extend(hopFeature)
        curlane = curedge+'_0' #curedge좌표
        curCord= self.sumo.lane.getShape(curlane)[0]
        state.extend(list(curCord))
        state.extend(list(self.destCord))#목적지 좌표
        
        return state

    def get_nextstate(self, veh, nextedge):
        next_state = []
        curidx = self.lanedict[nextedge]
        
        det = 'D'+nextedge+'_0'
        next_state.append(self.get_numVeh(det)) #vehicle number
        next_state.append(self.sumo.edge.getLastStepMeanSpeed(nextedge)) #avg speed
        next_state.append(self.sumo.lane.getLength(nextedge+'_0')) # length
        
        for hop in range(1,4):
            avgVeh, avgLength, avgSpeed = 0,0,0
            cnt = 0
            for i in np.where(self.hopadj[curidx]==hop)[0]: 
                cnt+=1
                edge = self.lanedict[i]
                det = 'D'+nextedge+'_0'
                avgVeh+=self.get_numVeh(det) #vehicle number
                avgLength += self.sumo.lane.getLength(nextedge+'_0') # length
                avgSpeed += self.sumo.edge.getLastStepMeanSpeed(edge) #avg speed
            hopFeature = [avgVeh, avgLength, avgSpeed]
            if cnt>1:
                hopFeature = [x/cnt for x in hopFeature]
            next_state.extend(hopFeature)    
        
        nextlane = nextedge+'_0' #nextedge좌표
        nextCord= self.sumo.lane.getShape(nextlane)[0]
        next_state.extend(list(nextCord))
        next_state.extend(list(self.destCord))
        return next_state

    def step(self, curedge,curlane, nextedge):
        
        beforeedge = curedge # 현재 edge 저장해두었다가 이동하면서 바뀌면 return 
        done = self.get_done(curedge)
        reward = self.get_reward(curedge,curlane, nextedge)

        if done:
            return reward, done
        
        self.sumo.vehicle.changeTarget(self.veh,nextedge) #Set next route
        #print('err route: ',self.sumo.vehicle.getRoute(self.veh))
        
        while self.sumo.simulation.getMinExpectedNumber() > 0:#Run a simulation until all vehicles have arrived
            self.sumo.simulationStep()
            try: #0322 10pm 수정s
                pos = self.sumo.vehicle.getPosition(self.veh)
            except self.sumo.TraCIException: #if unknown veh0 -> then assign 'veh0' again 
                while 1:
                    routeid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) #random id generator
                    if routeid not in self.sumo.route.getIDList():
                        break
                self.sumo.route.add(routeid, [beforeedge, nextedge]) #route 이름이 겹치면안된다고하니 수정필요!!!
                self.sumo.vehicle.add("veh0",routeid, typeID="agent")
                #pass or do something smarter
            curedge = self.get_RoadID(self.veh)# veh0 is not known error
            curlane = self.get_curlane(self.veh)
            done = self.get_done(curedge)
            if curedge !=beforeedge : #변했네!! 그럼 이제 다음 꺼 고르러 가야지
                break
            if done:
                break  
           
        return reward, done,beforeedge, curedge    

    
    
    