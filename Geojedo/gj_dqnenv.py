import time

import numpy as np
import traci
import random
import string
from scipy.spatial import distance
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
        
        self.cnt_visited = defaultdict(int)
        self.sumo =traci
        self.normdestCord = (0,0)
        
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
        #destlane = [i+'_0' for i in self.destination]
        self.destCord = self.sumo.lane.getShape(self.destination[0])[0]
        x,y = self.destCord
        x/=1400
        y/=1500
        self.normdestCord = (x,y)
        #print('self.destCord: ',self.destCord)
        
     
        
    def sumo_step(self):
        self.sumo.simulationStep()

    def sumoclose(self):
        self.sumo.close()
        
    def reset(self):
        self.cnt_visited = defaultdict(int)
        self.episode+=1 
        self.start_simulation()
        
        curlane = self.get_curlane(self.veh)
        while curlane=='':
            curlane = self.get_curlane(self.veh)
            self.sumo.simulationStep()
        curedge = self.get_RoadID(self.veh)
        #curedge = self.get_curedge(curlane)
        
        state = self.get_state(self.veh,curedge, curlane)
            
        return state
        

    def get_curlane(self,veh):
        curlane = self.sumo.vehicle.getLaneID(veh)
        return curlane

    def get_curedge(self,curlane): ##0217목 8pm 수정 오류 확인해봐야함  curlane ''뜨는거!!!!!!
        curedge = self.sumo.lane.getEdgeID(curlane)
        return curedge

    def get_done(self,curlane):
        done = False
        if curlane in self.destination:
            done = True
        return done
    
    def get_RoadID(self, veh):
        return traci.vehicle.getRoadID(veh)

    def get_reward(self, curedge, curlane, nextedge,done):
        reward = 0
        #reward = traveling time of curedge
        det = 'D'+curlane
        '''
        num_veh = self.get_numVeh(det)
        if num_veh ==1: # 자기자신  #length/speedlimit 
            traveltime = self.dict_edgelengths[curedge] / self.dict_edgelimits[curedge]
        else:
            #print('err1 get_reward num_veh 확인용 : ',num_veh) 
            traveltime = self.sumo.edge.getTraveltime(curedge) #(length/mean speed).
        '''
        
        duration = self.sumo.simulation.getTime()
        
        curCord= self.sumo.lane.getShape(curlane)[0]
        cur = list(curCord) #현재 좌표
        dest = list(self.destCord)#목적지 좌표
        dist = distance.euclidean(cur, dest)
        
        reward = -((duration/10000) +dist/1000)

        if done:
            reward+=10
        
        return reward
    
    def get_nextlane(self, curlane, action):
        #print('lane: {}, action: {}'.format( curlane, action))
        nextlane = self.dict_connection[curlane][action]
        return nextlane

    def set_visited(self, curlane, action):
        cnt = 0
        for i in self.dict_connection[curlane]:
            if i=="":
                cnt+=1
        if cnt<4: self.dict_connection[curlane][action]=""

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
        '''
        state.append(self.get_numVeh(det)) #vehicle number
        state.append(self.sumo.edge.getLastStepMeanSpeed(curedge)) #avg speed
        state.append(self.sumo.lane.getLength(curedge+'_0')) # length
        
        for hop in range(1,2):
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
        '''
        curlane = curedge+'_0' #curedge좌표
        x,y= self.sumo.lane.getShape(curlane)[0]
        x/=1700
        y/=1800
        cur = [x,y]
        state.extend(cur)
        dest = list(self.normdestCord)
        state.extend(dest)#목적지 좌표
        #dist = distance.euclidean(cur, dest)
        #state.append(dist) #현위치 목적지 간의 거리
        return state

    def get_nextstate(self, veh, nextedge):
        next_state = []
        curidx = self.lanedict[nextedge]
        det = 'D'+nextedge+'_0'
        '''
        next_state.append(self.get_numVeh(det)) #vehicle number
        next_state.append(self.sumo.edge.getLastStepMeanSpeed(nextedge)) #avg speed
        next_state.append(self.sumo.lane.getLength(nextedge+'_0')) # length
        
        for hop in range(1,2):
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
        '''
        nextlane = nextedge+'_0' #nextedge좌표
        x,y= self.sumo.lane.getShape(nextlane)[0]
        x/=1700
        y/=1800
        cur = [x,y]
        next_state.extend(cur)
        dest = list(self.normdestCord)
        next_state.extend(dest)
        #dist = distance.euclidean(cur, dest)
        #next_state.append(dist)
        return next_state

    def step(self, curedge,curlane, nextedge,nextlane, epsilon):
        timeout = False
        cntout = False
        blockFrom = ""
        blockTo = ""
        beforeedge = curedge # Save current edge to find out if the edge change to next edge!! 
        beforelane = curlane
        done = self.get_done(curlane)
        reward = self.get_reward(curedge,curlane, nextedge,done)
        if epsilon<0.05: #If explore enough, then give penalty about visiting the same road repeatedly
            if nextlane not in self.cnt_visited:
                    self.cnt_visited[nextlane]=1
            else:
                self.cnt_visited[nextlane]+=1
            if self.cnt_visited[nextlane]>3:
                cntout = True
                blockFrom = curlane
                blockTo = nextlane
        self.sumo.vehicle.changeTarget(self.veh,nextedge) #Set next route
        #print('err route: ',self.sumo.vehicle.getRoute(self.veh))
        cnt = 0
        while self.sumo.simulation.getMinExpectedNumber() > 0:#Run a simulation until all vehicles have arrived
            time = self.sumo.simulation.getTime()
            if time>3000:
                timeout = True
                break
            self.sumo.simulationStep()
 
            try: #0322 10pm 수정s
                pos = self.sumo.vehicle.getPosition(self.veh)
            except self.sumo.TraCIException: #if unknown veh -> then assign 'self.veh' again 
                while 1:
                    routeid = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6)) #random id generator
                    if routeid not in self.sumo.route.getIDList():
                        break
                ##-----------------------error unknown veh----------------
                #reward = -1000000
                #return reward, done, curlane, timeout, cntout, blockFrom, blockTo 
                ##-----------------------error unknown veh----------------
                self.sumo.route.add(routeid, [beforeedge, nextedge]) #Not allow duplicated route name!
                self.sumo.vehicle.add(self.veh, routeid, typeID="agent")
                
            curedge = self.get_RoadID(self.veh)# self.veh is unknown error
            curlane = self.get_curlane(self.veh)
            
            done = self.get_done(curlane)
            
            if curlane !=beforelane and curlane!="": #If change, go to the midware to select next action
                break
            if done:
                break
        return reward, done, curlane, timeout, cntout, blockFrom, blockTo 
