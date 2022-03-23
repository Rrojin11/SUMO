from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import pickle
import random
import pylab
import numpy as np
import matplotlib.pyplot as plt
from xml.etree.ElementTree import parse
from queue import Queue
from collections import defaultdict


from sumolib import checkBinary
from gj_dqnTrainedAgent import dqnTrainedAgent
from gj_dqnenv import gj_dqnEnv
from gj_dqnagent import dqnAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    print(tools)
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-N","--num_episode", 
                        default=100, help="numer of episode to run qlenv")
    optParser.add_option("--nogui", action="store_true",
                        default=False, help="run commandline version of sumo")
    optParser.add_option("--noplot", action="store_true",
                        default=False, help="save result in png")    
    optParser.add_option("--trained", "-T", action="store_true",
                        default=False, help="save result in png")                                    
    options, args = optParser.parse_args()
    return options

def get_toedges(net, fromedge):
    #calculate reachable nextedges
    tree = parse(net)
    root = tree.getroot()
    toedges = []
    for connection in root.iter("connection"):
        if connection.get("from")==fromedge:
            toedges.append(connection.get("to"))
    return toedges

def get_alledges(net):
    #get plain edges by parsing net.xml
    tree = parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    edgelists = [edge.get("id") for edge in alledgelists if ':' not in edge.get("id")]
    return edgelists

def get_lanesinfo(net):  
    lanesinfo = []
    tree = parse(net)
    root = tree.getroot()
    for lane in root.iter("lane"):
        lanesinfo.append(lane.attrib)

    return lanesinfo

def calculate_fromto(net):
    
    tree = parse(net)
    root = tree.getroot()
    
    dict_fromto = defaultdict(list)
   
    for connection in root.iter("connection"):
        curedge = connection.get("from")
        fromlane = connection.get("from")+'_'+connection.get("fromLane")
        tolane = connection.get("to")+'_'+connection.get("toLane")
      
        if ':' not in curedge:
            if fromlane in dict_fromto:
                dict_fromto[fromlane].append(tolane)
            else:
                dict_fromto[fromlane] = [tolane]
    return dict_fromto

def calculate_connections(net,action_size): # calculate dictionary of reachable edges(next edge) for every edge  
    tree = parse(net)
    root = tree.getroot()
    
    dict_connection = defaultdict(list)
  
    for connection in root.iter("connection"):
        curedge = connection.get("from")
        fromlane = connection.get("from")+'_'+connection.get("fromLane")
        tolane = connection.get("to")+'_'+connection.get("toLane")
      
        if ':' not in curedge:
            if fromlane in dict_connection:
                dict_connection[fromlane].append(tolane)
            else:
                dict_connection[fromlane] = [tolane]
    for k,v in dict_connection.items():
    
        if len(v)==0:
            dict_connection[k]=['','','','','']
        elif len(v)==1:
            dict_connection[k].extend(['']*4)
        elif len(v)==2:
            dict_connection[k].extend(['']*3)
        elif len(v)==3:
            dict_connection[k].extend(['']*2)
        elif len(v)==4:
            dict_connection[k].extend(['']*1)
    return dict_connection 

#hop count 계산 -> BFS알고리즘사용
def bfs(adj, start_node):
    x= len(adj)
    adj = np.asarray(adj)
    hops, visited =[0]*x,[0]*x
    q = Queue()
    
    for s in np.where(adj[start_node]==1)[0]:
        q.put(s)
        hops[s]=1
        visited[s]=1

    while q.qsize() > 0 :
        node = q.get()
        for nextN in np.where(adj[node]==1)[0]:
            if nextN != start_node and visited[nextN]==0:
                visited[nextN]=1
                q.put(nextN)
                hops[nextN]=hops[node]+1 
            
    return hops

def make_hopadj(adj): #hop count matrix 생성
    gj_hopadj=[]
    
    for i in range(len(adj)):
        gj_hopadj.append(bfs(adj, i))

    with open('./Geojedo/gj_hopadj.pkl', 'wb') as f:
        pickle.dump(gj_hopadj, f)
    
def generate_lanedetectionfile(net, det):   #generate det.xml file 
    lanesinfo = get_lanesinfo(net)
    with open(det,"w") as f:
        print('<additional>', file = f)
        for v in lanesinfo:
            print('        <laneAreaDetector id="%s" lane="%s" pos="0.0" length="%s" freq ="%s" file="dqn_detfile.out"/>'
            %('D'+v['id'], v['id'],v['length'],"1"), file = f)
        print('</additional>', file = f)
    

def get_alldets(net):
    lanesinfo = get_lanesinfo(net)
    alldets =  ['D'+lane['id'] for lane in lanesinfo]  
    return alldets

def plot_result(num_seed, episodes, scores, dirResult, num_episode):
    pylab.plot(episodes, scores, 'b')
    pylab.xlabel('episode')
    pylab.ylabel('Mean Travel Time')
    pylab.savefig(dirResult+str(num_episode)+'_'+str(num_seed)+'.png')    

def plot_trainedresult(num_seed, episodes, scores, dirResult, num_episode):
    pylab.plot(episodes, scores, 'b')
    pylab.xlabel('episode')
    pylab.ylabel('Mean Travel Time')
    pylab.savefig(dirResult+'TrainedModel'+str(num_episode)+'_'+str(num_seed)+'.png')  



##########@경로 탐색@##########
#DQN routing : routing by applying DQN algorithm (using qlEnv & alAgent)
def dqn_run(num_seed, trained,sumoBinary,plotResult, num_episode,net, trip, randomrou, add,dirResult,dirModel, sumocfg,fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size,lanedict,hopadj):  
    env = gj_dqnEnv(sumoBinary, net_file = net, cfg_file = sumocfg, edgelists = edgelists, alldets=alldets, dict_connection=dict_connection, veh = veh, destination = destination, state_size = state_size, action_size= action_size,lanedict = lanedict, hopadj = hopadj)
  
    if trained :
        agent = dqnTrainedAgent( num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel)
        print('**** [TrainedAgent {} Route Start] ****'.format(num_episode))
    else:
        agent = dqnAgent(num_seed, edgelists, dict_connection, state_size, action_size, num_episode, dirModel)
    
    start = time.time()

    scores, episodes = [],[]
    score_avg = 0
    for episode in range(num_episode):
        
        print("\n********#{} episode start***********".format(episode))

        #Random traffic 생성
        cmd_genDemand = "python \"C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py\" -n {} -o {} -r {} -b 0 -e 3600 -p 2 --additional-file {} --trip-attributes \"type='type1'\" --random\"".format(net, trip, randomrou, add)
        os.system(cmd_genDemand)   
   
        
        score = 0
        routes = []

        #reset environment
        state = env.reset() # state : 
        state = np.reshape(state,[1,state_size]) #for be 모델 input
        
        curedge = env.get_RoadID(veh) 
        curlane = env.get_curlane(veh)
        routes.append(curedge)
        print('%s -> ' %curedge, end=' ')
        done = False

        cnt=0
        while not done:     
            block = True
            #curedge = env.get_RoadID(veh) # [err0310] veh0 is not known error -->curedge=nextedge코드가 있으니 필요없는거아님> 0322/10시20분
            state = env.get_state(veh, curedge, curlane) 
            state = np.reshape(state,[1,state_size]) #for be model's input

            while block: #막힌 도로를 골랐을때(막힌 도로 = '' 로 저장해둠) do while 문 
                if curedge in destination:
                    break
                if trained:
                    qvalue, action = agent.get_trainedaction(state)
                else:
                    action = agent.get_action(state) #현재 edge에서 가능한 action 선택 

                nextedge = env.get_nextedge(curedge, action) #next edge 계산해서 env에 보냄.
                if nextedge!="" : break

            print('%s -> ' %nextedge, end=' ')
            routes.append(nextedge)
            
            next_state = env.get_nextstate(veh, nextedge)  
            next_state = np.reshape(state,[1,state_size]) #for be 모델 input
            reward, done,before,cur = env.step(curedge, curlane, nextedge) 
            #print('before: %s -> cur: %s'%(before, cur))
            score += reward

            if not trained: agent.append_sample(state, action, reward, next_state, done)

            if not trained and len(agent.memory)>= agent.train_start:
                agent.train_model()
            
            if score<-100000: #이 기능 테스트 필요 0219 6pm
                done = True

            if not trained and done: 
                #각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
                #sumo 종료
                env.sumoclose()
                #Mean Travel Time 계산
                score_avg = 0.9*score_avg +0.1*score if score_avg!=0 else score
                print("\n****episode : {} | score_avg : {} | memory_length : {} | epsilon : {}".format(episode, score_avg,len(agent.memory), agent.epsilon) )
                
                #Plot Result Mean Travel Time
                scores.append(-score_avg) #Mean Travel Time
                episodes.append(episode)
                if plotResult: plot_result(num_seed, episodes, scores, dirResult, num_episode)   
                break
            
            if trained and done: #trained 의미 : 이미 Trained된 모델 사용할 때-> append_sample, train_model, update_target_model 필요없음 
                env.sumoclose()
                score_avg = 0.9*score_avg +0.1*score if score_avg!=0 else score
                print("\n****Trained episode : {} | score_avg : {} ".format(episode, score_avg) )
                scores.append(-score)
                episodes.append(episode)
                if plotResult: plot_trainedresult(num_seed, episodes, scores, dirResult, num_episode)

            curedge = nextedge
            cnt+=1

        
    end = time.time()
    print('Source Code Time: ',end-start)

    #DQN Weights 저장
    agent.save_weights()
    
    sys.stdout.flush()
   


if __name__ == "__main__":
    net = "./Geojedo/Net/geojedo.net.xml"
    
    add = "./Geojedo/Add/gj_dqn.add.xml"
    trip = "./Geojedo/Rou/gj_dqn.trip.xml"
    randomrou = "./Geojedo/Rou/gj_dqnrandom.rou.xml"
    
    det = "./Geojedo/Add/gj_dqn.det.xml"
    sumocfg = "./Geojedo/gj_dqn.sumocfg"
    
    dirResult = './Geojedo/Result/dqn'
    dirModel = './Geojedo/Model/dqn'
    fcdoutput = './Geojedo/Output/dqn.fcdoutput.xml'

    veh = "veh0"
    
    destination = ["-240392105#0","-243399011#1"]
    successend = ["-240392105#0","-243399011#1"]
    state_size = 16
    action_size = 5


    options = get_options()
    if options.nogui: sumoBinary = checkBinary('sumo')
    else: sumoBinary = checkBinary('sumo-gui')

    if options.noplot: plotResult = False
    else: plotResult = True
    
    if options.noplot: plotResult = False
    else: plotResult = True
    
    if options.num_episode: num_episode =int(options.num_episode)
    else: num_episode = 300
  
    edgelists = get_alledges(net) # 395edges for "./Geojedo/Net/geojedo.net.xml" 
    
    dict_connection = calculate_connections(net, action_size) # dict_fromto + fill empty route as ''   
    dict_fromto = calculate_fromto(net) #only real connenciton information without fill empty as ''
       
    keys=sorted(dict_fromto.keys())
    len_lanes = len(keys) #429
    lanedict = defaultdict(int) #eg. 0 -239842076#0_0
    lanedict.update((i,v) for i,v in enumerate(keys))
    
    '''
    adj = [[0]*len_lanes for _ in range(len_lanes)]
    for a,b in [ (keys.index(a), keys.index(b)) for a, row in dict_fromto.items() for b in row ]:
        adj[a][b] = 1 #already saved as gj_adj.pkl

    with open('./Geojedo/gj_adj.pkl','wb') as f:
        pickle.dump(adj, f)
    
    make_hopadj(adj) #already saved as gj_hopadj.pkl
    '''

    with open('./Geojedo/gj_adj.pkl', 'rb') as f:
       adj = pickle.load(f) # 2d (395, 395)
    
    with open('./Geojedo/gj_hopadj.pkl', 'rb') as f:
       hopadj = pickle.load(f)
    hopadj = np.asarray(hopadj)

    dets = generate_lanedetectionfile(net,det) #already made
    alldets = get_alldets(net)
    #print(dict_connection["478427205#2_0"])
    
    """1) Run Simulation"""
    trained = False
    num_seed = random.randrange(1000)
    while True: #num_episode같아도 num_seed를 달리해서 겹치는 파일 생성 방지함. 
        file = dirModel + str(num_episode)+'_'+str(num_seed)+'.h5'
        if not os.path.isfile(file): break
    dqn_run(num_seed, trained, sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult,dirModel,
    sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size, lanedict, hopadj)
    
    '''
    """2) Run with pre-trained model : Load Weights & Route """
    
    trained = True
    num_seed = 975
    dqn_run(num_seed, trained, sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult,dirModel,
    sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size)
    '''