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

def get_edgesinfo(net):
    tree = parse(net)
    root = tree.getroot()
    alledgelists = root.findall("edge")
    edgesinfo = [x.find("lane").attrib for x in alledgelists]
    return edgesinfo

def calculate_fromto(edgelists, net):
    # calculate dictionary of reachable edges(next edge) for every edge  
    tree = parse(net)
    root = tree.getroot()
    
    dict_fromto = defaultdict(list)
    #edgelists = [x[1] for x in edgeindex]
    dict_fromto.update((k,[]) for k in edgelists)

    for connection in root.iter("connection"):
        curedge = connection.get("from")
        if ':' not in curedge:
            dict_fromto[curedge].append(connection.get("to"))
    return dict_fromto

def calculate_connections(edgelists, net,action_size):
    # calculate dictionary of reachable edges(next edge) for every edge  
    tree = parse(net)
    root = tree.getroot()
    
    dict_connection = defaultdict(list)
    #edgelists = [x[1] for x in edgeindex]
    dict_connection.update((k,[]) for k in edgelists)

    for connection in root.iter("connection"):
        curedge = connection.get("from")
        if ':' not in curedge:
            dict_connection[curedge].append(connection.get("to"))

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
    
def generate_lanedetectionfile(net, det):
    #generate det.xml file by setting a detector at the end of each lane (-10m)
    alledges = get_alledges(net)
    edgesinfo = get_edgesinfo(net)
    alllanes = [edge +'_0' for edge in alledges]
    alldets =  [edge.replace("E","D") for edge in alledges]  
  
    with open(det,"w") as f:
        print('<additional>', file = f)
        for i,v in enumerate(edgesinfo):
            
            print('        <laneAreaDetector id="%s" lane="%s" pos="0.0" length="%s" freq ="%s" file="dqn_detfile.out"/>'
            %(alldets[i], v['id'],v['length'],"1"), file = f)
        print('</additional>', file = f)
    return alldets

def get_alldets(alledges):
    alldets =  [edge.replace("E","D") for edge in alledges]
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
def dqn_run(num_seed, trained,sumoBinary,plotResult, num_episode,net, trip, randomrou, add,dirResult,dirModel, sumocfg,fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size,edgedict,hopadj):  
    env = gj_dqnEnv(sumoBinary, net_file = net, cfg_file = sumocfg, edgelists = edgelists, alldets=alldets, dict_connection=dict_connection, veh = veh, destination = destination, state_size = state_size, action_size= action_size,edgedict = edgedict, hopadj = hopadj)
  
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
        cmd_genDemand = "python \"C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py\" -n {} -o {} -r {} -b 0 -e 3600 -p 3 --additional-file {} --trip-attributes \"type='type1'\" --random\"".format(net, trip, randomrou, add)
        os.system(cmd_genDemand)   
   
        
        score = 0
        routes = []

        #reset environment
        state = env.reset() # state : 
        state = np.reshape(state,[1,state_size]) #for be 모델 input
        
        curedge = env.get_RoadID(veh) #0221수정: 위 2개 명령어 대신 get_RoadID로 통일시킴
        
        routes.append(curedge)
        print('%s -> ' %curedge, end=' ')
        done = False

        cnt=0
        while not done:     
            block = True
            #cnt = 0
            while block: #막힌 도로를 골랐을때(막힌 도로 = '' 로 저장해둠)
                if curedge ==destination:
                    break
             
                curedge = env.get_RoadID(veh) # [err] Still veh0 is not known error 
                state = env.get_state(veh, curedge) 
                state = np.reshape(state,[1,state_size]) #for be 모델 input

                if trained:
                    qvalue, action = agent.get_trainedaction(state)
                else:
                    action = agent.get_action(state) #현재 edge에서 가능한 (0,1,2) 중 선택 

                nextedge = env.get_nextedge(curedge, action) #next edge 계산해서 env에 보냄.
                if nextedge!="" : break

            print('%s -> ' %nextedge, end=' ')
            routes.append(nextedge)
            
            next_state = env.get_nextstate(veh, nextedge)  
            next_state = np.reshape(state,[1,state_size]) #for be 모델 input
            reward, done = env.step(curedge, nextedge) #changeTarget to nextedge
            score += reward

            if not trained: agent.append_sample(state, action, reward, next_state, done)

            if not trained and len(agent.memory)>= agent.train_start:
                agent.train_model()
            
            if score<-1000: #이 기능 테스트 필요 0219 6pm
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
    
    destination = '-239842081#2'
    successend = ["-239842081#2"]
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
    
    dict_connection = calculate_connections(edgelists, net, action_size) # dict_fromto + fill empty route as ''
    dict_fromto = calculate_fromto(edgelists, net) #only connenciton information from -> to edges
    len_edge = len(edgelists)
    keys=sorted(dict_fromto.keys())
    edgedict = defaultdict(int)
    edgedict.update((i,v) for i,v in enumerate(keys))
    
    '''             #already made and save as gj_adj.pkl
    adj = [[0]*len_edge for _ in range(len_edge)]
    for a,b in [(keys.index(a), keys.index(b)) for a, row in dict_fromto.items() for b in row]:
        adj[a][b] = 1
    with open('./Geojedo/gj_adj.pkl','wb') as f:
        pickle.dump(adj, f)
    
    make_hopadj(adj) #already made and save as gj_hopadj.pkl
    '''
    with open('./Geojedo/gj_adj.pkl', 'rb') as f:
       adj = pickle.load(f) # list of (395, 395)
    with open('./Geojedo/gj_hopadj.pkl', 'rb') as f:
       hopadj = pickle.load(f)
    hopadj = np.asarray(hopadj)
    
    dets = generate_lanedetectionfile(net,det) #이미 생성해둠!
    alldets = get_alldets(edgelists)
    

    """1) Run Simulation"""
    
    trained = False
    num_seed = random.randrange(1000)
    while True: #num_episode같아도 num_seed를 달리해서 겹치는 파일 생성 방지함. 
        file = dirModel + str(num_episode)+'_'+str(num_seed)+'.h5'
        if not os.path.isfile(file): break
    dqn_run(num_seed, trained, sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult,dirModel,
    sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size, edgedict, hopadj)
    
    
    """2) Run with pre-trained model : Load Weights & Route """
    '''
    trained = True
    num_seed = 975
    dqn_run(num_seed, trained, sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult,dirModel,
    sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size)
    '''