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
            dict_connection[k]=['','','','']
        elif len(v)==1:
            dict_connection[k].extend(['']*3)
        elif len(v)==2:
            dict_connection[k].extend(['']*3)
        elif len(v)==3:
            dict_connection[k].extend(['']*1)
        
    return dict_connection 

#hop count ?????? -> BFS??????????????????
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

def make_hopadj(adj): #hop count matrix ??????
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

def get_adjdata():
    with open('./Geojedo/gj_adj.pkl', 'rb') as f:
       adj = pickle.load(f) # 2d (395, 395)
    
    with open('./Geojedo/gj_hopadj.pkl', 'rb') as f:
       hopadj = pickle.load(f)
    hopadj = np.asarray(hopadj)
    return adj, hopadj

##########@?????? ??????@##########
#DQN routing : routing by applying DQN algorithm (using qlEnv & alAgent)
def dqn_run(num_seed, trained,sumoBinary,plotResult, num_episode,net, trip, randomrou, add,dirResult,dirModel, sumocfg,fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size,lanedict,hopadj):  
    
    if trained :
        agent = dqnTrainedAgent( num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel)
        print('**** [TrainedAgent {} Route Start] ****'.format(num_episode))
    else:
        agent = dqnAgent(num_seed, edgelists, dict_connection, state_size, action_size, num_episode, dirModel)
   
    start = time.time()

    scores, episodes = [],[]
    score_avg = 0
    success_num = 0
    for episode in range(num_episode):
        num_sample = 0
        print("\n********#{} episode start***********".format(episode))

        episode_score = 0
        
        for vehi in range(len(destination)):
            print("Episode {} : Veh{} started.".format(episode, vehi))
            #Random traffic ??????
            #cmd_genDemand = "python \"C:/Program Files (x86)/Eclipse/Sumo/tools/randomTrips.py\" -n {} -o {} -r {} -b 0 -e 3600 -p 3 --additional-file {} --trip-attributes \"type='type1'\" --random\"".format(net, trip, randomrou, add)
            #os.system(cmd_genDemand)  
            
            # Reset environment 
            veh = "veh"+str(vehi)
            env = gj_dqnEnv(sumoBinary, net_file = net, cfg_file = sumocfg, edgelists = edgelists, alldets=alldets, dict_connection=dict_connection, veh = veh, destination = destination[vehi], state_size = state_size, action_size= action_size,lanedict = lanedict, hopadj = hopadj)
            state = env.reset() 
            state = np.reshape(state,[1,state_size]) #for be keras model's input
            
            routes = []
            curedge = env.get_RoadID(veh) 
            curlane = env.get_curlane(veh)
            routes.append(curedge)
            print('%s -> ' %curlane, end=' ') #print path
            
            done = False
            timeout = False #Over 3000s, navigation ends
            cntout = False #repeatedly visit same lane over 3 times 
            cnt=0
            while (not done) and (not timeout) and (not cntout):     
                block = True
            
                state = env.get_state(veh, curedge, curlane) 
                
                state = np.reshape(state,[1,state_size]) #for be model's input
                cnt = 0
                
                while block: #?????? ????????? ????????????(?????? ?????? = '' ??? ????????????) do while 
                    if curlane in destination[vehi]:
                        break
                    if trained:
                        qvalue, action = agent.get_trainedaction(state)
                    else:
                        if destination[vehi][0] in (dict_connection[curlane]): #next lane??? destination????????? ?????? ????????????!!
                            action = dict_connection[curlane].index(destination[vehi][0])
                        else:
                            action,epsilon = agent.get_action(curlane, state) #?????? edge?????? ????????? action ?????? 

                    nextlane = env.get_nextlane(curlane, action) #next edge ???????????? env??? ??????.
                    
                    if nextlane!="" : 
                        nextedge = nextlane[:-2]
                        break

                routes.append(nextedge)
                
                print('%s -> ' %nextlane, end=' ')
                next_state = env.get_nextstate(veh, nextedge)  
                next_state = np.reshape(state,[1,state_size]) #for be ?????? input
                reward, done, nextlane, timeout,cntout,blockFrom, blockTo = env.step(curedge, curlane, nextedge,nextlane, epsilon) #Step Environment
                ##-----------------------error unknown veh----------------
                #if reward == -1000000:
                #    print('\nUnknonw error: ',nextlane)
                #    sys.exit()
                ##-----------------------error unknown veh----------------

                if nextlane =="":
                    print('[ERR] Next Lane null')
                    timeout = True
        
                episode_score += reward
                
                if not trained: 
                    num_sample+=1
                    agent.append_sample(state, action, reward, next_state, done)
                
                if not trained and (len(agent.memory)> agent.train_start*(1-agent.train_success_rate)+1) and (len(agent.success_memory)>agent.train_start*agent.train_success_rate+1): agent.train_model() #????????????

                if (not trained and done) or (not trained and timeout) or (not trained and cntout): #Done by success or Timeout or Cntout
                    if not trained and (len(agent.memory)> agent.train_start*(1-agent.train_success_rate)+1) and (len(agent.success_memory)>agent.train_start*agent.train_success_rate+1):
                                    if episode%3==0: agent.update_target_model() #5?????? ?????????????????? ?????? ????????? ????????? ???????????? ????????????
                    if done:
                        print('Last lane:',nextlane)
                        success_num+=1
                        print('veh0 Arrived Successfully ^____^')
                        print('Added success_sample : ',num_sample)
                        success_sample = [agent.memory.pop() for _ in range(num_sample)]
                        agent.success_memory.extend(success_sample)

                    if timeout:
                        print('\n* Finished by Time Out !!!')

                    if cntout: #blockFrom->blockTo(3????????? ??????) ???????????????
                        for idx, lane in enumerate(dict_connection[blockFrom]):
                            if blockTo in lane:
                                dict_connection[blockFrom][idx] = ""
                                #print('After change: ', dict_connection[blockFrom])
                        print('\n* Finished by Count Out !!!')

                    
                    #sumo ??????
                    env.sumoclose()
                    break
                
                if trained and done: #trained ?????? : ?????? Trained??? ?????? ????????? ???-> append_sample, train_model, update_target_model ???????????? 
                    env.sumoclose()
                    score_avg = 0.9*score_avg +0.1*episode_score if score_avg!=0 else episode_score
                    print("\n****Trained episode : {} | score_avg : {} ".format(episode, score_avg) )
                    scores.append(-episode_score)
                    episodes.append(episode)
                    if plotResult: plot_trainedresult(num_seed, episodes, scores, dirResult, num_episode)

                curedge = nextedge
                curlane = nextlane
                cnt+=1
        
        #Mean Travel Time ??????
        score_avg = 0.9*score_avg +0.1*episode_score if score_avg!=0 else episode_score
        print("\n* Episode : {} | episode_score: {}| score_avg : {} | length of memory : {} vs success_memory : {}| epsilon : {}".format(episode,episode_score, score_avg,len(agent.memory),len(agent.success_memory),agent.epsilon))
        
        #Plot Result Mean Travel Time
        scores.append(-score_avg) #Mean Travel Time
        episodes.append(episode)
        if plotResult: plot_result(num_seed, episodes, scores, dirResult, num_episode)   

        #epsilon decay
        agent.decaying_epsilon()
        end = time.time()
        print("Succes # : {} out of Try #{}".format(success_num,episode*len(destination)))
        print('Source Code Time: ',end-start)

        #DQN Weights ??????
        agent.save_weights()
    
    sys.stdout.flush()
   
  
if __name__ == "__main__":
    net = "./Geojedo/Net/geojedo_simple.net.xml" #network.net.xml
    
    add = "./Geojedo/Add/gj_dqn.add.xml" #Traffic Flow type : normal_car & truck
    trip = "./Geojedo/Rou/gj_dqn.trip.xml" #Random Traffic flow generation
    randomrou = "./Geojedo/Rou/gj_dqnrandom.rou.xml"
    det = "./Geojedo/Add/gj_dqn.det.xml" #detection per lane

    sumocfg = "./Geojedo/gj_dqn.sumocfg" #sumocfg -> gj_agent.rou.xml(initial point)?????? 
    
    dirResult = './Geojedo/Result/dqn' #Result
    dirModel = './Geojedo/Model/dqn'

    fcdoutput = './Geojedo/Output/dqn.fcdoutput.xml'

    veh = "veh"
    
    destination0 = ["-240392105#0_0"] #yellow
    destination1 = ["-243463807#3_0"] #red
    destination2 = ["-458050462#1_0"] #blue
    destination3 = ["320980930#3_0"] #green
    destination4 = ["320980919#0_0"] #purple
    destination = [destination0,destination1,destination2,destination3,destination4 ]
    state_size = 4
    action_size = 4

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
                                  # 227edges for "./Geojedo/Net/geojedo_simple.net.xml"
    dict_connection = calculate_connections(net, action_size) # calculate lane connection : dict_fromto + fill empty route as ''   
                                                               # action_size = 4 for "./Geojedo/Net/geojedo_simple.net.xml"
                                                               # action_size = 5 for "./Geojedo/Net/geojedo.net.xml" 
    dict_fromto = calculate_fromto(net) #only real connenciton information without fill empty as ''
    #print('Possible next lane from initial lane: ',dict_fromto["-320980921#2_0"])
    keys=sorted(dict_fromto.keys())
    len_lanes = len(keys) 
    print("Num of lanes: ", len_lanes)
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
    hopadj, adj = get_adjdata()
    dets = generate_lanedetectionfile(net,det) #already made
    alldets = get_alldets(net)

    """1) Run Simulation"""
    trained = False
    num_seed = random.randrange(1000)
    while True: #num_episode????????? num_seed??? ???????????? ????????? ?????? ?????? ?????????. 
        file = dirModel + str(num_episode)+'_'+str(num_seed)+'.h5'
        if not os.path.isfile(file): break
    print('Num of edges: ',len(edgelists))
    
    dqn_run(num_seed, trained, sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult,dirModel,sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size, lanedict, hopadj)
    
    '''
    """2) Run with pre-trained model : Load Weights & Route """
    
    trained = True
    num_seed = 975
    dqn_run(num_seed, trained, sumoBinary, plotResult, num_episode, net, trip, randomrou, add, dirResult,dirModel,
    sumocfg, fcdoutput, edgelists,alldets, dict_connection,veh,destination, state_size, action_size)
    '''