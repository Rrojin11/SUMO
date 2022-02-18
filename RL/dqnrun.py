from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import optparse
import random
import pylab
import numpy as np
import matplotlib.pyplot as plt
from xml.etree.ElementTree import parse
from collections import defaultdict

from matplotlib.ticker import MaxNLocator



from sumolib import checkBinary
import traci

from dqnenv import dqnEnv
from dqnagent import dqnAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    print(tools)
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-N","--num_episode", 
                        default=30, help="numer of episode to run qlenv")
    optParser.add_option("--nogui", action="store_true",
                        default=False, help="run commandline version of sumo")
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

def calculate_connections(edgelists, net):
    # calculate dictionary of reachable edges(next edge) for every edge  
    tree = parse(net)
    root = tree.getroot()
    
    dict_connection = defaultdict(list)
    dict_connection.update((k,[]) for k in edgelists)

    for connection in root.iter("connection"):
        curedge = connection.get("from")
        if ':' not in curedge:
            dict_connection[curedge].append(connection.get("to"))

    for k,v in dict_connection.items():
        if len(v)==0:
            dict_connection[k]=['','','']
        elif len(v)==1:
            dict_connection[k].append('')
            dict_connection[k].append('')
        elif len(v)==2:
            dict_connection[k].append('')
    return dict_connection 


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

##########@경로 탐색@##########
#DQN routing : routing by applying DQN algorithm (using qlEnv & alAgent)
def dqn_run(sumoBinary, num_episode,net, sumocfg, edgelists,alldets, dict_connection,veh,endpoint,destination, state_size, action_size):  
    env = dqnEnv(sumoBinary, net_file = net, cfg_file = sumocfg, edgelists = edgelists, alldets=alldets, dict_connection=dict_connection, veh = veh, endpoint = endpoint, destination = destination, state_size = state_size, action_size= action_size)
    agent = dqnAgent(edgelists, dict_connection, state_size, action_size)
    

    #curedge = env.reset() #테스트용 으로 실제 없애야함
    #print('DQN Test용: ',curedge)
    
    travel_times,scores, episodes = [],[],[]
    score_avg = 0
    for episode in range(num_episode):
        print("\n********#{} episode start***********".format(episode))
        #reset environment
        #generate random output 
        cmd_genDemand = "python ./tools/randomTrips.py -n ./RL/dqn.net.xml -o ./RL/dqn.trip.xml -r ./RL/dqnrandom.rou.xml -p 2 --additional-file ./RL/dqn.add.xml --trip-attributes \"type='type1'\" --random"
        os.system(cmd_genDemand)     
        start = time.time()
        score = 0
        routes = []
       
        state = env.reset() #58개 
        
        state = np.reshape(state,[1,state_size]) #for be 모델 input
        curlane = env.get_curlane(veh)
        curedge = env.get_curedge(curlane)
        routes.append(curedge)
        print('%s -> ' %curedge, end=' ')
        done = False

        cnt=0
        while not done:     
            block = True
            while block: #막힌 도로를 고름 (막힌 도로 = '', not like 'E*' or '-E*')
                if curedge in endpoint:
                    break
                curlane = env.get_curlane(veh)
                curedge = env.get_curedge(curlane)
                #print(' err1 curedge: ',curedge)
                state = env.get_state(veh, curedge) 
                state = np.reshape(state,[1,state_size]) #for be 모델 input
                
                action = agent.get_action(state) #현재 edge에서 가능한 (0,1,2) 중 선택 

                nextedge = env.get_nextedge(curedge, action) #next edge 계산해서 env에 보냄.

                if nextedge!="" : break

                #agent.learn_block(curedge, action) #막힌 도로 선택시, blockreward 부여(blockreward - -100)
            
            print('%s -> ' %nextedge, end=' ')
            routes.append(nextedge)
            
            next_state = env.get_nextstate(veh, nextedge)  
            next_state = np.reshape(state,[1,state_size]) #for be 모델 input
            reward, done = env.step(curedge, nextedge) #changeTarget to nextedge
            score += reward

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory)>= agent.train_start:
                agent.train_model()
            
            if done:
                #각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                agent.update_target_model()
                
                score_avg = 0.9*score_avg +0.1*score if score_avg!=0 else score
                print("\n****episode : {} | score_avg : {} | memory_length : {} | epsilon : {}".format(episode, score_avg,len(agent.memory), agent.epsilon) )
                
                #결과 Plot
                end = time.time()
                travel_times.append(end-start)
                scores.append(score)
                episodes.append(episode)

                pylab.plot(episodes, travel_times, 'b')
                pylab.xlabel('episode')
                pylab.ylabel('Travel Time')
                pylab.savefig('./RL/result/dqn'+str(num_episode)+'.png')               

                if nextedge==endpoint[0]:
                    print('Arrived:) ')
                else:
                    print('Bad Arrived:( ')
                break
            
            curedge = nextedge
            cnt+=1
     
    sys.stdout.flush()
            
    

if __name__ == "__main__":
    net = "./RL/dqn.net.xml"
    add = "./RL/dqn.add.xml"
    det = "./RL/dqn.det.xml"
    sumocfg = "./RL/dqn.sumocfg"
    
    veh = "veh0"
    endpoint = ['E9','-E0']
    destination = 'E9'
    successend = ["E9"]
    state_size = 64
    action_size = 3

    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    edgelists = get_alledges(net) #총 20개 출력
    dict_connection = calculate_connections(edgelists, net)
    dets = generate_lanedetectionfile(net,det) #이미 생성해둠!
    alldets = get_alldets(edgelists)
    """Run Simulation"""
    #2) Run in qlearning environment
    num_episode= int(options.num_episode)
    dqn_run(sumoBinary, num_episode, net, sumocfg, edgelists,alldets, dict_connection,veh,endpoint,destination, state_size, action_size)
 