from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import matplotlib.pyplot as plt
from xml.etree.ElementTree import parse
from collections import defaultdict

from matplotlib.ticker import MaxNLocator



from sumolib import checkBinary
import traci

from qlenvironment import qlEnv
from qlagent import qlAgent

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    print(tools)
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')



def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("-N","--episodenum", 
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

def generate_detectionfile(net, det):
    #generate det.xml file by setting a detector at the end of each lane (-10m)
    with open(det,"w") as detections:
        alledges = get_alledges(net)
        alllanes = [edge +'_0' for edge in alledges]
        alldets =  [edge.replace("E","D") for edge in alledges]
        
        pos = -10
        print('<additional>', file = detections)
        for i in range(len(alledges)):
            print(' <e1Detector id="%s" lane="%s" pos="%i" freq="30" file="cross.out" friendlyPos="x"/>'
            %(alldets[i], alllanes[i],pos), file = detections)
        print('</additional>', file = detections)
        return alldets

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

    
##########@경로 탐색@##########
#1. random routing : routing randomly by calculating next route(edge) when the current edge is changed.
def random_run(sumoBinary, sumocfg, edgelists):
    traci.start([sumoBinary, "-c", sumocfg, "--tripinfo-output", "tripinfo.xml"]) #start sumo server using cmd
    
    step = 0
    traci.route.add("rou1", ["E19", "E0", "E1","E2","E3","E4"]) #default route
    traci.vehicle.add("veh0", "rou1")
    print('[ Veh0 Random Routes ]')
    traci.simulationStep()
    beforelane = traci.vehicle.getLaneID("veh0")
    beforeedge = traci.lane.getEdgeID(beforelane)
    print(beforeedge,end=' -> ') #start point

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        try:
            #get current edges
            curlane = traci.vehicle.getLaneID("veh0") # ** Error Point (2022/2/8) : Detector 지우고 edge변경시 routesetting하는 걸로 해결 가능 **
            curedge = traci.lane.getEdgeID(curlane)
            #curdet = curedge.replace('E','D') #골인 부분에서만 detector사용
            if traci.inductionloop.getLastStepVehicleNumber("D4")>0 or traci.inductionloop.getLastStepVehicleNumber("-D0")>0: #골인 detector
                print("[ Veh0 Arrived ]")
                traci.close()

            if curedge in edgelists and curedge !=beforeedge :   
            #if curdet in alldets and traci.inductionloop.getLastStepVehicleNumber(curdet)>0: 
                toedges = get_toedges(net, curedge)#get possible toedges (= target for nextedge)
                nextedge = random.choice(toedges)#choose next edge randomly
                print(curedge,end=' -> ')
                traci.vehicle.changeTarget('veh0',nextedge)
                beforeedge = curedge
        except:
            print('Simulation error occured. ')
            break
        step += 1
    traci.close()
    sys.stdout.flush()

if __name__ == "__main__":
    net = "./RL/rl1.net.xml"
    det = "./RL/rl2.det.xml"
    sumocfg = "./RL/rl2.sumocfg"
    veh = "veh0"
    endpoint = ['E20','-E19']
    badpoint = ['E5','E2','E13']
    successend = ["E20"]
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    
    alldets = generate_detectionfile(net, det) #generate detector file
    edgelists = get_alledges(net)
    dict_connection = calculate_connections(edgelists, net)
    
    """Run Simulation"""
    #1) Run randomly
    random_run(sumoBinary, sumocfg, edgelists)

   