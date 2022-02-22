import numpy as np
import random
from collections import defaultdict

class qlAgent():
    def __init__(self, edgelists, dict_connection):
        self.edgelists = edgelists
        self.dict_connection = dict_connection
        self.step_size = 0.3
        self.discount_factor=0.9
        self.epsilon=0.2
        self.minepsilon = 0.05
        self.decayingrate = 0.95
        self.actions = list(range(3))
        self.qtable=defaultdict(list)
        self.blockreward = -100
        
    def set_qtable(self):
        self.qtable = self.dict_connection.copy()
        for k,v in self.qtable.items():
            q = []
            for i in v:
                if i!="":
                    q.append(0.0)
                else:
                    q.append(-100.0)
            self.qtable[k] = q
        return self.qtable

    def learn_block(self, curedge, action):
        self.qtable[curedge][action]+=self.blockreward

    def learn(self, curedge, action, reward, nextedge):
        
        q1 = self.qtable[curedge][action]
        q2 = reward+self.discount_factor*max(self.qtable[nextedge])
        self.qtable[curedge][action] += self.step_size*(q2-q1)
        self.qtable[curedge][action] = round(self.qtable[curedge][action],3)
    
    def set_episilon(self):
        self.epsilon = min(self.minepsilon, self.epsilon*self.decayingrate)
        
    def get_action(self, curedge):
        
        if np.random.rand()<self.epsilon:
            action = np.random.choice(self.actions)
            
        else:
            qlist = self.qtable[curedge]
            maxlist = np.argwhere(qlist == np.amax(qlist))
            maxlist = maxlist.flatten().tolist()
            action = random.choice(maxlist)
            
        return action
    
