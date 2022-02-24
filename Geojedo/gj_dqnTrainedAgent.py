import numpy as np
import random
from collections import defaultdict
from collections import deque
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

from gj_dqn import DQN

class dqnTrainedAgent():
    def __init__(self, num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel):
        self.edgelists = edgelists
        self.dict_connection = dict_connection
        self.state_size = state_size
        self.action_size = action_size
        self.num_episode = num_episode
        self.dirModel = dirModel    
        self.num_seed = num_seed
      
        #모델 생성
        self.trained_model = self.get_trainedmodel()
        self.trained_model.built = True
        self.trained_model.load_weights(dirModel+str(num_episode)+'_'+str(self.num_seed)+'.h5')
        
    #After training -> get action by trained weights 
    def get_trainedmodel(self):
        trained_model = DQN(self.action_size)
        return trained_model

    def get_trainedaction(self, state):
        qvalue = self.trained_model(state)
        
        action = np.argmax(qvalue[0])
        return qvalue, action