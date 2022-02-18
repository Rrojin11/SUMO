import numpy as np
import random
from collections import defaultdict
from collections import deque
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

from dqn import DQN

class dqnAgent():
    def __init__(self, edgelists, dict_connection, state_size, action_size):
        self.edgelists = edgelists
        self.dict_connection = dict_connection
        self.state_size = state_size
        self.action_size = action_size

        #DQN
        self.discount_factor=0.99
        self.learning_rate = 0.001
        self.epsilon=1.0
        self.epsilon_decay=0.95
        self.epsilon_min = 0.01
        self.batch_size = 32    #조절
        self.train_start = 500 #조절
        
        #리플레이 메모리 최대크기 2000
        self.memory = deque(maxlen=2000) #조절
        
        #모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(learning_rate = self.learning_rate)
        self.update_target_model()

    #타겟모델 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

      
    def get_action(self, state): #state = curedge
        
        if np.random.rand()<=self.epsilon:
            action = random.randrange(self.action_size)
        else:
            qvalue = self.model(state)
            action = np.argmax(qvalue[0])
            #maxlist = np.argwhere(qlist == np.amax(qvalue))
            #maxlist = maxlist.flatten().tolist()
            #action = random.choice(maxlist)
        return action
    

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch]) 
        rewards = np.array([sample[2] for sample in mini_batch]) 
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch]) 

        model_params = self.model.trainable_variables

        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis = 1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis = -1)
            targets = rewards + (1-dones)*self.discount_factor *max_q
            loss = tf.reduce_mean(tf.square(targets-predicts))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))