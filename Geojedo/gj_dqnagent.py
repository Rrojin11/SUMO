import numpy as np
import random
from collections import defaultdict
from collections import deque
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2.adam import Adam

from gj_dqn import DQN

class dqnAgent():
    def __init__(self, num_seed, edgelists, dict_connection, state_size, action_size,num_episode, dirModel):
        self.edgelists = edgelists
        self.dict_connection = dict_connection
        self.state_size = state_size
        self.action_size = action_size
        self.num_episode = num_episode
        self.dirModel = dirModel
        self.num_seed = num_seed
        #DQN
        self.discount_factor=0.99
        self.learning_rate = 0.001
        self.epsilon=1.0
        self.epsilon_decay=0.999
        self.epsilon_min = 0.05
        self.batch_size = 216    #조절
        self.train_start = 3000 #조절
        self.train_success_rate = 0.7 #조절

        #리플레이 메모리 최대크기 10000
        self.memory = deque(maxlen=10000)
        self.success_memory = deque(maxlen= 10000)
        #모델 생성
        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(learning_rate = self.learning_rate)
        self.update_target_model()

     
    #타겟모델 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

      
    def get_action(self, curlane, state): 
        tmp = self.dict_connection[curlane]
        cnt = self.action_size
        for lane in tmp[::-1]:
            if lane=="":
                cnt-=1
            else:
                break
        if np.random.rand()<=self.epsilon:
            action = random.randrange(cnt)
        else:
            qvalue = self.model(state)
            action = np.argmax(qvalue[0])
            
        return action, self.epsilon



    def save_weights(self):
        self.model.save_weights(self.dirModel+str(self.num_episode)+'_'+str(self.num_seed)+'.h5')

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decaying_epsilon(self):
        if len(self.memory)>max(self.batch_size,self.train_start):
            if self.epsilon>self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def train_model(self):
        if len(self.memory)>max(self.batch_size,self.train_start):
            
            mini_batch_success = random.sample(self.memory, int(self.batch_size*self.train_success_rate))
            mini_batch = random.sample(self.memory, int(self.batch_size*(1-self.train_success_rate)))

            states = np.array([sample[0][0] for sample in mini_batch])
            states = np.append(states,np.array([sample[0][0] for sample in mini_batch_success]), axis = 0)

            actions = np.array([sample[1] for sample in mini_batch]) 
            actions = np.append(actions, np.array([sample[1] for sample in mini_batch_success]))

            rewards = np.array([sample[2] for sample in mini_batch]) 
            rewards = np.append(rewards, np.array([sample[2] for sample in mini_batch_success]))

            next_states = np.array([sample[3][0] for sample in mini_batch])
            next_states = np.append(next_states, np.array([sample[3][0] for sample in mini_batch_success]), axis = 0)

            dones = np.array([sample[4] for sample in mini_batch]) 
            dones = np.append(dones, np.array([sample[4] for sample in mini_batch_success]))
            
            model_params = self.model.trainable_variables

            with tf.GradientTape() as tape:
                predicts = self.model(states)
                one_hot_action = tf.one_hot(actions, self.action_size)
                predicts = tf.reduce_sum(one_hot_action * predicts, axis = 1)

                
                model_predicts = self.model(next_states)
                model_predicts_action= np.argmax(model_predicts, axis = -1) # shape : (#mini_batch,)
                one_hot_next_action = tf.one_hot(model_predicts_action, self.action_size) # shape : (#mini_batch,action_size)

                target_predicts = self.target_model(next_states)

                target_predicts = tf.stop_gradient(target_predicts)

                max_q = tf.reduce_sum(one_hot_next_action * target_predicts, axis=1) # shape : (#mini_batch,)
                targets = rewards + (1-dones)*self.discount_factor *max_q
                loss = tf.reduce_mean(tf.square(targets-predicts))

            grads = tape.gradient(loss, model_params)
            self.optimizer.apply_gradients(zip(grads, model_params))

 