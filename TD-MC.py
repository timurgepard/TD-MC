import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from tensorflow.keras.optimizers import Adam, SGD, Adagrad
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras import Model
from tensorflow.keras import backend as K


import random
import numpy as np
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import math
import gym
from collections import deque
import math



class Replay:
    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)

    def add_experience(self, transition):
        self.buffer.append(transition)

    def sample_batch(self):
        arr = np.array(random.sample(self.buffer, self.batch_size))
        states_batch = np.vstack(arr[:, 0])
        actions_batch = np.array(list(arr[:, 1]))
        rewards_batch = np.vstack(arr[:, 2])
        states_next = np.vstack(arr[:, 3])
        st_dev_batch = np.vstack(arr[:, 4])
        done_batch = np.vstack(arr[:, 5])
        return states_batch, actions_batch, rewards_batch, states_next, st_dev_batch, done_batch

def atanh(x):
    return K.abs(x)*K.tanh(x)

class _actor_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        state = Input(shape=self.state_dim, dtype='float64')
        x = Dense(128, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = concatenate([state, x])
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim),1/np.sqrt(128+self.state_dim)))(x)
        x = concatenate([state, x])
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=out)

class _dist_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_range = action_bound_range

    def model(self):
        state = Input(shape=self.state_dim, dtype='float64')
        x = Dense(32, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        std = Dense(self.action_dim, activation='sigmoid',kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=4*std)

class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float64')
        action = Input(shape=(self.action_dim,), name='action_input')
        x = concatenate([state, action])
        x = Dense(128, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim+self.action_dim),1/np.sqrt(self.state_dim+self.action_dim)))(x)
        std = Input(shape=(self.action_dim,), name='std_input')
        x = concatenate([x, state, std])
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim+self.action_dim),1/np.sqrt(128+self.state_dim+self.action_dim)))(x)
        out = Dense(1, activation='linear')(x)
        return Model(inputs=[state, action, std], outputs=out)

class DDPG():
    def __init__(self,
                 env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 divide_rewards_by = 1,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 discount_factor  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.gamma = discount_factor  ## discount factor
        self.explore_time = explore_time
        self.act_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.n_episodes = n_episodes
        self.rewards_norm = divide_rewards_by

        self.env = env
        self.action_dim = action_dim = env.action_space.shape[0]

        self.cache = []
        self.x = 0.0
        self.eps = 1.0

        observation_dim = len(env.reset())
        self.state_dim = state_dim = observation_dim

        self.max_steps = max_time_steps  ## Time limit for a episode

        self.ANN_Adam = Adam(self.act_learning_rate)
        self.sNN_Adam = Adagrad(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)

        self.replay = Replay(self.max_buffer_size, self.batch_size)

        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.sNN = _dist_network(self.state_dim, self.action_dim).model()
        self.QNN = _critic_network(self.state_dim, self.action_dim).model()

        self.ANN_t = _actor_network(self.state_dim, self.action_dim).model()
        self.sNN_t = _dist_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _critic_network(self.state_dim, self.action_dim).model()

        self.ANN_t.set_weights(self.ANN.get_weights())
        self.sNN_t.set_weights(self.sNN.get_weights())
        self.QNN_t.set_weights(self.QNN.get_weights())

        self.dq_da_rec = deque(maxlen=10)
        self.sma_ = 0.0
        self.tr = 0
        self.tr_ = 0
        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def forward(self, state):
        action = self.ANN(state)
        st_dev = self.sNN(state)
        action = action[0] + tf.random.normal([self.action_dim], 0.0, st_dev[0])
        return np.clip(action, -1.0, 1.0), st_dev[0]




    #############################################
    # --------------Update Networks--------------#
    #############################################

    def ANN_update(self, ANN, sNN, QNN, opt1, opt2, St):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            s = sNN(St)
            Q = QNN([St, A, s])
        dq_dp = tape.gradient(Q, [A,s])

        self.dq_da_rec.append(dq_dp[0])
        sma = np.mean(self.dq_da_rec, axis=0)
        dq_da = self.sma_ + (dq_dp[0]-sma)
        dq_da = tf.math.abs(dq_da)*tf.math.tanh(dq_da)
        dq_ds = dq_dp[1]
        self.sma_ = sma

        da_dtheta = tape.gradient(A, ANN.trainable_variables, output_gradients=-dq_da)
        opt1.apply_gradients(zip(da_dtheta, ANN.trainable_variables))
        
        dstd_dw = tape.gradient(s, sNN.trainable_variables, output_gradients=-dq_ds)
        opt2.apply_gradients(zip(dstd_dw, sNN.trainable_variables))

    def NN_update(self,QNN,input,output):
        with tf.GradientTape() as tape:
            L = (1/2)*(output-QNN(input))**2
        gradient = tape.gradient(L, QNN.trainable_variables)
        self.QNN_Adam.apply_gradients(zip(gradient, QNN.trainable_variables))


    def TD(self):
        self.tr += 1
        St, At, rt, St_, st, dt = self.replay.sample_batch()
        self.update_target()
        A_ = self.ANN_t(St_)
        s_ = self.sNN_t(St_)
        Q_ = self.QNN_t([St_,A_,s_])
        Q = rt + (1-dt)*self.gamma*Q_
        self.NN_update(self.QNN, [St,At,st], Q)
        self.ANN_update(self.ANN, self.sNN, self.QNN, self.ANN_Adam, self.sNN_Adam, St)


    def update_target(self):
        self.tow_update(self.ANN_t, self.ANN, 0.001)
        self.tow_update(self.sNN_t, self.sNN, 0.001)
        self.tow_update(self.QNN_t, self.QNN, 0.001)

    def tow_update(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target


    def save(self):
        self.ANN.save('./models/actor.h5')
        self.QNN.save('./models/critic_pred.h5')



    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')
        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []

        for episode in range(self.n_episodes):
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            done_cnt = 0
            Rt = 0.0
            for t in range(self.max_steps):
                action, st_dev = self.forward(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)
                self.replay.buffer.append([state, action, reward/self.rewards_norm, state_next, st_dev, done])
                cnt += 1
                score += reward
                if done or t==self.max_steps-1:
                    break
                else:
                    #self.env.render(mode="human")
                    if len(self.replay.buffer)>self.batch_size:
                        if cnt%(1+self.explore_time//cnt)==0:
                            self.TD()
               
                state = state_next

            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')


            if episode>=50 and episode%50==0:
                self.save()
                print('%d: %f, %f, | %f | step %d' % (episode, score, avg_score, np.mean(st_dev), cnt))

            




#env = gym.make('Pendulum-v1').env
#env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('HumanoidMuJoCoEnv-v0').env
#env = gym.make('BipedalWalkerHardcore-v3').env
env = gym.make('BipedalWalker-v3').env
#env = gym.make('HalfCheetahMuJoCoEnv-v0').env
#env = gym.make('HumanoidPyBulletEnv-v0').env

ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 divide_rewards_by = 10,
                 max_buffer_size =128000, # maximum transitions to be stored in buffer
                 batch_size = 128, # batch size for training actor and critic networks
                 max_time_steps = 200,# no of time steps per epoch
                 discount_factor  = 0.99,
                 explore_time = 10000,
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
