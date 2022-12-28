import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD
from collections import deque


import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#from buffer import Replay
#from actor_critic import _actor_network,_critic_network
import math

import gym
import pybullet_envs
import time

#from ou_noise import OUActionNoise

from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate
#from tensorflow_addons.layers import NoisyDense
from tensorflow.keras import Model
from tensorflow.keras import backend as K


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
        x = concatenate([x, state])
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim),1/np.sqrt(128+self.state_dim)))(x)
        x = concatenate([x, state])
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=out)


class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float64')
        x = Dense(128, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        action = Input(shape=(self.action_dim,), name='action_input')
        x = concatenate([x, state, action])
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim+self.action_dim),1/np.sqrt(128+self.state_dim+self.action_dim)))(x)
        x = concatenate([x, state, action])
        out = Dense(1, activation='linear')(x)
        return Model(inputs=[state, action], outputs=out)


class Replay:
    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.pool = deque(maxlen=2*max_buffer_size)

    def sample(self):
        arr = np.random.default_rng().choice(self.pool, size=self.batch_size, replace=False)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = np.array(list(arr[:, 1]))
        #rewards_batch = np.vstack(arr[:, 2])
        return_batch = np.vstack(arr[:, 3])
        states_next = np.vstack(arr[:, 4])
        gamma_batch = np.vstack(arr[:, 5])
        return states_batch, actions_batch, return_batch, states_next, gamma_batch

class DDPG():
    def __init__(self,
                 env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 divide_rewards_by=1,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 discount_factor  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.gamma = discount_factor  ## discount factor
        self.explore_time = explore_time

        self.critic_learning_rate = learning_rate
        self.act_learning_rate = 0.1*learning_rate
        self.dist_learning_rate = 0.05*learning_rate

        self.n_episodes = n_episodes
        self.env = env
        self.action_dim = action_dim = env.action_space.shape[0]
        observation_dim = len(env.reset())
        self.state_dim = observation_dim
        self.rewards_norm = divide_rewards_by

        self.cache = []
        self.x = 0.0
        self.eps = math.exp(-self.x)
        self.tr_step = 2
        self.n_steps = 4#round(4/self.eps)
        self.horizon = 64


        self.max_steps = max_time_steps
        self.tr = 0
        self.tr_ = 0



        self.ANN_opt = Adam(self.act_learning_rate)
        self.QNN_opt = Adam(self.critic_learning_rate)
        self.replay = Replay(self.max_buffer_size, self.batch_size)
        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN = _critic_network(self.state_dim, self.action_dim).model()
        self.ANN_t = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _critic_network(self.state_dim, self.action_dim).model()
        self.ANN_t.set_weights(self.ANN.get_weights())
        self.QNN_t.set_weights(self.QNN.get_weights())
        self.dq_da_rec, self.sma_ = deque(maxlen=10), 0.0

        #self.action_noise = OUActionNoise(action_dim)

        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def chose_action(self, state):
        action = self.ANN(state)[0]
        if random.uniform(0.0, 1.0)<self.eps:
            action += tf.random.normal([self.action_dim], 0.0, 2*self.eps)
            #action += tf.random.normal([self.action_dim], 0.0, 10*self.eps)
        return np.clip(action, -1.0, 1.0)

    #############################################
    # --------------Update Networks--------------#
    #############################################

    def ANN_update(self, ANN, QNN, opt, St):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            tape.watch(A)
            Q = QNN([St, A])
        dq_da = tape.gradient(Q, A)

        self.dq_da_rec.append(dq_da)
        sma = np.mean(self.dq_da_rec, axis=0)
        dq_da = self.sma_ + (dq_da - sma)
        dq_da = tf.math.abs(dq_da)*tf.math.tanh(dq_da)
        self.sma_ = sma

        da_dtheta = tape.gradient(A, ANN.trainable_variables, output_gradients=-dq_da)
        opt.apply_gradients(zip(da_dtheta, ANN.trainable_variables))


    def NN_update(self,QNN,opt,input,output):
        with tf.GradientTape() as tape:
            L = (1/2)*(output-QNN(input))**2
        gradient = tape.gradient(L, QNN.trainable_variables)
        opt.apply_gradients(zip(gradient, QNN.trainable_variables))


    def TD(self):
        self.tr += 1
        St, At, Rt, St_, gamma = self.replay.sample()
        self.update_target()
        A_ = self.ANN_t(St_)
        Q_ = self.QNN_t([St_, A_])
        Q = Rt + gamma*Q_
        self.NN_update(self.QNN, self.QNN_opt, [St, At], Q)
        self.ANN_update(self.ANN, self.QNN, self.ANN_opt, St)

    def update_target(self):
        self.tow_update(self.ANN_t, self.ANN, 0.005)
        self.tow_update(self.QNN_t, self.QNN, 0.005)

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
        self.ANN_t.save('./models/actor_target.h5')
        self.QNN.save('./models/critic_pred.h5')
        self.QNN_t.save('./models/critic_target.h5')


    def eps_step(self, tr):
        self.x += (tr-self.tr_)*self.dist_learning_rate
        self.eps = 0.75*math.exp(-self.x)+0.25
        self.n_steps = round(4/self.eps)
        self.tr_ = tr



    def train(self):
        #with open('Scores.txt', 'w+') as f:
            #f.write('')
        state_dim = len(self.env.reset())
        cnt, score_history = 0, []
        for episode in range(self.n_episodes):
            score = 0.0
            end, end_cnt, terminal_reward = False, 0, 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            for t in range(self.max_steps+self.n_steps):

                if not end:
                    #self.env.render(mode="human")
                    action = self.chose_action(state)
                    state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                    state_next = np.array(state_next).reshape(1, state_dim)
                    score += reward
                    reward /= self.rewards_norm
                    if done or t>=self.max_steps-1: end = True
                    cnt += 1

                    if len(self.replay.buffer)>self.batch_size and cnt%(self.tr_step+self.explore_time//cnt)==0:
                        self.TD()
                else:
                    if done: reward=reward/self.horizon
                    if end_cnt>=self.horizon:
                        for i in range(self.horizon-1):
                            del self.replay.buffer[-1]
                        break
                    end_cnt += 1


                self.replay.buffer.append([state, action, reward, reward, state_next, self.gamma])
                if len(self.replay.buffer)>=1 and t>=self.n_steps:
                    Return = 0.0
                    t_back = min(t, self.horizon)
                    for ti in range(-1, -t_back, -1):
                        i = -(ti+1) # 0,1,2...
                        Return = self.gamma**i*Return + self.replay.buffer[ti][2]
                        self.replay.buffer[ti][3] = Return
                        self.replay.buffer[ti][4] = state_next
                        self.replay.buffer[ti][5] = self.gamma**(i+1) #i+1: 1, 2, 3
                        self.replay.pool.append(self.replay.buffer[ti])


                if not end: state = state_next


            self.eps_step(self.tr)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            #with open('Scores.txt', 'a+') as f:
                #f.write(str(score) + '\n')

            if episode>=50 and episode%50==0:
                #self.save()
                print('%d: %f, %f, | %f | replay size %d | step %d' % (episode, score, avg_score, self.eps, len(self.replay.buffer), self.n_steps))
                #self.action_noise.reset()


#env = gym.make('Pendulum-v1').env
#env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('BipedalWalker-v3').env
env = gym.make('HumanoidBulletEnv-v0').env


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 divide_rewards_by = 1,
                 max_buffer_size =256000, # maximum transitions to be stored in buffer
                 batch_size = 128, # batch size for training actor and critic networks
                 max_time_steps = 200,# no of time steps per epoch
                 discount_factor  = 0.99,
                 explore_time = 12800,
                 learning_rate = 0.002,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
