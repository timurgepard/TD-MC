import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD
from collections import deque


import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math

import gym
import pybullet_envs

from collections import deque

from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras import backend as K



class Replay:
    def __init__(self, max_buffer_size, batch_size):
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.pool = deque(maxlen=max_buffer_size)


    def sample(self):
        arr = np.random.default_rng().choice(self.pool, size=self.batch_size, replace=False)
        states_batch = np.vstack(arr[:, 0])
        actions_batch = np.array(list(arr[:, 1]))
        st_dev_batch = np.vstack(arr[:, 2])
        return_batch = np.vstack(arr[:, 4])
        states_next = np.vstack(arr[:, 5])
        gamma_batch = np.vstack(arr[:, 6])
        done_batch = np.vstack(arr[:, 7])
        return states_batch, actions_batch, st_dev_batch, return_batch, states_next, gamma_batch, done_batch



def atanh(x):
    return K.abs(x)*K.tanh(x)

class _actor_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, dtype='float64')
        x = Dense(128, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = LayerNormalization()(x)
        x = concatenate([x, state])
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim),1/np.sqrt(128+self.state_dim)))(x)
        x = LayerNormalization()(x)
        x = concatenate([x, state])
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)
        st_dev = Dense(self.action_dim, activation='sigmoid',kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=[out, tf.clip_by_value(0.5*st_dev, 1e-3, 0.5)])


class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float64')
        action = Input(shape=(self.action_dim,), name='action_input')
        st_dev = Input(shape=(self.action_dim,), name='std_input')
        x = concatenate([state, action])
        x = Dense(128, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim+self.action_dim),1/np.sqrt(self.state_dim+self.action_dim)))(x)
        x = concatenate([x, state, st_dev])
        x = Dense(96, activation=atanh, kernel_initializer=RU(-1/np.sqrt(128+self.state_dim+self.action_dim),1/np.sqrt(128+self.state_dim+self.action_dim)))(x)
        out = Dense(1, activation='linear')(x)
        return Model(inputs=[state, action, st_dev], outputs=out)





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
        self.dist_learning_rate = 0.1*learning_rate

        self.n_episodes = n_episodes
        self.env = env
        self.action_dim = action_dim = env.action_space.shape[0]
        observation_dim = len(env.reset())
        self.state_dim = observation_dim
        self.rewards_norm = divide_rewards_by

        self.cache = []
        self.x = 0.0
        self.eps = 1.0
        self.eps = math.exp(-self.x)
        self.n_steps = 4
        self.tr_step = 2

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
        self.gauss_const = math.log(1/math.sqrt(2*math.pi))


        #############################################
        #----Action based on exploration policy-----#
        #############################################


    def add_noise(self, ANN_t, ANN, scale):
        ANN_t.set_weights(ANN.get_weights())
        for layer in ANN_t.trainable_weights:
            layer.assign_add(np.random.normal(loc=0.0, scale=0.1*scale, size=layer.shape))


    def chose_action(self, state):
        action, st_dev = self.ANN_t(state)
        action =  tf.random.normal([self.action_dim], action[0], st_dev[0])
        self.std.append(st_dev[0])
        return np.tanh(action), st_dev

    #############################################
    # --------------Update Networks--------------#
    #############################################

    def log_prob(self, A, s):
        At = tf.random.normal(A.shape, A, s)
        At = tf.math.tanh(At)
        log_prob = self.gauss_const-tf.math.log(s)-(1/2)*((A-At)/s)**2-tf.math.log(1-At**2+1e-3)
        log_prob = tf.math.reduce_sum(log_prob, axis=1, keepdims=True)
        return log_prob


    def ANN_update(self, ANN, QNN, opt, St):
        with tf.GradientTape(persistent=True) as tape:
            A,s = ANN(St)
            Q = -(QNN([St, A, s])-self.log_prob(A,s))
        dq_da = tape.gradient(Q, [A,s])

        self.dq_da_rec.append(dq_da)
        sma = np.mean(self.dq_da_rec, axis=0)
        dq_da = self.sma_ + (dq_da - sma)
        dq_da = tf.math.abs(dq_da)*tf.math.tanh(dq_da)
        self.sma_ = sma

        da_dtheta = tape.gradient([A,s], ANN.trainable_variables, output_gradients=[dq_da[0],dq_da[1]])
        opt.apply_gradients(zip(da_dtheta, ANN.trainable_variables))


    def NN_update(self,QNN,opt,input,output):
        with tf.GradientTape() as tape:
            L = (1/2)*(output-QNN(input))**2
        gradient = tape.gradient(L, QNN.trainable_variables)
        opt.apply_gradients(zip(gradient, QNN.trainable_variables))


    def TD1(self):
        self.St, self.At, self.st, Rt, St_, gamma, dt = self.replay.sample()
        self.add_noise(self.ANN_t,self.ANN, self.eps)
        self.tow_update(self.QNN_t, self.QNN, 0.005)
        A_,s_ = self.ANN(St_)
        Q_ = self.QNN_t([St_, A_, s_])-self.log_prob(A_,s_)
        self.Q = Rt + (1-dt)*gamma*Q_


    def TD2(self):
        self.tr += 1
        self.NN_update(self.QNN, self.QNN_opt, [self.St, self.At, self.st], self.Q)
        self.ANN_update(self.ANN, self.QNN, self.ANN_opt, self.St)



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


    def eps_step(self, tr):
        self.x += (tr-self.tr_)*self.dist_learning_rate
        self.eps = 0.8*math.exp(-self.x)+0.2
        self.n_steps = round(4/self.eps)
        self.tr_ = tr



    def train(self):
        #with open('Scores.txt', 'w+') as f:
            #f.write('')
        state_dim = len(self.env.reset())
        cnt, score_history, t_history = 0, [], []
        self.td = 0

        for episode in range(self.n_episodes):
            self.std = []
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)


            for t in range(self.max_steps):
                self.env.render(mode="human")
                action, st_dev = self.chose_action(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)
                score += reward
                cnt += 1
                reward /= self.rewards_norm


                self.replay.buffer.append([state, action, st_dev, reward, reward, state_next, self.gamma, done])
                if len(self.replay.buffer)>=1 and t>=1:
                    Return = 0.0
                    t_back = min(t, self.n_steps)
                    for ti in range(-1, -t_back-1, -1):
                        Return = self.gamma*Return + self.replay.buffer[ti][3]
                        self.replay.buffer[ti][4] = Return
                        self.replay.buffer[ti][5] = state_next
                        self.replay.buffer[ti][6] = self.gamma**abs(ti) #1, 2, 3
                        self.replay.buffer[ti][7] = done
                        self.replay.pool.append(self.replay.buffer[ti])

                    if len(self.replay.buffer)>10*self.batch_size:
                        if cnt%(self.tr_step+self.explore_time//cnt)==0:
                            self.td += 1
                            if self.td==1:
                                self.TD1()
                            elif self.td==2:
                                self.TD2()
                                self.td = 0

                if done:
                    break
                else:
                    state = state_next
                    #action = action_next

            self.eps_step(self.tr)
            score_history.append(score)
            t_history.append(t)
            avg_score = np.mean(score_history[-100:])
            avg_t = np.mean(t_history[-100:])
            #with open('Scores.txt', 'a+') as f:
                #f.write(str(score) + '\n')

            if episode>=100 and episode%100==0:
                print('%d: %f, avg %f, | eps %f | std %f | replay buffer size %d | pool size %d | avg steps at ep %d | steps %d' % (episode, score, avg_score, self.eps, np.mean(self.std[10]), len(self.replay.buffer), len(self.replay.pool), avg_t, cnt))


                #self.save()



#env = gym.make('Pendulum-v0').env
#env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('HumanoidMuJoCoEnv-v0').env
#env = gym.make('BipedalWalkerHardcore-v3').env
#env = gym.make('BipedalWalker-v3').env
#env = gym.make('HalfCheetahMuJoCoEnv-v0').env

env = gym.make('HumanoidBulletEnv-v0').env


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 divide_rewards_by = 1,
                 max_buffer_size =128000, # maximum transitions to be stored in buffer
                 batch_size = 128, # batch size for training actor and critic networks
                 max_time_steps = 200,# no of time steps per epoch
                 discount_factor  = 0.99,
                 explore_time = 6400,
                 learning_rate = 0.001,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
