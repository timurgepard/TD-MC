import tensorflow as tf
import multiprocessing as mp
import ctypes
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform as RU
from tensorflow.keras.layers import Dense, Input, concatenate, LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras import backend as K

import random
import numpy as np
import math
from collections import deque

import gym
import pybullet_envs

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# single GPU accelaration is not so efficient because GPU bus is involved
# If necessary and CUDA fully installed, to enable GPU processing comment above



class Replay:
    def __init__(self, max_buffer_size, max_time_steps, batch_size):
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_time_steps)
        self.pool = deque(maxlen=max_buffer_size)
        self.priorities = deque(maxlen=max_buffer_size)
        self.indexes = []


    def add_experience(self, transition):
        self.pool.append(transition)
        self.priorities.append(1.0)
        ln = len(self.pool)
        if ln <= self.max_buffer_size: self.indexes.append(ln-1)

    def add_priorities(self, indices,priorities):
        for idx,priority in zip(indices,priorities):
            self.priorities[idx]=priority[0].numpy()

    def sample(self):
        if len(self.pool)>100*self.batch_size:
            #sampled PER, takes bigger sample from population, then takes weighted batch, like net fishing
            sampled_idxs = random.sample(self.indexes, 100*self.batch_size)
            indices = random.choices(sampled_idxs, k=self.batch_size, weights=[self.priorities[indx] for indx in sampled_idxs])
        else:
            #random sample
            indices = random.sample(self.indexes, self.batch_size)

        batch = [self.pool[indx] for indx in indices]
        states, actions, st_devs, rewards, returns, next_states, gammas, dones = zip(*batch)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        st_devs = tf.convert_to_tensor(st_devs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        gammas = tf.convert_to_tensor(gammas, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        return states, actions, st_devs, returns, next_states, gammas, dones, indices



def atanh(x):
    return K.abs(x)*K.tanh(x)

class _actor_network():
    def __init__(self, state_dim, action_dim,action_bound_range=1):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, dtype='float64')
        x = Dense(256, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim),1/np.sqrt(self.state_dim)))(state)
        x = concatenate([x, state])
        x = Dense(192, activation=atanh, kernel_initializer=RU(-1/np.sqrt(256+self.state_dim),1/np.sqrt(256+self.state_dim)))(x)
        x = concatenate([x, state])
        out = Dense(self.action_dim, activation='tanh',kernel_initializer=RU(-0.003,0.003))(x)
        st_dev = Dense(self.action_dim, activation='softplus',kernel_initializer=RU(-0.003,0.003))(x)
        return Model(inputs=state, outputs=[out, tf.clip_by_value(st_dev, 1e-6, 2)])


class _critic_network():
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def model(self):
        state = Input(shape=self.state_dim, name='state_input', dtype='float64')
        action = Input(shape=self.action_dim, name='action_input')
        st_dev = Input(shape=self.action_dim, name='std_input')
        x = concatenate([state, action])
        x = Dense(256, activation=atanh, kernel_initializer=RU(-1/np.sqrt(self.state_dim+self.action_dim),1/np.sqrt(self.state_dim+self.action_dim)))(x)
        x = concatenate([x, state, st_dev])
        x = Dense(192, activation=atanh, kernel_initializer=RU(-1/np.sqrt(256+self.state_dim+self.action_dim),1/np.sqrt(256+self.state_dim+self.action_dim)))(x)
        out = Dense(1, activation='linear')(x)
        return Model(inputs=[state, action, st_dev], outputs=out)



class DDPG():
    def __init__(self,
                 env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 discount_factor=0.99,
                 max_buffer_size =64000, # maximum transitions to be stored in buffer
                 batch_size =128, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 explore_time = 12800, # time steps for random actions for exploration
                 learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.gamma = discount_factor
        self.explore_time = explore_time

        self.critic_learning_rate = learning_rate
        self.act_learning_rate = 0.1*learning_rate
        self.dist_learning_rate = 0.01*learning_rate

        self.n_episodes = n_episodes
        self.env = env
        self.action_dim = action_dim = env.action_space.shape[0]
        observation_dim = len(env.reset())
        self.state_dim = observation_dim

        self.x = 0.0
        self.eps = 1.0
        self.n_steps = round(4/self.eps) #4 steps
        self.tr_step = 2

        self.max_steps = max_time_steps
        self.tr = 0
        self.tr_ = 0


        self.ANN_opt = Adam(self.act_learning_rate)
        self.QNN_opt = Adam(self.critic_learning_rate)
        self.replay = Replay(self.max_buffer_size, self.max_steps, self.batch_size)
        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.ANN_ = _actor_network(self.state_dim, self.action_dim).model() #actor with parametric noise
        self.ANN_t = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _critic_network(self.state_dim, self.action_dim).model()
        self.ANN_t.set_weights(self.ANN.get_weights())
        self.QNN_t.set_weights(self.QNN.get_weights())
        self.dq_da_rec, self.sma_ = deque(maxlen=10), 0.0 #for SMA smoothing actor's update
        self.p, self.dq_da = 4.0, 0.0
        self.gauss_const = math.log(1/math.sqrt(2*math.pi)) # for log_prob calculation

    def save(self):
        self.ANN.save('./models/actor.h5')
        self.ANN_t.save('./models/actor_target.h5')
        self.QNN.save('./models/critic.h5')
        self.QNN_t.save('./models/critic_target.h5')

    #############################################
    #----Action based on exploration policy-----#
    #############################################

    def add_noise(self, ANN_, ANN, scale):
        ANN_.set_weights(ANN.get_weights())
        for layer in ANN_.trainable_weights:
            layer.assign_add(np.random.normal(loc=0.0, scale=0.0777*scale, size=layer.shape))

    # at training steps Gaussian Noise (to work with -log_prob), else parametric noise with epsilon decrease in ANN_ during gradient update
    def chose_action(self, state, cnt):
        action, st_dev = self.ANN_(state)
        action = action[0]
        if cnt%self.tr_step==0: action += tf.random.normal([self.action_dim], 0.0, st_dev[0])

        self.std.append(st_dev[0]) #logging
        return tf.math.tanh(action), st_dev[0]
        #return np.clip(action, -1.0, 1.0), st_dev[0]

    #############################################
    # --------------Update Networks--------------#
    #############################################

    def log_prob(self, A, s):
        At = tf.random.normal(A.shape, A, s)
        Att = tf.math.tanh(At)
        log_prob = self.gauss_const-tf.math.log(s)-(1/2)*((A-Att)/s)**2-tf.math.log(1-Att**2+1e-6)
        log_prob = tf.math.reduce_sum(log_prob, axis=1, keepdims=True)
        return log_prob


    def tow_update(self, target, online, tow):
        init_weights = online.get_weights()
        update_weights = target.get_weights()
        weights = []
        for i in tf.range(len(init_weights)):
            weights.append(tow * init_weights[i] + (1 - tow) * update_weights[i])
        target.set_weights(weights)
        return target

    #Kalman filter with abs*tanh rectifier
    def Kalman_filter(self, dq_da, s):
        dq_da = tf.convert_to_tensor(dq_da, dtype=tf.float32)
        p = self.p + s
        K = p/(p+self.eps)
        dq_da = self.dq_da + K*(dq_da-self.dq_da)
        self.dq_da = tf.math.abs(dq_da)*tf.math.tanh(dq_da)
        self.p = (1-K)*self.p
        return self.dq_da

    # Timur filter, sma_prev + (dq_da - sma), sma for 10
    def sma_filter(self, dq_da):
        self.dq_da_rec.append(dq_da)
        sma = tf.reduce_mean(list(self.dq_da_rec), axis=0)
        dq_da = self.sma_ + (dq_da - sma)
        dq_da = tf.math.abs(dq_da)*tf.math.tanh(dq_da)
        self.sma_, self.dq_da_rec[-1] = sma, dq_da
        return dq_da

    def TD_Sutton(self):
        self.tr += 1
        St, At, st, Rt, St_, gamma, dt, idx = self.replay.sample()
        self.tow_update(self.ANN_t, self.ANN, 0.005)
        self.tow_update(self.QNN_t, self.QNN, 0.005)
        A_,s_ = self.ANN_t(St_)
        Q_ = self.QNN_t([St_, A_, s_])-self.log_prob(A_,s_)
        Q = Rt + (1-dt)*gamma*Q_

        #DDPG critic network regression to target but with -log_prob and critic taking st_dev as input
        with tf.GradientTape() as tape:
            se = (1/2)*(Q-self.QNN([St, At, st]))**2
            mse = tf.math.reduce_mean(se, axis=0)
        self.replay.add_priorities(idx,se)
        gradient = tape.gradient(mse, self.QNN.trainable_variables)
        self.QNN_opt.apply_gradients(zip(gradient, self.QNN.trainable_variables))

        #DDPG's actor gradient update but with -log_prob, critic taking st_dev as input and smoothing:
        with tf.GradientTape(persistent=True) as tape:
            A,s = self.ANN(St)
            Q = (self.QNN([St, A, s])-self.log_prob(A,s))
            Q = -tf.math.reduce_mean(Q, axis=0, keepdims=True)
        dq_da = tape.gradient(Q, [A,s])
        dq_da = self.Kalman_filter(dq_da,s)
        da_dtheta = tape.gradient([A,s], self.ANN.trainable_variables, output_gradients=[dq_da[0],dq_da[1]])
        self.ANN_opt.apply_gradients(zip(da_dtheta, self.ANN.trainable_variables))
        self.add_noise(self.ANN_,self.ANN, self.eps)



    # epsilon decrease is episode wise but depends on how many training steps were at the last episode
    def eps_step(self, tr):
        self.x += (tr-self.tr_)*self.dist_learning_rate
        self.eps = 0.75*math.exp(-self.x)+0.25 # 0.25 is some noise at the end
        self.n_steps = round(4/self.eps) # n-steps increases from 4 to 16
        self.tr_ = tr


    def train(self):
        #with open('Scores.txt', 'w+') as f:
            #f.write('')
        state_dim = len(self.env.reset())
        cnt, score_history, t_history = 0, [], []

        for episode in range(self.n_episodes):
            self.std = []
            score = 0.0
            state = tf.convert_to_tensor([self.env.reset()], dtype=tf.float32)
            for t in range(self.max_steps):
                #self.env.render(mode="human")

                action, st_dev = self.chose_action(state, cnt)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = tf.convert_to_tensor([state_next], dtype=tf.float32)
                score += reward
                cnt += 1
                self.replay.buffer.append([state[0], action, st_dev, reward, reward, state_next[0], self.gamma, done])
                if len(self.replay.buffer)>=1:
                    Return = 0.0
                    t_back = min(t, self.n_steps)
                    for ti in range(-1, -t_back-2, -1):
                        Return = self.gamma*Return + self.replay.buffer[ti][3]
                        self.replay.buffer[ti][4] = tf.convert_to_tensor([Return], dtype=tf.float32)
                        self.replay.buffer[ti][5] = state_next[0]
                        self.replay.buffer[ti][6] = tf.convert_to_tensor([self.gamma**abs(ti)], dtype=tf.float32) #1, 2, 3
                        self.replay.buffer[ti][7] = tf.convert_to_tensor([done], dtype=tf.float32)
                        self.replay.add_experience(self.replay.buffer[ti])

                    if len(self.replay.pool)>self.batch_size:
                        if cnt%(self.tr_step+self.explore_time//cnt)==0:
                            self.TD_Sutton()

                if done: break
                state = state_next


            self.eps_step(self.tr)
            score_history.append(score)
            t_history.append(t)
            #with open('Scores.txt', 'a+') as f:
                #f.write(str(score) + '\n')
            if episode>=50 and episode%50==0:
                print('%d: %f, avg %f, | eps %f | std %f | replay buffer size %d | pool size %d | avg steps at ep %d | steps %d' % (episode, score, np.mean(score_history[-100:]), self.eps, np.mean(self.std), len(self.replay.buffer), len(self.replay.pool), np.mean(t_history[-100:]), cnt))
                self.save()

        queue.close()
        queue.join_thread()
        p.join()

env = gym.make('HumanoidBulletEnv-v0').env
#env = gym.make('BipedalWalker-v3').env


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 discount_factor=0.99,
                 max_buffer_size =2000000, # maximum transitions to be stored in buffer
                 batch_size = 128, # batch size for training actor and critic networks
                 max_time_steps = 200,# no of time steps per epoch
                 explore_time = 6400,
                 learning_rate = 0.002,
                 n_episodes = 1000000) # no of episodes to run

ddpg.train()
