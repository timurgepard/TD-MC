import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp


import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from buffer import Record
from actor_critic import _actor_network,_critic_network
import math

import gym
#import gym_vrep
import pybulletgym
import time


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
                 clip = 25,
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

        self.n_steps = clip
        self.max_steps = max_time_steps  ## Time limit for a episode

        self.ANN_Adam = Adam(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)

        self.record = Record(self.max_buffer_size, self.batch_size)

        self.ANN = _actor_network(self.state_dim, self.action_dim).model()
        self.QNN = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_t = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_t.set_weights(self.QNN.get_weights())


        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def forward(self, tstate):
        action = self.ANN(tstate)[0]
        eps = max(self.eps, 0.1)
        if random.uniform(0.0, 1.0)<self.eps:
            action += tf.random.normal([self.action_dim], 0.0, 2*eps)
        return np.clip(action, -1.0, 1.0)


    def update_buffer(self):
        active_steps = len(self.cache) - self.n_steps
        if active_steps>0:
            for t, (St,At,Rt,St_) in enumerate(self.cache):
                if t<active_steps:
                    Ql = Qt = 0.0
                    for k in range(t, t+self.n_steps):
                        Qt += self.gamma**(k-t)*self.cache[k][2]
                        if k<self.n_steps: Ql += 0.1*0.9**k*Qt
                    Qt_ = (Qt - Rt)/self.gamma + self.gamma**self.n_steps*self.cache[self.n_steps][2]
                    Ql_ = Ql +  0.1*0.9**self.n_steps*Qt + 0.9**(self.n_steps+1)*Qt_
                    Ql += 0.9**self.n_steps*Qt
                    self.record.add_experience([St,At,Rt,Ql,St_,Ql_])
            self.cache = self.cache[-self.n_steps:]


    #############################################
    # --------------Update Networks--------------#
    #############################################

    def ANN_update(self, ANN, QNN, opt, St):
        with tf.GradientTape(persistent=True) as tape:
            A = ANN(St)
            Q = QNN([St, A])
        dq_da = tape.gradient(Q, A)
        dq_da = tf.math.abs(dq_da)*tf.math.tanh(dq_da/2)
        da_dtheta = tape.gradient(A, ANN.trainable_variables, output_gradients=-dq_da)
        opt.apply_gradients(zip(da_dtheta, ANN.trainable_variables))


    def NN_update(self,QNN,input,output):
        with tf.GradientTape() as tape:
            mse = (1/2)*(output-QNN(input))**2
        gradient = tape.gradient(mse, QNN.trainable_variables)
        self.QNN_Adam.apply_gradients(zip(gradient, QNN.trainable_variables))


    def TD_1(self):
        self.eps_step()
        self.St, self.At, self.rt, self.Ql, self.St_, self.Ql_ = self.record.sample_batch()
        self.QNN_t.set_weights(self.QNN.get_weights())
        self.NN_update(self.QNN_t, [self.St, self.At], self.Ql)
        self.ANN_update(self.ANN, self.QNN_t, self.ANN_Adam, self.St)

    def TD_2(self):
        A_ = self.ANN(self.St_)
        Q_ = self.QNN_t([self.St_, A_])
        Qt = self.rt + self.gamma*(Q_+self.Ql_)/2
        self.NN_update(self.QNN, [self.St, self.At], Qt)
        self.ANN_update(self.ANN, self.QNN, self.ANN_Adam, self.St)



    def save(self):
        self.ANN.save('./models/actor.h5')
        self.QNN.save('./models/critic_pred.h5')
        self.QNN_t.save('./models/critic_target.h5')


    def eps_step(self):
        self.x += 0.5*self.act_learning_rate
        self.eps = math.exp(-self.x)*math.cos(self.x)


    def train(self):
        with open('Scores.txt', 'w+') as f:
            f.write('')
        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []
        self.td = 0
        Rt = 0.0
        for episode in range(self.n_episodes):
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            done_cnt = 0
            for t in range(self.max_steps+self.n_steps):

                action = self.forward(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)

                if done or t==self.max_steps-1:
                    if Rt == 0.0: Rt = reward
                    if abs(Rt)>50*abs(score/t):
                        reward = Rt/25
                    if done_cnt>self.n_steps:
                        break
                    else:
                        done_cnt += 1
                else:
                    self.env.render(mode="human")
                    cnt += 1
                    score += reward
                    if cnt%10 == 0:
                        self.update_buffer()

                    if len(self.record.buffer)>self.batch_size:
                        if cnt%(1+self.explore_time//cnt)==0:
                            self.td += 1
                            if self.td==1:
                                self.TD_1()
                            elif self.td==2:
                                self.TD_2()
                                self.td = 0


                self.cache.append([state, action, reward/self.rewards_norm, state_next])
                state = state_next


            self.update_buffer()
            self.cache = []

            score_history.append(score+Rt)
            avg_score = np.mean(score_history[-10:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')


            if episode>=10 and episode%10==0:
                print('%d: %f, %f, | %f | record size %d' % (episode, score, avg_score, self.eps, len(self.record.buffer)))
                self.save()








    def test(self):

        with open('Scores.txt', 'w+') as f:
            f.write('')

        self.eps = 0.0
        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []

        for episode in range(self.n_episodes):

            self.episode = episode

            done = False
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)

            t = 0
            done_cnt = 0

            for t in range(self.max_steps):

                self.env.render()

                action = self.forward(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)


                if done:
                    if reward <=-100 or reward >=100:
                        reward = reward/self.n_steps

                    if done_cnt>self.n_steps:
                        done_cnt = 0
                        break
                    else:
                        done_cnt += 1

                score += reward
                state = state_next

                cnt += 1


            score_history.append(score)
            avg_score = np.mean(score_history[-10:])

            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f ' % (episode, score, avg_score))


#env = gym.make('Pendulum-v1').env
#env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('HumanoidMuJoCoEnv-v0').env
#env = gym.make('BipedalWalkerHardcore-v3').env
#env = gym.make('BipedalWalker-v3').env
#env = gym.make('HalfCheetahMuJoCoEnv-v0').env
env = gym.make('HumanoidPyBulletEnv-v0').env

ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 divide_rewards_by = 10000,
                 max_buffer_size =2000000, # maximum transitions to be stored in buffer
                 batch_size = 100, # batch size for training actor and critic networks
                 max_time_steps = 2000,# no of time steps per epoch
                 clip = 700,
                 discount_factor  = 0.99,
                 explore_time = 10000,
                 actor_learning_rate = 0.0002,
                 critic_learning_rate = 0.002,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
