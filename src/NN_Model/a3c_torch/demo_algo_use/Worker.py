# System Part
import math
import os
import numpy
# Pytorch Part
import torch
import torch.nn as T_nn
import torch.nn.functional as T_nn_Function
import torch.multiprocessing as T_multiprocess

# Neural Network
import NN
# Global Varible
import Varible as Var
# Functions
import utils

import gymnasium as gym


class Worker(T_multiprocess.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, env):
        super(Worker, self).__init__()
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = NN.Neural_Network(Var.N_S, Var.N_A)  # local network
        self.env = gym.make('Pendulum-v1').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < Var.MAX_EP:
            s = self.env.reset()
            s = s[0]
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            for t in range(Var.MAX_EP_STEP):
                if self.name == 'w0':
                    self.env.render()

                a = self.lnet.choose_action(utils.v_wrap(s[None, :]))

                s_, r, done, _, _ = self.env.step(a[0].clip(-2, 2))
                r = float(r)

                if t == Var.MAX_EP_STEP - 1:
                    done = True
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append((r + 8.1) / 8.1)  # normalize

                if total_step % Var.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    utils.push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, Var.GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        utils.record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)

                        break
                s = s_
                total_step += 1



        self.res_queue.put(None)
