'''
Reinforcement Learning (A3C)
Engine: Pytorch
Process: multi

File Name: main
Describe:
    For test a3c_torch model.
        NN.py  | the neural Network
        Worker.py | the Worker Setting

Athor: Ryen Zhao
Date: 2023/Aug/03
'''

"""
import part
"""
# System Part
from datetime import datetime
import math
import os
# Pytorch Part
import torch
import torch.nn as T_nn
import torch.nn.functional as T_nn_Function
import torch.multiprocessing as T_multiprocess
import matplotlib.pyplot as plt

# A3C
from NN import Neural_Network
from shared_adam import SharedAdam
from Worker import Worker
import Varible as Var
import utils
if __name__ == "__main__":
    start_time = datetime.now()


    open("record/record.csv", "w")
    utils.write("record/record.csv", "w", "global_ep, name, global_ep_r, ep_r\n")
    gnet = Neural_Network(Var.N_S, Var.N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.95, 0.999))  # global optimizer
    global_ep, global_ep_r, res_queue = T_multiprocess.Value('i', 0), T_multiprocess.Value('d', 0.), T_multiprocess.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, Var.env) for i in range(T_multiprocess.cpu_count())]

    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    End_time = datetime.now()
    print("Time usage", (End_time-start_time))
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()

