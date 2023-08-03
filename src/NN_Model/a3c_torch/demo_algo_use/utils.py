"""
Package of multi-use functions

Author: Ryen Zhao
Date: 2023/Aug/03
"""

import torch
from torch import nn
import numpy


def set_init(layers):
    for layer in layers:
        # initialize parameter
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def v_wrap(np_array, dtype=numpy.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def push_and_pull(opt, lnet, gnet, done, s_, bs, ba, br, gamma):
    if done:
        v_s_ = 0.  # terminal
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]
        # 取出下一个状态s'下values价值数值

    buffer_v_target = []
    for r in br[::-1]:  # reverse buffer r
        v_s_ = r + gamma * v_s_
        # n步折扣价值函数
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()  # 反向列表元素

    loss = lnet.loss_function(
        v_wrap(numpy.vstack(bs)),
        v_wrap(numpy.array(ba), dtype=numpy.int64) if ba[0].dtype == numpy.int64 else v_wrap(numpy.vstack(ba)),
        v_wrap(numpy.array(buffer_v_target)[:, None]))
    # v_wrap()函数是用来将类型转换

    # calculate local gradients and push local parameters to global
    opt.zero_grad()
    loss.backward()
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad
        # 将刚刚计算的梯度传递给全局网路
    opt.step()

    # pull global parameters
    lnet.load_state_dict(gnet.state_dict())


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        "Ep:", global_ep.value,
        name,
        "| Ep_r: %.0f" % global_ep_r.value,
    )
    write("record/record.csv", "a", "%d, %s, %f, %f\n" % (global_ep.value, name, global_ep_r.value, ep_r))

def write(target, type, content):
    f = open(target, type)
    f.write(content)
    f.close()