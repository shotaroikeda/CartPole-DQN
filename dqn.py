## Imports and Constants
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym

import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


## DQN
class BasicNet(nn.Module):
    def __init__(self, shape=(4,64,2)):
        super(BasicNet, self).__init__()
        i, h1, o = shape
        self.linear1 = nn.Linear(i,h1)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(h1,o)


    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)        # F.relu() works as well
        x = self.linear2(x)

        return x

NUM_EPOCHS = 10000
DISCOUNT_FACTOR = 0.9
env = gym.make('CartPole-v1')
net = BasicNet()

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

time_step = 0

## Test stuff

def np_to_var(arr):
    return Variable(torch.Tensor(arr).view(1,-1))

s = env.reset()

## Test Step
import time
def test(render=False):
    reward = 0
    s = env.reset()
    s = Variable(torch.Tensor(s).view(1,-1))
    done = False
    while not done:
        if render: env.render()

        q_vals_var = net(s)
        q_vals, act_idx = q_vals_var.max(1)
        act_idx = act_idx.data[0,0]

        s2, r, done, _ = env.step(act_idx)
        s = Variable(torch.Tensor(s2).view(1,-1))

        reward += r

    return reward


## One Step
def _train_step(env, time_step,
                reward=0.0, s=None, render=False, debug=False):
    if not s:
        s = env.reset()
        s = Variable(torch.Tensor(s).view(1,-1), requires_grad=True)

    if render:
        env.render()

    q_vals_var = net(s)
    if debug: print(q_vals_var)
    q_vals, act_idx = q_vals_var.max(1)
    act_idx = act_idx.data[0,0]

    epsilon = 99.0 / ((time_step/1000.0)+100.0)
    if debug:
        print("Epsilon %10.5f" % (epsilon))
    if random.random() < epsilon:
        act_idx = env.action_space.sample()

    s2, r, done, _ = env.step(act_idx)
    reward+=r

    s = np_to_var(s2)
    r = Variable(torch.Tensor([r]))
    # Q-Learning
    if not done:
        future_q_val = net(s)
        best_q_val = future_q_val.max(1)[0].data[0,0]

        q_target = r + DISCOUNT_FACTOR*best_q_val
        loss = F.smooth_l1_loss(q_vals, q_target)
        if debug:
            print("Predicted: %s Actual: %s" % (q_vals, q_target))
    else:
        predicted_q = net(s)
        best_q = predicted_q.max(1)[0]
        r = Variable(torch.Tensor([-reward*0.5-10])) # Hardcode the loss here
        if debug:
            print("Predicted: %s Actual: %s" % (best_q, r))
        loss = F.smooth_l1_loss(best_q, r)
        s = None

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return s, reward

## Debug
def debug(env, time_step):
    s = _train_step(env, time_step, debug=True)
    while s is not None:
        time_step += 1
        s = _train_step(env, time_step, s=s, debug=True)

    print(test())
    return time_step

## Randomized Agent
# Plot the scores the agent gets
pts = np.zeros(NUM_EPOCHS // 100)
s = None
for i in range(NUM_EPOCHS):
    s, reward = _train_step(env, time_step, s=s)
    while s is not None:
        time_step += 1
        s, reward = _train_step(env, time_step, reward=reward, s=s)

    if i % 100 == 0:
        pts[i//100] = test(render=True)
        print(pts[i//100])

plt.plot(np.arange(len(pts)), pts, label='Scores', color='red')
ax = plt.axes()
ax.set_xlabel('Epoch (10^2)')
ax.set_ylabel('Score')
ax.set_title('Scores at Various Epoches')
plt.legend(loc='upper left')
plt.savefig('plot.png', dpi=400, bbox_inches='tight')
