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
matplotlib.use('AGG')

import matplotlib.pyplot as plt
import numpy as np

NUM_EPOCHS = 30000
DISCOUNT_FACTOR = 0.9
GPU = True
LOAD_NET = False

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

    def save(self, fname="data/model_state_dict.pkl"):
        model = self.state_dict()
        with open(fname, 'w') as f:
            torch.save(model, f)

    @classmethod
    def load(self, fname="data/model_state_dict.pkl"):
        with open(fname, 'r') as f:
            model = torch.load(f)

        new_model = BasicNet()
        new_model.load_state_dict(model)
        return new_model

env = gym.make('CartPole-v1')
net = BasicNet()
if GPU:
    net = net.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
time_step = 0

## Test stuff

def np_to_var(arr):
    if GPU:
        return Variable(torch.Tensor(arr).view(1,-1)).cuda()
    else:
        return Variable(torch.Tensor(arr).view(1,-1))

s = env.reset()

## Test Step
def test(render=False):
    reward = 0
    s = env.reset()
    s = np_to_var(s)
    done = False
    while not done:
        if render: env.render()

        q_vals_var = net(s)
        q_vals, act_idx = q_vals_var.max(1)
        act_idx = act_idx.data[0,0]

        s2, r, done, _ = env.step(act_idx)
        s = np_to_var(s2)

        reward += r

    return reward


## One Step
def _train_step(env, time_step,
                reward=0.0, s=None, render=False, debug=False):
    if not s:
        s = env.reset()
        s = np_to_var(s)

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
    r = np_to_var([r])
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
        if r > 499.0: # end game reached
            r = np_to_var([10])
        else:
            r = np_to_var([-reward*0.5-10]) # Hardcode the loss here
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
    s, r = _train_step(env, time_step, debug=True)
    while s is not None:
        time_step += 1
        s, r = _train_step(env, time_step, reward=r, s=s, debug=True)

    print(test())
    return time_step

## Randomized Agent
# Plot the scores the agent gets
pts = np.zeros(NUM_EPOCHS // 500)
s = None
for i in xrange(NUM_EPOCHS):
    s, reward = _train_step(env, time_step, s=s)
    while s is not None:
        time_step += 1
        s, reward = _train_step(env, time_step, reward=reward, s=s)

    if i % 500 == 0:
        pts[i//500] = test(render=False)
        print(pts[i//500])

net.save()

plt.plot(np.arange(len(pts)), pts, label='Scores', color='red')
ax = plt.axes()
ax.set_xlabel('Epoch (10^2)')
ax.set_ylabel('Score')
ax.set_title('Scores at Various Epoches')
plt.legend(loc='upper left')
plt.savefig('plot.png', dpi=400, bbox_inches='tight')
