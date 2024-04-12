import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Softmax)

    def forward(self, obs, deterministic=False):
        action_probs = self.net(obs)
        if deterministic:
            reduce = False
            if len(action_probs.shape) == 1:
                action_probs = action_probs.unsqueeze(0)
                reduce = True
            pi_action = torch.argmax(action_probs, dim=1)
            if reduce:
                pi_action = pi_action.item()
            # print("Deterministic action: ", pi_action.shape)
        else:
            reduce = False
            if len(action_probs.shape) == 1:
                action_probs = action_probs.unsqueeze(0)
                reduce = True
            pi_action = torch.multinomial(action_probs, 1)
            if reduce:
                pi_action = pi_action.item()
            # print("Stochastic action: ", pi_action.shape)
        
        eps = torch.where(action_probs == 0.0, torch.tensor(1e-8), torch.tensor(0.0))
        log_action_probs = torch.log(action_probs + eps)
        return pi_action, action_probs, log_action_probs


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Softmax)

    def forward(self, obs, act):
        # print("obs: ", obs.shape)
        # print("act: ", act.shape)
        all_action_values = self.q(obs)
        # print("all_action_values: ", all_action_values.shape)
        chosen_action_values = torch.gather(all_action_values, 1, act.long())
        print("chosen_action_values: ", chosen_action_values.grad_fn)
        # print("chosen_action_values: ", chosen_action_values.shape)
        chosen_action_values = torch.squeeze(chosen_action_values, -1) 
        # print("chosen_action_values: ", chosen_action_values.shape)
        return chosen_action_values, all_action_values

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.n

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _, _ = self.pi(obs, deterministic)
            return a