import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random

from kerbal_rl.envs.hover import HoverV0

#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha      = 0.00001
gamma           = 0.98
batch_size      = 64
buffer_limit    = 50000
tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update


class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc_mu = nn.Linear(128,1)
        self.fc_std  = nn.Linear(128,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

def main():
    max_thrust_ratio = 0.4
    max_altitude = 700
    max_step = 2000
    max_speed = 80


    env = HoverV0(
        max_thrust_ratio=max_thrust_ratio,
        max_altitude=max_altitude,
        max_step=max_step,
        max_speed=max_speed,
        )
    pi = PolicyNet(lr_pi)
    pi.load_state_dict(torch.load('pi_420.pt'))
    pi.eval()

    score = 0.0
    print_interval = 1

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        while not done:
            a, log_prob= pi(torch.from_numpy(s).float())
            s_prime, r, done, info = env.step([a.item()])
            score +=r
            s = s_prime
                
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f} alpha:{:.4f}".format(n_epi, score/print_interval, pi.log_alpha.exp()))
            score = 0.0

    env.close()

if __name__ == '__main__':
    main()