import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import MultivariateNormal

from networks.ppo_net import CentralizedPPOActor, CentralizedPPOCritic
from policy.base_policy import BasePolicy
from utils.config_utils import ConfigObjectFactory
from utils.train_utils import weight_init


class CentralizedPPO(BasePolicy):

    def __init__(self, env_info: dict):

        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = env_info['n_agents']
        self.action_dim = env_info['action_dim']

        self.rnn_hidden_dim = 64
        actor_input_shape = env_info['obs_space'] * self.n_agents
        self.ppo_actor = CentralizedPPOActor(actor_input_shape, self.action_dim, self.n_agents, self.rnn_hidden_dim)
        self.ppo_critic = CentralizedPPOCritic(env_info['state_space'])
        self.optimizer_actor = torch.optim.Adam(params=self.ppo_actor.parameters(),
                                                lr=self.train_config.lr_actor)
        self.optimizer_critic = torch.optim.Adam(params=self.ppo_critic.parameters(),
                                                 lr=self.train_config.lr_critic)

        self.model_path = os.path.join(self.train_config.model_dir, self.env_config.learn_policy)
        self.result_path = os.path.join(self.train_config.result_dir, self.env_config.learn_policy)
        self.init_path(self.model_path, self.result_path)
        self.ppo_actor_path = os.path.join(self.model_path, "ppo_actor.pth")
        self.ppo_critic_path = os.path.join(self.model_path, "ppo_critic.pth")


        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.ppo_actor.to(self.device)
        self.ppo_critic.to(self.device)


        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.01)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        self.init_wight()

    def init_wight(self):
        self.ppo_actor.apply(weight_init)
        self.ppo_critic.apply(weight_init)

    def init_hidden(self, batch_size):
        self.rnn_hidden = torch.zeros((batch_size, self.rnn_hidden_dim)).to(self.device)

    def learn(self, batch_data: dict, episode_num: int):

        obs = batch_data['obs'].to(self.device).detach()
        state = batch_data['state'].to(self.device)
        actions = batch_data['actions'].to(self.device)
        log_probs = batch_data['log_probs'].to(self.device)
        batch_size = sum(batch_data['per_episode_len'])
        rewards = batch_data['rewards']
        obs = obs.reshape(batch_size, -1)
        discount_reward = self.get_discount_reward(rewards).to(self.device)
        self.init_hidden(batch_size)

        with torch.no_grad():
            state_value = self.ppo_critic(state)
            advantage_function = discount_reward - state_value
  
            advantage_function = ((advantage_function - advantage_function.mean()) / (
                    advantage_function.std() + 1e-10)).unsqueeze(dim=-1)

        for i in range(self.train_config.learn_num):

  
            action_means, self.rnn_hidden = self.ppo_actor(obs, self.rnn_hidden.detach())
            dist = MultivariateNormal(action_means, self.cov_mat)
            curr_log_probs = dist.log_prob(actions)

            ratios = torch.exp(curr_log_probs - log_probs)
            surr1 = ratios * advantage_function
            surr2 = torch.clamp(ratios, 1 - self.train_config.ppo_loss_clip,
                                1 + self.train_config.ppo_loss_clip) * advantage_function
            actor_loss = (-torch.min(surr1, surr2)).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()


            curr_state_value = self.ppo_critic(state)
            critic_loss = nn.MSELoss()(curr_state_value, discount_reward)
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

    def get_cov_mat(self) -> Tensor:
        return self.cov_mat

    def get_discount_reward(self, batch_reward: list) -> Tensor:
        discount_rewards = []
        for reward in reversed(batch_reward):
            discounted_reward = 0
            for one_reward in reversed(reward):
                discounted_reward = one_reward + discounted_reward * self.train_config.gamma
                discount_rewards.insert(0, discounted_reward)
        return torch.Tensor(discount_rewards)

    def save_model(self):
        torch.save(self.ppo_actor.state_dict(), self.ppo_actor_path)
        torch.save(self.ppo_critic.state_dict(), self.ppo_critic_path)

    def load_model(self):
        self.ppo_actor.load_state_dict(torch.load(self.ppo_actor_path))
        self.ppo_critic.load_state_dict(torch.load(self.ppo_critic_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.ppo_actor_path) and os.path.exists(self.ppo_critic_path)
