import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from agents.Base_Agent import Base_Agent
from utilities.data_structures.Replay_Buffer import Replay_Buffer
from agents.actor_critic_agents.SAC import SAC
from utilities.Utility_Functions import create_actor_distribution
import datetime
import os

class SAC_Discrete(SAC):
    """The Soft Actor Critic for discrete actions. It inherits from SAC for continuous actions and only changes a few
    methods."""
    agent_name = "SAC"
    def __init__(self, config):
        Base_Agent.__init__(self, config)
        assert self.action_types == "DISCRETE", "Action types must be discrete. Use SAC instead for continuous actions"
        assert self.config.hyperparameters["Actor"]["final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.hyperparameters = config.hyperparameters
        self.shuffle_channels = False
        if self.config.use_NN:
            self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
            self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic", override_seed=self.config.seed + 1)
        else:
            self.critic_local = self.create_CNN(input_dim=self.state_size, output_dim=["linear", self.action_size], key_to_use="Critic")
            self.critic_local_2 = self.create_CNN(input_dim=self.state_size, output_dim=["linear", self.action_size],
                                            key_to_use="Critic", override_seed=self.config.seed + 1)
            self.shuffle_channels = True
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        if self.config.use_NN:
            self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
            self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                                key_to_use="Critic")
        else:
            self.critic_target = self.create_CNN(input_dim=self.state_size, output_dim=["linear", self.action_size],
                                            key_to_use="Critic")
            self.critic_target_2 = self.create_CNN(input_dim=self.state_size, output_dim=["linear", self.action_size],
                                                key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"], self.hyperparameters["batch_size"],
                                    self.config.seed, device=self.device)
        if self.config.use_NN:
            self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        else:
            self.actor_local = self.create_CNN(input_dim=self.state_size, output_dim=["linear", self.action_size], key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                          lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        self.automatic_entropy_tuning = self.hyperparameters["automatically_tune_entropy_hyperparameter"]
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)
        else:
            self.alpha = self.hyperparameters["entropy_term_weight"]
        assert not self.hyperparameters["add_extra_noise"], "There is no add extra noise option for the discrete version of SAC at moment"
        self.add_extra_noise = False
        self.do_evaluation_iterations = self.hyperparameters["do_evaluation_iterations"]

        if self.config.load_path:
            print("Loading in saved policy")
            self.locally_load_policy(self.config.load_path)

        try: 
            self.max_episode_steps = self.environment.max_episode_steps
        except:
            try: 
                self.max_episode_steps = self.environment._max_episode_steps
            except:
                self.max_episode_steps = self.environment.unwrapped.max_episode_steps

    def produce_action_and_action_info(self, state):
        """Given the state, produces an action, the probability of the action, the log probability of the action, and
        the argmax action"""
        # print("oringal state shape", state.shape)
        # if self.shuffle_channels:
        #     if len(state.shape) == 3:
        #         state = torch.unsqueeze(state, 0)
        #     state = torch.permute(state, [0, 3, 1, 2])
        #     # print("state", state)
        #     # print("state.shape", state.shape)
        #     # assert False
        # print("state shape", state.shape)
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action
    
    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""
        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    
    def locally_save_policy(self):
        savetime = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
        savetime += f"_{self.agent_name}_{self.config.environment_short_name}"
        os.makedirs(f"./models/{savetime}")
        print(f"Saving policy to: ./models/{savetime}")
        torch.save(self.actor_local, f"./models/{savetime}/actor_local.pt")
        torch.save(self.critic_local, f"./models/{savetime}/critic_local.pt")
        torch.save(self.critic_local_2, f"./models/{savetime}/critic_local_2.pt")
        torch.save(self.critic_target, f"./models/{savetime}/critic_target.pt")
        torch.save(self.critic_target_2, f"./models/{savetime}/critic_target_2.pt")

    def locally_load_policy(self, path):
        self.actor_local = torch.load(os.path.join(path, "actor_local.pt"))
        self.critic_local = torch.load(os.path.join(path, "critic_local.pt"))
        self.critic_local_2 = torch.load(os.path.join(path, "critic_local_2.pt"))
        self.critic_target = torch.load(os.path.join(path, "critic_target.pt"))
        self.critic_target_2 = torch.load(os.path.join(path, "critic_target_2.pt"))
                                         
        # self.actor_local.load_state_dict(torch.load(os.path.join(path, "actor_local.pt")))
        # self.critic_local.load_state_dict(torch.load(os.path.join(path, "critic_local.pt")))
        # self.critic_local_2.load_state_dict(torch.load(os.path.join(path, "critic_local_2.pt")))
        # self.critic_target.load_state_dict(torch.load(os.path.join(path, "critic_target.pt")))
        # self.critic_target_2.load_state_dict(torch.load(os.path.join(path, "critic_target_2.pt")))
