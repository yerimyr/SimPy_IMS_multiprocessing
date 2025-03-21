import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from config_RL import *

class ActorCritic(nn.Module):
    """
    Actor-Critic model for PPO with MultiDiscrete action space.

    Args:
        state_dim: Dimension of the state space.
        action_dims: List containing the number of discrete actions per action dimension.
        hidden_size: Number of neurons in hidden layers.
    """
    def __init__(self, state_dim, action_dims, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.action_dims = action_dims

        # Policy Network (Actor)
        self.actor_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.action_heads = nn.ModuleList([nn.Linear(hidden_size, dim) for dim in action_dims])  # MultiDiscrete

        # Value Network (Critic)
        self.critic_fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the Actor-Critic network.

        Args:
            state: Current state of the environment.

        Returns:
            action_probs: Probability distributions for MultiDiscrete action dimensions.
            value: Estimated state value.
        """
        actor_features = self.actor_fc(state)
        action_probs = [torch.softmax(head(actor_features), dim=-1) for head in self.action_heads]  # MultiDiscrete
        value = self.critic_fc(state)
        return action_probs, value

class PPOAgent:
    """
    PPO Agent with MultiDiscrete action space handling.

    This class implements the Proximal Policy Optimization (PPO) algorithm 
    for environments with MultiDiscrete action spaces. The agent consists 
    of an Actor-Critic model and uses the Generalized Advantage Estimation (GAE)
    method for efficient policy updates.

    Args:
        state_dim: Dimension of the state space.
        action_dims: Number of discrete actions for each action dimension.
        lr: Learning rate for the optimizer.
        gamma: Discount factor for future rewards.
        clip_epsilon: Clipping range for PPO.
        update_steps: Number of training epochs per update.
    """
    def __init__(self, state_dim, action_dims, lr=LEARNING_RATE, gamma=GAMMA, clip_epsilon=CLIP_EPSILON, update_steps=UPDATE_STEPS):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps
        self.device = DEVICE
    
        self.policy = ActorCritic(state_dim, action_dims).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        
    def select_action(self, state):
        """
        Selects an action for MultiDiscrete environments.

        Args:
            state: Current state of the environment.

        Returns:
            actions: Selected actions for each action dimension.
            log_prob: Summed log probability of the selected actions because of multidiscrete environment.
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs, _ = self.policy(state)
        
        actions = []
        log_probs = []
        for dist in action_probs:
            categorical_dist = Categorical(dist)
            action = categorical_dist.sample()
            actions.append(action.item())
            log_probs.append(categorical_dist.log_prob(action))
        
        return np.array(actions), torch.sum(torch.stack(log_probs)) 
    
    def store_transition(self, transition):
        """
        Stores a transition in memory.
        
        Args:
        transition: A tuple containing:
            - state: The current state.
            - action: The action taken.
            - reward: The reward received after taking the action.
            - next_state: The state after taking the action.
            - done: Whether the episode has ended.
            - log_prob: The log probability of the selected action.
        """
        self.memory.append(transition)

    def update(self):
        """
        Performs PPO update using stored experience.

        This function processes stored transitions, computes advantages,
        and updates the policy and value networks using PPO loss.
        """
        if not self.memory:
            print("Memory is empty.")
            return

        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)

        _, values = self.policy(states)
        _, next_values = self.policy(next_states)
        next_values[dones == 1] = 0

        advantages = self._compute_gae(rewards, values.squeeze(), self.gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        value_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)

        batch_size = BATCH_SIZE  
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)

        for _ in range(self.update_steps):
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_value_target = value_target[batch_indices]

                action_probs, values_new = self.policy(batch_states)
                
                log_probs_new = []
                for j, dist in enumerate(action_probs):
                    categorical_dist = Categorical(dist)
                    log_probs_new.append(categorical_dist.log_prob(batch_actions[:, j]))
                log_probs_new = torch.sum(torch.stack(log_probs_new), dim=0)
                
                ratio = torch.exp(log_probs_new - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values_new.squeeze(), batch_value_target.detach())
                
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward(retain_graph=True)
                self.optimizer.step()
        
        self.clip_epsilon = max(0.1, self.clip_epsilon * 0.995)
        self.memory.clear()
    
    def _compute_gae(self, rewards, values, gamma, lambda_=0.95):
        """
        Computes Generalized Advantage Estimation (GAE) for PPO.

        Args:
            rewards: Rewards obtained from environment.
            values: Estimated values of the states.
            gamma: Discount factor.
            lambda_: Smoothing factor for GAE.

        Returns:
            torch.Tensor: Computed advantage estimates.
        """
        advantages = torch.zeros_like(rewards, device=self.device)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i] - values[i - 1] if i > 0 else 0
            gae = delta + gamma * lambda_ * gae
            advantages[i] = gae
        return advantages
