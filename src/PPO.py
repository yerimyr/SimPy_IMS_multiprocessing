import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from config_RL import *

class ActorCritic(nn.Module):
    """
    Actor-Critic model for Proximal Policy Optimization (PPO).
    
    Attributes:
        actor (nn.Module): Neural network for the policy (actor).
        critic (nn.Module): Neural network for the value function (critic).
    """
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Policy Network (Actor)
        self.actor_fc1 = nn.Linear(state_dim, 64)
        self.actor_fc2 = nn.Linear(64, 64)
        self.actor_out = nn.Linear(64, action_dim)
        
        # Value Network (Critic)
        self.critic_fc1 = nn.Linear(state_dim, 64)
        self.critic_fc2 = nn.Linear(64, 64)
        self.critic_out = nn.Linear(64, 1)
        
    def forward(self, state):
        # Actor
        x = torch.relu(self.actor_fc1(state))
        x = torch.relu(self.actor_fc2(x))
        action_probs = torch.softmax(self.actor_out(x), dim=-1)  # Convert to probability distribution
        
        action_probs = action_probs.view(-1)
        
        # Critic
        v = torch.relu(self.critic_fc1(state))
        v = torch.relu(self.critic_fc2(v))
        value = self.critic_out(v)  # Estimate state value
        
        return action_probs, value

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    
    Attributes:
        policy: The Actor-Critic network.
        optimizer: Optimizer for updating the policy.
        memory: Storage for on-policy experiences.
        gamma: Discount factor for future rewards.
        clip_epsilon: PPO clipping parameter.
        update_steps: Number of optimization steps per update.
        device: Device to run the model (CPU/GPU).
    """
    def __init__(self, state_dim, action_dim, lr=LEARNING_RATE, gamma=GAMMA, clip_epsilon=0.5, update_steps=5):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_steps = update_steps
        self.device = DEVICE
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
    
    def select_action(self, state):
        """
        Selects an action based on the current policy.
        
        Args:
            state: The current environment state.
        
        Returns:
            action: Chosen action.
            log_prob: Log probability of the chosen action.
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_probs, _ = self.policy(state)  # Get action probabilities from policy
        dist = Categorical(action_probs)  # Create a categorical distribution
        action = dist.sample()  # Sample an action
           
        return action.item(), dist.log_prob(action)  # Return action and log probability
    
    def store_transition(self, transition):
        """
        Stores a transition (experience) in memory.
        
        Args:
            transition (tuple): (state, action, reward, next_state, done, log_prob)
        """
        self.memory.append(transition)  # Save the transition for future updates

    def update(self):
        """
        Performs the PPO policy update using stored experiences at the end of an episode.
        """
        if len(self.memory) == 0:  # 한 에피소드가 끝난 후에만 학습
            return

        # Unpack stored experiences and convert to tensors
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)

        # Compute state values
        _, values = self.policy(states)
        _, next_values = self.policy(next_states)
        
        if dones[-1] == 1:
            next_values[-1] = 0

        # Compute Generalized Advantage Estimation (GAE)
        def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
            advantages = torch.zeros_like(rewards, device=self.device)
            gae = 0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + gamma * next_values[i] * (1 - dones[i]) - values[i]
                gae = delta + gamma * lambda_ * gae
                advantages[i] = gae
            return advantages

        advantages = compute_gae(rewards, values.squeeze(), self.gamma, lambda_=0.95)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute target values for critic
        value_target = rewards + self.gamma * next_values.squeeze() * (1 - dones)

        # Mini-batch training
        batch_size = BATCH_SIZE  
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices) 

        # PPO policy update with mini-batches
        for _ in range(self.update_steps):
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_value_target = value_target[batch_indices]

                # Forward pass
                action_probs, values_new = self.policy(batch_states)
                dist = Categorical(action_probs)
                log_probs_new = dist.log_prob(batch_actions)

                # Compute the ratio of new vs old policy probabilities
                ratio = torch.exp(log_probs_new - batch_log_probs_old)

                # Compute PPO loss
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # Compute value function loss (Critic loss)
                value_loss = nn.MSELoss()(values_new.squeeze(), batch_value_target.detach())

                # Update the policy
                self.optimizer.zero_grad()
                (policy_loss + value_loss).backward(retain_graph=True)
                self.optimizer.step()

        self.clip_epsilon = max(0.1, self.clip_epsilon * 0.995)
        
        # Clear memory after update
        self.memory.clear()


