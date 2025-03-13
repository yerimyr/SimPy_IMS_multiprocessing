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
    def __init__(self, state_dim, action_dim, lr=LEARNING_RATE, gamma=GAMMA, clip_epsilon=0.2, update_steps=5):
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
        Performs the PPO policy update using stored experiences.
        """
        if len(self.memory) == 0:
            return
        
        # Unpack stored experiences and convert device
        states, actions, rewards, next_states, dones, log_probs_old = zip(*self.memory) 
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32).to(self.device)
        
        # Compute state values
        _, values = self.policy(states)  # Value of current states
        _, next_values = self.policy(next_states)  # Value of next states
        
        # Compute advantages using the Bellman equation
        advantages = (rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()).detach()
        
        # PPO policy update
        for _ in range(self.update_steps):
            action_probs, values_new = self.policy(states)
            dist = Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)
            
            # Compute the ratio of new vs old policy probabilities
            ratio = torch.exp(log_probs_new - log_probs_old)
            
            surrogate1 = ratio * advantages  # Standard policy gradient loss
            surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages  # Clipped loss
            policy_loss = -torch.min(surrogate1, surrogate2).mean()  # Final policy loss
            
            # Compute value function loss (Critic loss)
            value_loss = nn.MSELoss()(values_new.squeeze(), rewards.squeeze())
            
            # Update the policy
            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward(retain_graph=True)  # Compute gradients
            self.optimizer.step()  # Apply gradients
        
        self.memory = []  # Clear memory after update
