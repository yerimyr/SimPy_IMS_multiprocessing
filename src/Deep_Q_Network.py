import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) 모델
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)  # Q-value 출력
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience Replay Buffer (Single-Agent)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        새로운 경험을 버퍼에 저장
        """
        self.buffer.append((state, action, reward, next_state, done))
          

    def sample(self, batch_size):
        """
        배치 크기만큼 랜덤 샘플링하여 학습 데이터 반환
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),  # 단일 행동 유지
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)
    
GLOBAL_REPLAY_BUFFER = ReplayBuffer(capacity=100000)

class DQNAgent:
    """
    Single-Agent Deep Q-Network (DQN) 학습을 위한 에이전트
    """
    def __init__(self, state_dim, action_dim, buffer_size, lr=1e-3, gamma=0.99, target_update_interval=1000, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.update_step = 0
        self.loss = None

        # 경험 리플레이 버퍼 생성
        self.buffer = GLOBAL_REPLAY_BUFFER
        
        # Q-network 및 Target-network 생성
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 타겟 네트워크는 업데이트될 때만 변경됨

        # 옵티마이저 설정
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state, epsilon = 0.1):
        """
        ε-greedy 정책을 사용하여 action 선택
        """
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0)
        if random.random() < epsilon:  
            return np.random.randint(0, self.action_dim)  # 탐색 (Exploration)
        else: 
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values).item()  # 단일 행동 값 반환
            return action

    def update(self, batch_size):
        """
        Q-network 업데이트 (경험 재생)
        """
        if len(self.buffer) < batch_size:
            return  # 충분한 경험이 쌓이지 않으면 학습하지 않음

        # 경험 리플레이에서 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 현재 상태의 Q값 계산
        q_values = self.q_network(states).gather(1, actions)

        # 다음 상태의 Q값 계산 (타겟 네트워크 사용)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # 손실 계산 및 네트워크 업데이트
        loss = nn.MSELoss()(q_values, target_q_values)           
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss = loss.item()

        # 주기적으로 타겟 네트워크 업데이트
        if self.update_step % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
