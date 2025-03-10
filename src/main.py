import os
import time
import multiprocessing
import torch
from GymWrapper import GymInterface
from Deep_Q_Network import DQNAgent
from config_RL import *

# Number of cores to use for multiprocessing
N_MULTIPROCESS = 1

def build_model(env):
    """
    Build and return a DQNAgent model using the observation and action spaces from the given environment.

    Args:
        env (GymInterface): The Gym environment instance.

    Returns:
        DQNAgent: A new DQNAgent instance.
    """
    model = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        buffer_size=BUFFER_SIZE,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        device=DEVICE
    )
    return model

def simulation_worker(core_id, model_state_dict):
    """
    Worker function executed in each process.

    This function:
      - Creates its own GymInterface environment.
      - Loads the global model parameters (model_state_dict) into the local agent.
      - Runs a simulation episode(SIM_TIME), storing transitions in its local replay buffer.
      - After the episode, samples a batch of transitions (BATCH_SIZE) from the local replay buffer,
        converts the tensors to numpy arrays on CPU, and returns them.

    Args:
        core_id: ID of the current worker.
        model_state_dict: Global model's state dictionary to be loaded into the worker's model.

    Returns:
        sample: A tuple containing sampled transitions (states, actions, rewards, next_states, dones)
                       as numpy arrays if enough samples exist; otherwise, None.
    """
    # Create a GymInterface environment for this worker process
    env_instance = GymInterface()
    # Load the global model parameters into the worker's local agent's Q-network
    env_instance.agent.q_network.load_state_dict(model_state_dict)
    
    state = env_instance.reset()
    done = False
    while not done:
        action = env_instance.agent.select_action(state, epsilon = max(0.1, 1.0 - N_EPISODES/500))
        next_state, reward, done, _ = env_instance.step(action)
        env_instance.agent.buffer.push(state, action, reward, next_state, done)
        state = next_state

    if len(env_instance.agent.buffer) >= BATCH_SIZE:
        sample = env_instance.agent.buffer.sample(BATCH_SIZE)
        sample = tuple(x.cpu().numpy() for x in sample)
    else:
        sample = None

    return sample

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    
    computation_times = []

    for run in range(5):
        print(f"=============== experiment {run+1} ===============")
        # Create a GymInterface environment in the main process to access observation and action spaces.
        env = GymInterface()

        # Create or load the global model.
        if LOAD_MODEL:
            model = DQNAgent(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                buffer_size=BUFFER_SIZE,
                lr=LEARNING_RATE,
                gamma=GAMMA,
                device=DEVICE
            )
            model.q_network.load_state_dict(
                DQNAgent.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME)).q_network.state_dict()
            )
            print(f"{LOAD_MODEL_NAME} loaded successfully")
        else:
            model = build_model(env)
        
        start_time = time.time()

        # Main episode loop:
        # For each episode, run simulations in parallel across multiple processes,
        # collect batches of transitions from each worker, store them in the global replay buffer,
        # and update the global model using these transitions.
        for episode in range(N_EPISODES):
            # Retrieve the current global model parameters to share with all workers.
            model_state_dict = model.q_network.state_dict()

            # Create a multiprocessing pool with N_MULTIPROCESS workers.
            with multiprocessing.Pool(N_MULTIPROCESS) as pool:
                # starmap함수를 통해 각 코어에 대해 simulation_worker함수를 병렬로 실행, 각 코어에 코어 id와 model의 파라미터가 전달됨.
                # 결과적으로 각 코어는 자신의 시뮬레이션 에피소드를 실행한 후, 자체 replay buffer에서 BATCH_SIZE만큼 샘플링한 배치를 numpy 배열 형태로 반환하고 results 리스트에 저장함.
                results = pool.starmap(simulation_worker, [(i, model_state_dict) for i in range(N_MULTIPROCESS)]) 
            
            # Process each worker's returned sample batch.
            for sample in results:
                if sample is not None:
                    batch_size_in_sample = sample[0].shape[0]
                    # For each transition in the batch, push it into the global replay buffer.
                    for i in range(batch_size_in_sample):  # 샘플 배치 내의 각 transition을 하나씩 순회함.
                        state_arr = sample[0][i]
                        action_val = int(sample[1][i][0])  # 단일 원소
                        reward_val = float(sample[2][i][0])  # 단일 원소
                        next_state_arr = sample[3][i]
                        done_val = bool(sample[4][i][0])  # 단일 원소
                        model.buffer.push(state_arr, action_val, reward_val, next_state_arr, done_val)
                    # Update the global model using the global replay buffer with a batch size of BATCH_SIZE.
                    model.update(batch_size=BATCH_SIZE)
            
            print(f"Episode {episode+1}: Multiprocess simulation complete and model updated {N_MULTIPROCESS} times.")
        
        if SAVE_MODEL:
            model_path = os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME)
            torch.save(model.q_network.state_dict(), model_path)
            print(f"{SAVED_MODEL_NAME} is saved successfully")
        
        end_time = time.time()
        computation_time = (end_time - start_time) / 60
        computation_times.append(computation_time)
        print(f"experiment {run+1} computation time (m): {computation_time:.2f} minutes")
        
    print("\n=============== experiment 5회 완료 ================")
    print("각 experiment의 Computation time (분): ")
    for idx, t in enumerate(computation_times, 1):
        print(f"experiment {idx}: {t:.2f} minutes")
