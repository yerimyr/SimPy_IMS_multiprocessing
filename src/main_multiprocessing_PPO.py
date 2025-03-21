import os
import time
import multiprocessing
import math
import torch
from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *

# Number of cores to use for multiprocessing
N_MULTIPROCESS = 5

def build_model(env):
    """
    Create and return a PPOAgent model for the given Gym environment.

    Args:
        env (GymInterface): The Gym environment instance.

    Returns:
        PPOAgent: A PPO agent initialized with proper dimensions and hyperparameters.
    """
    state_dim = len(env.reset())
    action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)]  # MultiDiscrete
    model = PPOAgent(
        state_dim=state_dim,
        action_dims=action_dims,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        clip_epsilon=CLIP_EPSILON,
        update_steps=UPDATE_STEPS
    )
    return model


def simulation_worker(core_id, model_state_dict, return_dict):
    """
    Run a PPO agent episode in a separate process and store collected transitions.

    Args:
        core_id: Unique ID of the worker process.
        model_state_dict: Shared global policy network parameters.
        return_dict: Manager dictionary to store results from each process.

    Returns:
        None: Stores list of transitions in return_dict[core_id].
    """
    env = GymInterface()
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)

    state = env.reset()
    done = False
    transitions = []

    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        transitions.append((state, action, reward, next_state, done, log_prob.item()))
        state = next_state

    return_dict[core_id] = transitions


def process_transitions(transitions):
    """
    Combine and unpack transition data collected from multiple worker processes.

    Args:
        transitions (list): A list of lists where each inner list contains transitions from a worker.

    Returns:
        tuple: Separate lists for states, actions, rewards, next_states, dones, and log_probs.
    """
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    for t in transitions:
        for tr in t:
            states.append(tr[0])
            actions.append(tr[1])
            rewards.append(tr[2])
            next_states.append(tr[3])
            dones.append(tr[4])
            log_probs.append(tr[5])
    return states, actions, rewards, next_states, dones, log_probs


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    for run in range(5):
        print(f"=============== experiment {run+1} ===============")
        env = GymInterface()
        
        if LOAD_MODEL:
            state_dim = len(env.reset())
            action_dims = [len(ACTION_SPACE) for _ in range(MAT_COUNT)] 
            model = PPOAgent(
                state_dim=state_dim,
                action_dims=action_dims,
                lr=LEARNING_RATE,
                gamma=GAMMA,
                clip_epsilon=CLIP_EPSILON,
                update_steps=UPDATE_STEPS
            ) 
            model.policy.load_state_dict(torch.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME)))
            print(f"{LOAD_MODEL_NAME} loaded successfully")
        else:
            model = build_model(env)

        start_time = time.time()
        computation_times = []
        episode_counter = 0
        num_groups = math.ceil(N_EPISODES / N_MULTIPROCESS)

        for group in range(num_groups):
            remaining = N_EPISODES - episode_counter
            current_core = min(N_MULTIPROCESS, remaining)

            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            model_state_dict = model.policy.state_dict()

            processes = []
            for i in range(current_core):
                p = multiprocessing.Process(target=simulation_worker, args=(i, model_state_dict, return_dict))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

            all_transitions = [return_dict[i] for i in range(current_core)]
            states, actions, rewards, next_states, dones, log_probs = process_transitions(all_transitions)

            for i in range(len(states)):
                model.store_transition((states[i], actions[i], rewards[i], next_states[i], dones[i], log_probs[i]))

            model.update()
            episode_counter += current_core
            print(f"Episode group {group+1} complete. PPO model updated with {current_core} episodes.")

        if SAVE_MODEL:
            model_path = os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME)
            torch.save(model.policy.state_dict(), model_path)
            print(f"{SAVED_MODEL_NAME} saved successfully")

        end_time = time.time()
        computation_time = (end_time - start_time) / 60
        computation_times.append(computation_time)
        print(f"experiment {run+1} computation time (m): {computation_time:.2f} minutes")

    print("\n=============== experiment 5회 완료 ================")
    print("각 experiment의 Computation time (분): ")
    for idx, t in enumerate(computation_times, 1):
        print(f"experiment {idx}: {t:.2f} minutes")