import GymWrapper as gw
import os
import time
import multiprocessing
import torch
from GymWrapper import GymInterface 
from PPO import PPOAgent
from config_RL import *
from torch.utils.tensorboard import SummaryWriter

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

N_MULTIPROCESS = 5

def build_model(env):
    """
    Build a PPO model using environment info.

    Args:
        env: Environment to get state and action dimensions.

    Returns:
        PPOAgent: Initialized PPO agent with given dimensions.
    """
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
    return model

def simulation_worker(core_index, model_state_dict):
    """
    Simulates one episode using a local copy of the PPO model.

    Args:
        core_index (int): Index of the current process.
        model_state_dict (dict): State dict from the main PPO model.

    Returns:
        tuple: (core_index, sim_time, finish_time, transitions, episode_reward)
    """
    env = GymInterface()
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)

    start_sim_time = time.time()
    state = env.reset()
    done = False
    episode_transitions = []
    episode_reward = 0
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state
    finish_sim_time = time.time()
    sim_time = finish_sim_time - start_sim_time
    return (core_index, sim_time, finish_sim_time, episode_transitions, episode_reward)

def process_transitions(transitions):
    """
    Processes raw transition tuples into structured component lists.

    Args:
        transitions (list): List of episode transition tuples.

    Returns:
        tuple: (states, actions, rewards, next_states, dones, log_probs)
    """
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    for worker_transitions in transitions:
        for tr in worker_transitions:
            states.append(tr[0])
            actions.append(tr[1])
            rewards.append(tr[2])
            next_states.append(tr[3])
            dones.append(tr[4])
            log_probs.append(tr[5])
    return states, actions, rewards, next_states, dones, log_probs

def worker_wrapper(args):
    return simulation_worker(*args)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    all_experiment_results = []
    
    for experiment_idx in range(3):
        print(f"========== Experiment {experiment_idx+1} ==========")
        total_episodes = N_EPISODES
        episode_counter = 0
        episode_sim_times = []
        episode_transmit_times = []
        episode_gpu_update_times = []

        env_main = GymInterface()
        if LOAD_MODEL:
            state_dim = len(env_main.reset())
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
            model = build_model(env_main)

        start_time = time.time()

        while episode_counter < total_episodes:
            batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
            model_state_dict = model.policy.state_dict()
            tasks = [(i, model_state_dict) for i in range(batch_workers)]

            sim_times = []
            transmit_times = []

            for result in pool.imap_unordered(worker_wrapper, tasks):
                core_index, sim_time, finish_sim_time, transitions, episode_reward = result

                receive_time = time.time()
                transmit_time = receive_time - finish_sim_time

                sim_times.append(sim_time)
                transmit_times.append(transmit_time)

                states, actions, rewards, next_states, dones, log_probs = process_transitions([transitions])
                for j in range(len(states)):
                    model.store_transition((states[j], actions[j], rewards[j], next_states[j], dones[j], log_probs[j]))

                episode_counter += 1
                main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, global_step=episode_counter)
                main_writer.add_scalar("reward_average", episode_reward, global_step=episode_counter)

            avg_sim_time = sum(sim_times) / len(sim_times)
            avg_transmit_time = sum(transmit_times) / len(transmit_times)

            update_start = time.time()
            model.update()
            update_end = time.time()
            gpu_update_time = update_end - update_start

            episode_sim_times.append(avg_sim_time)
            episode_transmit_times.append(avg_transmit_time)
            episode_gpu_update_times.append(gpu_update_time)

            print(f"Episode {episode_counter}: Sim {avg_sim_time:.4f}s, Transmit {avg_transmit_time:.4f}s, GPU {gpu_update_time:.4f}s")

        final_avg_sim_time = sum(episode_sim_times) / len(episode_sim_times)
        final_avg_transmit_time = sum(episode_transmit_times) / len(episode_transmit_times)
        final_avg_gpu_update_time = sum(episode_gpu_update_times) / len(episode_gpu_update_times)

        end_time = time.time()
        computation_time = (end_time - start_time) / 60
    
        all_experiment_results.append({
            'Sim': final_avg_sim_time,
            'Transmit': final_avg_transmit_time,
            'GPU': final_avg_gpu_update_time,
            'Total_Minutes': computation_time
        })
    
    pool.close()
    pool.join()
    
    print("\n====== Experiment Summary 결과 (3회) ======")
    for i, result in enumerate(all_experiment_results):
        print(f"[Experiment {i+1}] Sim: {result['Sim']:.4f}s | Transmit: {result['Transmit']:.4f}s | GPU: {result['GPU']:.4f}s | Total: {result['Total_Minutes']:.2f}min")