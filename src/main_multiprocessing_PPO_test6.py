import os
import time
import multiprocessing

import torch
from torch.utils.tensorboard import SummaryWriter

from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
N_MULTIPROCESS = 5

def build_model(env):
    """
    Build a PPO model using environment info.
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
    """
    env = GymInterface()
    agent = build_model(env)
    agent.policy.load_state_dict(model_state_dict)
    # Move the inference model to CPU
    agent.policy.to('cpu')  ########## 추론용모델(cpu) 구현 부분 ##########
    agent.device = torch.device('cpu') 
    
    start_sampling = time.time()
    state = env.reset()
    done = False
    episode_transitions = []
    episode_reward = 0
    while not done:
        action, log_prob = agent.select_action(state)  ########## 추론용모델(gpu) 구현 부분 ##########
        next_state, reward, done, _ = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state
    finish_sampling = time.time()
    sampling = finish_sampling - start_sampling

    return core_index, sampling, finish_sampling, episode_transitions, episode_reward

def process_transitions(transitions):
    """
    Processes raw transition tuples into structured component lists.
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

    total_episodes = N_EPISODES
    episode_counter = 0

    # timing records
    episode_copy_times = []
    episode_sampling_times = []
    episode_transmit_times = []
    episode_learning_times = []

    # build or load main model on GPU
    env_main = GymInterface()
    if LOAD_MODEL:
        model = build_model(env_main)
        model.policy.load_state_dict(
            torch.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME))
        )
        print(f"{LOAD_MODEL_NAME} loaded successfully")
    else:
        model = build_model(env_main)

    start_time = time.time()

    while episode_counter < total_episodes:
        batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
        # measure parameter copy time
        start_copy = time.time()
        model_state_dict = model.policy.state_dict()
        copy_time = time.time() - start_copy
        episode_copy_times.append(copy_time)

        tasks = [(i, model_state_dict) for i in range(batch_workers)]

        sampling_times = []
        transmit_times = []

        # integrated buffer: gather all worker results synchronously
        results = pool.map(worker_wrapper, tasks)  ########## integrated buffer 구현 부분 ##########

        all_transitions = []
        for core_index, sampling, finish_sampling, transitions, episode_reward in results:
            receive_time = time.time()
            transfer = receive_time - finish_sampling

            episode_sampling_times.append(sampling)
            episode_transmit_times.append(transfer)

            sampling_times.append(sampling)
            transmit_times.append(transfer)
            all_transitions.extend(transitions)

            episode_counter += 1
            main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter)
            main_writer.add_scalar("reward_average", episode_reward, episode_counter)

        # store all transitions at once
        states, actions, rewards, next_states, dones, log_probs = process_transitions([all_transitions])
        for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, log_probs):
            model.store_transition((s, a, r, ns, d, lp))

        avg_sampling = sum(sampling_times) / len(sampling_times)
        avg_transfer = sum(transmit_times) / len(transmit_times)

        # learning update on GPU
        start_learning = time.time()
        model.update()
        learning = time.time() - start_learning
        episode_learning_times.append(learning)

        print(
            f"Episode {episode_counter}: Copy {copy_time:.3f}s, Sampling {avg_sampling:.3f}s, "
            f"Transfer {avg_transfer:.3f}s, Learning {learning:.3f}s"
        )

    # experiment summary
    total_time = (time.time() - start_time) / 60
    final_avg_copy = sum(episode_copy_times) / len(episode_copy_times)
    final_avg_sampling = sum(episode_sampling_times) / len(episode_sampling_times)
    final_avg_transfer = sum(episode_transmit_times) / len(episode_transmit_times)
    final_avg_learning = sum(episode_learning_times) / len(episode_learning_times)

    print(
        f"\n[Experiment Summary] "
        f"Copy {final_avg_copy:.3f}s | Sampling {final_avg_sampling:.3f}s | "
        f"Transfer {final_avg_transfer:.3f}s | Learning {final_avg_learning:.3f}s | "
        f"Total {total_time:.2f}min\n"
    )

    pool.close()
    pool.join()