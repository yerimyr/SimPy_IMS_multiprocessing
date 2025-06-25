import os
import time
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
N_MULTIPROCESS = 2

def build_model(env):
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
        next_state, reward, done, _ = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state
    finish_sim_time = time.time()
    sim_time = finish_sim_time - start_sim_time

    return core_index, sim_time, finish_sim_time, episode_transitions, episode_reward

def process_transitions(transitions):
    states, actions, rewards, next_states, dones, log_probs = [], [], [], [], [], []
    for worker_transitions in transitions:
        for (s, a, r, ns, d, lp) in worker_transitions:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
            log_probs.append(lp)
    return states, actions, rewards, next_states, dones, log_probs

def worker_wrapper(args):
    return simulation_worker(*args)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    total_episodes = N_EPISODES
    episode_counter = 0

    env_main = GymInterface()
    model = build_model(env_main)

    start_time = time.time()

    while episode_counter < total_episodes:
        batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)

        start_copy = time.time()
        model_state_dict = model.policy.state_dict()
        param_copy = time.time() - start_copy

        tasks = [(i, model_state_dict) for i in range(batch_workers)]

        worker_results = []
        
        overhead_start = time.time()
        for result in pool.imap_unordered(worker_wrapper, tasks):
            worker_results.append(result)
        overhead_end = time.time()
        overhead_time = overhead_end - overhead_start
        
        sorted_results = sorted(worker_results, key=lambda x: x[2])
        learn_end_times = [0] * len(sorted_results)

        previous_learn_end = None

        for i, result in enumerate(sorted_results):
            core_index, sampling, finish_sim_time, transitions, episode_reward = result
            receive_time = time.time()

            waiting1 = 0.0 if previous_learn_end is None else max(0, previous_learn_end - finish_sim_time)
            transfer_start = time.time()

            states, actions, rewards, next_states, dones, log_probs = process_transitions([transitions])
            for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, log_probs):
                model.store_transition((s, a, r, ns, d, lp))

            transfer_time = time.time() - transfer_start

            learn_start = time.time()
            model.update()
            learn_end = time.time()

            learning = learn_end - learn_start
            previous_learn_end = learn_end

            learn_end_times[i] = learn_end

            episode_counter += 1

            main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter)
            main_writer.add_scalar("reward_average", episode_reward, episode_counter)

            print(f"Worker {core_index} done — episode {episode_counter}: "
                  f"Copy {param_copy:.6f}s, Sampling {sampling:.6f}s, "
                  f"Waiting1 {waiting1:.6f}s, Transfer {transfer_time:.6f}s, "
                  f"Learning {learning:.6f}s,  overhead {overhead_time:.6f}s")

        latest_learn_end_time = max(learn_end_times)

        for i, result in enumerate(sorted_results):
            core_index = result[0]
            waiting2 = max(0, latest_learn_end_time - learn_end_times[i])
            print(f"Worker {core_index} Waiting2: {waiting2:.6f}s")

    total_time = (time.time() - start_time)
    print(f"\n[Experiment Summary] Total Time: {total_time:.6f}s\n")

    pool.close()
    pool.join()
