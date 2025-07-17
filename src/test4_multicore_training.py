import os
import time
import multiprocessing
import torch
from torch.utils.tensorboard import SummaryWriter
from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *

main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)
N_MULTIPROCESS = 1

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

    start_sampling = time.time()
    state = env.reset()
    done = False
    transitions = []
    total_reward = 0
    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        transitions.append((state, action, reward, next_state, done, log_prob.item()))
        total_reward += reward
        state = next_state
    finish_sampling = time.time()
    sampling_time = finish_sampling - start_sampling

    for s, a, r, ns, d, lp in transitions:
        agent.store_transition((s, a, r, ns, d, lp))

    start_update = time.time()
    agent.update()
    learn_time = time.time() - start_update

    return core_index, sampling_time, learn_time, total_reward, agent.policy.state_dict()

def worker_wrapper(args):
    return simulation_worker(*args)

def average_state_dicts(state_dicts):
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    return avg_state_dict

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    total_episodes = N_EPISODES
    episode_counter = 0

    total_sampling_time = 0.0
    total_learning_time = 0.0

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
        model_state_dict = model.policy.state_dict()
        tasks = [(i, model_state_dict) for i in range(batch_workers)]

        results = pool.map(worker_wrapper, tasks)

        updated_dicts = []
        for core_index, sampling_time, learn_time, reward, updated_state_dict in results:
            episode_counter += 1
            total_sampling_time += sampling_time
            total_learning_time += learn_time
            updated_dicts.append(updated_state_dict)

            main_writer.add_scalar(f"reward_core_{core_index+1}", reward, episode_counter)
            main_writer.add_scalar("reward_average", reward, episode_counter)
            print(f"[Worker {core_index}] Episode {episode_counter}: "
                  f"Sampling Time {sampling_time:.6f}s, Learn Time {learn_time:.6f}s, Reward {reward:.2f}")

        new_state_dict = average_state_dicts(updated_dicts)
        model.policy.load_state_dict(new_state_dict)

    total_time = (time.time() - start_time) / 60
    total_sampling_time = total_sampling_time / N_MULTIPROCESS
    total_learning_time = total_learning_time / N_MULTIPROCESS

    print(f"\n[Experiment Summary] "
          f"Total Sampling Time: {total_sampling_time:.6f}s | "
          f"Total Learning Time: {total_learning_time:.6f}s | "
          f"Total Time: {total_time:.6f}min\n")

    pool.close()
    pool.join()
