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

# Helper to flatten and store transitions
from GymWrapper import process_transitions  # ensure this is imported or defined above

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

    # Transfer parameters and measure time
    start_transfer = time.time()
    agent.policy.load_state_dict(model_state_dict)
    agent.policy.to('cpu')
    agent.device = torch.device('cpu')
    transfer_time = time.time() - start_transfer
    print(f"[Worker {core_index}] Parameter transfer to CPU: {transfer_time:.6f} sec")

    # Run one episode
    start_sim_time = time.time()
    state = env.reset()
    done = False
    episode_transitions = []
    episode_reward = 0.0

    while not done:
        action, log_prob = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state

    finish_sim_time = time.time()
    sim_time = finish_sim_time - start_sim_time
    return core_index, sim_time, finish_sim_time, episode_transitions, episode_reward, transfer_time

def worker_wrapper(args):
    return simulation_worker(*args)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)
    all_experiment_results = []

    for experiment_idx in range(3):
        print(f"========== Experiment {experiment_idx+1} ==========")

        env_main = GymInterface()
        if LOAD_MODEL:
            model = build_model(env_main)
            model.policy.load_state_dict(torch.load(os.path.join(SAVED_MODEL_PATH, LOAD_MODEL_NAME)))
            print(f"{LOAD_MODEL_NAME} loaded successfully")
        else:
            model = build_model(env_main)

        total_episodes = N_EPISODES
        episode_counter = 0
        episode_sim_times = []
        episode_transmit_times = []
        episode_transfer_times = []
        episode_gpu_update_times = []

        start_time = time.time()

        while episode_counter < total_episodes:
            batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
            model_state_dict = model.policy.state_dict()
            tasks = [(i, model_state_dict) for i in range(batch_workers)]

            # Synchronous gather of worker results (Integrated Buffer)
            results = pool.map(worker_wrapper, tasks)

            sim_times = []
            transmit_times = []
            transfer_times = []
            transitions_list = []

            # Collect and record metrics, accumulate transitions
            for core_index, sim_time, finish_sim_time, transitions, episode_reward, transfer_time in results:
                receive_time = time.time()
                transmit_time = receive_time - finish_sim_time

                sim_times.append(sim_time)
                transmit_times.append(transmit_time)
                transfer_times.append(transfer_time)
                transitions_list.append(transitions)

                main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter+1)
                main_writer.add_scalar("reward_average", episode_reward, episode_counter+1)
                episode_counter += 1

            # Flatten and store all transitions at once
            states, actions, rewards, next_states, dones, log_probs = process_transitions(transitions_list)
            for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, log_probs):
                model.store_transition((s, a, r, ns, d, lp))

            avg_sim = sum(sim_times) / len(sim_times)
            avg_transmit = sum(transmit_times) / len(transmit_times)
            avg_transfer = sum(transfer_times) / len(transfer_times)

            # PPO update on GPU
            update_start = time.time()
            model.update()
            update_end = time.time()
            gpu_update_time = update_end - update_start

            episode_sim_times.append(avg_sim)
            episode_transmit_times.append(avg_transmit)
            episode_transfer_times.append(avg_transfer)
            episode_gpu_update_times.append(gpu_update_time)

            print(
                f"Episode {episode_counter}: "
                f"Sim {avg_sim:.3f}s, "
                f"Transfer {avg_transfer:.6f}s, "
                f"Transmit {avg_transmit:.3f}s, "
                f"GPU Update {gpu_update_time:.3f}s"
            )

        # Experiment summary
        final_avg_sim = sum(episode_sim_times) / len(episode_sim_times)
        final_avg_transmit = sum(episode_transmit_times) / len(episode_transmit_times)
        final_avg_transfer = sum(episode_transfer_times) / len(episode_transfer_times)
        final_avg_gpu = sum(episode_gpu_update_times) / len(episode_gpu_update_times)
        total_time = (time.time() - start_time) / 60

        print(
            f"\n[Experiment {experiment_idx+1} Summary] "
            f"Sim {final_avg_sim:.3f}s | "
            f"Transfer {final_avg_transfer:.6f}s | "
            f"Transmit {final_avg_transmit:.3f}s | "
            f"GPU {final_avg_gpu:.3f}s | "
            f"Total {total_time:.2f}min\n"
        )

        all_experiment_results.append({
            'Sim': final_avg_sim,
            'Transfer': final_avg_transfer,
            'Transmit': final_avg_transmit,
            'GPU': final_avg_gpu,
            'Total_Minutes': total_time
        })

    pool.close()
    pool.join()

    print("====== All Experiments Summary ======")
    for i, res in enumerate(all_experiment_results):
        print(
            f"[Exp {i+1}] Sim: {res['Sim']:.3f}s | "
            f"Transfer: {res['Transfer']:.6f}s | "
            f"Transmit: {res['Transmit']:.3f}s | "
            f"GPU: {res['GPU']:.3f}s | "
            f"Total: {res['Total_Minutes']:.2f}min"
        )
