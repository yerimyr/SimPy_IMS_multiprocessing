import os
import time
import csv
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
    
    # validation) 학습용 모델(device) 위치 확인
    #print(f"[Main] Training model on device: {model.device}")
    
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
    # Move the inference model to CPU
    agent.policy.to('cpu')  ########## 추론용모델(cpu) 구현 부분 ##########
    agent.device = torch.device('cpu') 
    
    # validation) 추론용 모델이 코어 개수만큼 있는지 확인
    #print(f"[Worker {core_index} PID {os.getpid()}] Inference model id: {id(agent.policy)}")

    # validation) 추론용 모델이 CPU/GPU에 남아있는지 확인
    #print(f"[Worker {core_index} | PID {os.getpid()}] Inference model.device: {agent.device}")
    #print(f"[Worker {core_index}] first_param.device: {next(agent.policy.parameters()).device}")
    
    start_sampling = time.time()
    state = env.reset()
    
    # validation) 시뮬레이터가 CPU에서 동작하는지 확인
    #print(f"[Worker {core_index}] state type: {type(state)}")  # list 혹은 numpy.ndarray 여야 함
    
    done = False
    episode_transitions = []
    episode_reward = 0
    while not done:
        action, log_prob = agent.select_action(state)  ########## 추론용모델(gpu) 구현 부분 ##########
        
        # validation) select_action()이 어디에서 돌아가는지 -> 추론용모델의 위치 확인
        #print(f"[Worker {core_index}] log_prob.device: {log_prob.device}")
        
        next_state, reward, done, _ = env.step(action)
        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state
    finish_sampling = time.time()
    sampling = finish_sampling - start_sampling

    return core_index, sampling, finish_sampling, episode_transitions, episode_reward

def process_transitions(transitions):
    """
    Flattens a list of per-worker transition lists.
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

def  run_training():
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    total_episodes = N_EPISODES
    episode_counter = 0

    # timing records
    episode_copy_times = []
    episode_sampling_times = []
    episode_transfer_times = []
    episode_total_learning_times = []
    episode_learning_times = []
    episode_waiting_times = []

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
        
        # validation) Total parameter count
        #total_params = sum(p.numel() for p in model.policy.parameters())
        #print(f"Total parameters: {total_params}")

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
        transfer_times = []
        finish_sampling_times = []
        # validation) integrated buffer가 맞는지 확인
        #print(f"[Main] Integrated buffer: pool.map 시작 -> {batch_workers} tasks")
        #start_collect = time.time()
        # integrated buffer: gather all worker results synchronously
        results = pool.map(worker_wrapper, tasks)  ########## integrated buffer 구현 부분 ##########
        #collect_time = time.time() - start_collect
        #print(f"[Main] Integrated buffer: 모든 워커 완료 ({len(results)} results) 수집 시간 {collect_time:.3f}s")

        all_transitions = []
        for core_index, sampling, finish_sampling, transitions, episode_reward in results:
            receive_time = time.time()
            transfer = receive_time - finish_sampling

            sampling_times.append(sampling)
            transfer_times.append(transfer)
            finish_sampling_times.append(finish_sampling)
            all_transitions.extend(transitions)

            # tensorboard
            episode_counter += 1
            main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter)
            main_writer.add_scalar("reward_average", episode_reward, episode_counter)

        # store all transitions at once
        start_total_learning = time.time()
        states, actions, rewards, next_states, dones, log_probs = process_transitions([all_transitions])
        for s, a, r, ns, d, lp in zip(states, actions, rewards, next_states, dones, log_probs):
            model.store_transition((s, a, r, ns, d, lp))

        # total learning update on GPU
        model.update()
        total_learning = time.time() - start_total_learning
        episode_total_learning_times.append(total_learning)

        # learning time
        learn = model.learn_time
        episode_learning_times.append(learn)
        
        waiting_times = [max(finish_sampling_times) - x for x in finish_sampling_times]
        transfer_times = [x - y for x, y in zip(transfer_times, waiting_times)]
        
        avg_sampling = sum(sampling_times) / len(sampling_times) 
        avg_transfer = sum(transfer_times) / len(transfer_times)
        avg_waiting = sum(waiting_times) / len(waiting_times)


        episode_sampling_times.append(avg_sampling)
        episode_transfer_times.append(avg_transfer)
        episode_waiting_times.append(avg_waiting)

        print(
            f"Episode {episode_counter}: Copy {copy_time:.3f}s, Sampling {avg_sampling:.3f}s, "
            f"Transfer {avg_transfer:.3f}s, Total_Learning {total_learning:.3f}s, Learning {learn:.3f}s"
        )

    # experiment summary
    total_time = (time.time() - start_time) / 60
    final_avg_param_copy = sum(episode_copy_times)
    final_avg_sampling = sum(episode_sampling_times)
    final_avg_transfer = sum(episode_transfer_times)
    final_avg_total_learning = sum(episode_total_learning_times)
    final_avg_learning = sum(episode_learning_times)
    final_avg_waiting = sum(episode_waiting_times)

    print(
        f"\n[Experiment Summary] "
        f"Copy {final_avg_param_copy:.6f}s | "
        f"Sampling {final_avg_sampling:.6f}s | "
        f"Waiting {final_avg_waiting:.6f}s | "
        f"Transfer {final_avg_transfer:.6f}s | "
        f"Total_Learn {final_avg_total_learning:.6f}s | "
        f"Learn {final_avg_learning:.6f}s | "
        f"Total {total_time:.6f}min\n"
    )
    # Assuming the variables are already calculated as in your summary
    data = {
        'Copy': final_avg_param_copy,
        'Sampling': final_avg_sampling,
        'Waiting1':final_avg_waiting,
        'Transfer': final_avg_transfer,
        'Total_Learn': final_avg_total_learning,
        'Learn': final_avg_learning,
        'Total': total_time
    }

    # Save to CSV
    with open(f"{N_MULTIPROCESS}core_test6_누적.csv", 'w', newline='') as csvfile:
        fieldnames = ['Copy', 'Sampling', 'Waiting1', 'Transfer', 'Total_Learn', 'Learn', 'Total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({key: f'{value:.6f}' for key, value in data.items()})
    
    pool.close()
    pool.join()

if __name__ == '__main__':
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=0,
            active=100,
            repeat=1 
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(TENSORFLOW_LOGS),
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    )
    with profiler as prof:
        run_training()