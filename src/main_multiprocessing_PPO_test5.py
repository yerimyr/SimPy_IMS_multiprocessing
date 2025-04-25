import os
import time
import multiprocessing

import torch
from torch.utils.tensorboard import SummaryWriter

from GymWrapper import GymInterface
from PPO import PPOAgent
from config_RL import *

# TensorBoard writer
main_writer = SummaryWriter(log_dir=TENSORFLOW_LOGS)

N_MULTIPROCESS = 5

def build_model(env):
    """
    새로운 PPO 에이전트를 만듭니다.
    (기본적으로 DEVICE에 따라 GPU에 올라갑니다)
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
    워커 프로세스: 
    1) 학습용 모델의 파라미터를 로드
    2) 추론용 모델만 CPU로 이동하며 시간 측정
    3) 전체 에피소드 시뮬레이션 실행
    """
    # 1) 환경 및 에이전트 생성 (에이전트는 기본적으로 GPU에 올라감)
    env = GymInterface()
    agent = build_model(env)

    # 2) 파라미터 로드 + CPU로 이동, 걸린 시간 측정
    start_transfer = time.time()
    agent.policy.load_state_dict(model_state_dict)
    agent.policy.to('cpu')   # 여기서 CPU로 복사
    agent.device = torch.device('cpu')
    transfer_time = time.time() - start_transfer

    print(f"[Worker {core_index}] Parameter transfer to CPU: {transfer_time:.6f} sec")

    # 3) 시뮬레이션 루프
    start_sim_time = time.time()
    state = env.reset()
    done = False
    episode_transitions = []
    episode_reward = 0.0

    while not done:
        # CPU 위에서 추론
        action, log_prob = agent.select_action(state)
        # CPU 시뮬레이터 실행
        next_state, reward, done, _ = env.step(action)

        episode_transitions.append((state, action, reward, next_state, done, log_prob.item()))
        episode_reward += reward
        state = next_state

    finish_sim_time = time.time()
    sim_time = finish_sim_time - start_sim_time

    # 워커가 메인에 반환하는 값에 transfer_time 추가
    return core_index, sim_time, finish_sim_time, episode_transitions, episode_reward, transfer_time

def worker_wrapper(args):
    return simulation_worker(*args)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    pool = multiprocessing.Pool(processes=N_MULTIPROCESS)

    all_experiment_results = []

    # 실험을 3회 반복
    for experiment_idx in range(3):
        print(f"========== Experiment {experiment_idx+1} ========== ")

        # 학습용 모델 생성 (GPU)
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
        episode_transfer_times = []
        episode_transmit_times = []
        episode_gpu_update_times = []

        start_time = time.time()

        # 에피소드 루프
        while episode_counter < total_episodes:
            batch_workers = min(N_MULTIPROCESS, total_episodes - episode_counter)
            # 1) 최신 파라미터 떼오기 (GPU 메모리)
            model_state_dict = model.policy.state_dict()

            # 2) 워커에 전달할 태스크 생성
            tasks = [(i, model_state_dict) for i in range(batch_workers)]

            # 3) 병렬 실행 및 결과 수집
            sim_times = []
            transmit_times = []

            for result in pool.imap_unordered(worker_wrapper, tasks):
                (core_index,
                 sim_time,
                 finish_sim_time,
                 transitions,
                 episode_reward,
                 transfer_time) = result

                # 파라미터 전송 시간 기록
                episode_transfer_times.append(transfer_time)

                # 시뮬레이션 실행 시간 기록
                sim_times.append(sim_time)

                # 전송 지연 시간 기록
                receive_time = time.time()
                transmit_times.append(receive_time - finish_sim_time)

                # transition을 메인 모델 메모리에 저장
                for tr in transitions:
                    model.store_transition(tr)

                episode_counter += 1
                main_writer.add_scalar(f"reward_core_{core_index+1}", episode_reward, episode_counter)
                main_writer.add_scalar("reward_average", episode_reward, episode_counter)

            # 4) GPU에서 PPO 업데이트
            update_start = time.time()
            model.update()
            update_end = time.time()
            gpu_update_time = update_end - update_start

            # 평균 전송 시간 계산 및 출력
            avg_transfer = sum(episode_transfer_times) / len(episode_transfer_times)

            # 기록들 모아두기
            episode_sim_times.append(sum(sim_times)/len(sim_times))
            episode_transmit_times.append(sum(transmit_times)/len(transmit_times))
            episode_gpu_update_times.append(gpu_update_time)

            print(f"Episode {episode_counter}: "
                  f"Sim {episode_sim_times[-1]:.3f}s, "
                  f"Last Transfer {episode_transfer_times[-1]:.6f}s, "
                  f"Avg Transfer {avg_transfer:.6f}s, "
                  f"Transmit {episode_transmit_times[-1]:.3f}s, "
                  f"GPU Update {gpu_update_time:.3f}s")

        # 실험별 요약
        final_avg_sim_time = sum(episode_sim_times) / len(episode_sim_times)
        final_avg_transmit_time = sum(episode_transmit_times) / len(episode_transmit_times)
        final_avg_gpu_update_time = sum(episode_gpu_update_times) / len(episode_gpu_update_times)
        final_avg_transfer_time = sum(episode_transfer_times) / len(episode_transfer_times)

        computation_time = (time.time() - start_time) / 60

        print(f"\n[Experiment {experiment_idx+1} Summary] "
              f"Sim {final_avg_sim_time:.3f}s | "
              f"Avg Transfer {final_avg_transfer_time:.6f}s | "
              f"Transmit {final_avg_transmit_time:.3f}s | "
              f"GPU {final_avg_gpu_update_time:.3f}s | "
              f"Total {computation_time:.2f}min\n")

        all_experiment_results.append({
            'Sim': final_avg_sim_time,
            'Transfer': final_avg_transfer_time,
            'Transmit': final_avg_transmit_time,
            'GPU': final_avg_gpu_update_time,
            'Total_Minutes': computation_time
        })

    pool.close()
    pool.join()

    print("====== All Experiments Summary ======")
    for i, res in enumerate(all_experiment_results):
        print(f"[Exp {i+1}] Sim: {res['Sim']:.3f}s | "
              f"Transfer: {res['Transfer']:.6f}s | "
              f"Transmit: {res['Transmit']:.3f}s | "
              f"GPU: {res['GPU']:.3f}s | "
              f"Total: {res['Total_Minutes']:.2f}min")