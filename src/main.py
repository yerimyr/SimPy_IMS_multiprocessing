import GymWrapper as gw
import time
import HyperparamTuning as ht
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from log_SimPy import *
from log_RL import *
from Deep_Q_Network import *
import multiprocessing
import os

NUM_PROCESSES = 5  # 사용하고 싶은 코어 개수

def build_model():
    """
    새로운 DQN 모델을 생성
    """
    env = GymInterface()  # 환경을 초기화
    model = DQNAgent(
        state_dim=env.observation_space.shape[0], 
        action_dim=env.action_space.n, 
        buffer_size=BUFFER_SIZE, 
        lr=LEARNING_RATE, 
        gamma=GAMMA
    )
    return model

def run_episode(process_id):
    """
    멀티프로세스로 실행할 각 에피소드 학습 함수
    """
    env = GymInterface()  # 각 프로세스에서 독립적인 환경 생성
    model = build_model()  # 각 프로세스에서 독립적인 모델 생성

    for episode in range(N_EPISODES // NUM_PROCESSES):  # 각 프로세스에서 실행할 에피소드 수 분배
        state = env.reset()
        done = False
        while not done:
            action = model.select_action(state, epsilon=max(0.1, 1.0 - episode / 500))
            next_state, reward, done, _ = env.step(action)
            model.buffer.push(state, action, reward, next_state, done)
            state = next_state
            model.update(batch_size=32)

    return model  # 학습된 모델 반환

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)  # Windows에서 멀티프로세싱 실행 설정

    start_time = time.time()  # 실행 시간 측정 시작

    # 멀티프로세스 실행
    with multiprocessing.Pool(NUM_PROCESSES) as pool:
        models = pool.map(run_episode, range(NUM_PROCESSES))

    # 가장 좋은 모델 선택 (총 보상이 가장 높은 모델)
    best_model = max(models, key=lambda m: sum(m.total_reward_over_episode) if m.total_reward_over_episode else 0)
    # 최종 모델 저장
    if SAVE_MODEL:
        best_model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
        print(f"{SAVED_MODEL_NAME} is saved successfully")

    training_end_time = time.time()  # 학습 종료 시간 측정

    # 학습된 모델 평가
    env = GymInterface()  # 새로운 평가 환경 생성
    mean_reward, std_reward = gw.evaluate_model(best_model, env, N_EVAL_EPISODES)
    print(f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 실행 시간 출력
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n",
          f"Test time: {(end_time - training_end_time)/60:.2f} minutes")

    # 환경 렌더링 (필요하면 실행)
    env.render()
