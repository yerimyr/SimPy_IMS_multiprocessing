import GymWrapper as gw
import time
import HyperparamTuning as ht  # Module for hyperparameter tuning
from GymWrapper import GymInterface
from config_SimPy import *
from config_RL import *
from stable_baselines3 import DQN, DDPG, PPO
from log_SimPy import *
from log_RL import *
from Deep_Q_Network import *
import multiprocessing

NUM_PROCESSES = 14  # 사용하고 싶은 코어 개수

def build_model():
    model = DQNAgent(state_dim=env.observation_space.shape[0], 
                         action_dim=env.action_space.n, 
                         buffer_size=BUFFER_SIZE, lr=LEARNING_RATE, gamma=GAMMA)
    return model


'''
def export_report(inventoryList):
    for x in range(len(inventoryList)):
        for report in DAILY_REPORTS:
            export_Daily_Report.append(report[x])
    daily_reports = pd.DataFrame(export_Daily_Report)
    daily_reports.columns = ["Day", "Name", "Type",
                         "Start", "Income", "Outcome", "End"]
    daily_reports.to_csv("./Daily_Report.csv")
'''


# Start timing the computation
start_time = time.time()

# Create environment
env = GymInterface()

# Run hyperparameter optimization if enabled
if OPTIMIZE_HYPERPARAMETERS:
    ht.run_optuna()
    # Calculate computation time and print it
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes ")
else:
    # Build the model
    if LOAD_MODEL:
        if RL_ALGORITHM == "DQN":
            model = DQN.load(os.path.join(
                SAVED_MODEL_PATH, LOAD_MODEL_NAME), env=env)
        print(f"{LOAD_MODEL_NAME} is loaded successfully")
    else:
        model = build_model()
        # Train the model
        for episode in range(N_EPISODES):
            state = env.reset()
            done = False
            while not done:
                action = model.select_action(state, epsilon = max(0.1, 1.0 - episode/500))
                next_state, reward, done, _ = env.step(action)                
                model.buffer.push(state, action, reward, next_state, done)
                state = next_state
                model.update(batch_size=32)
        if SAVE_MODEL:
            model.save(os.path.join(SAVED_MODEL_PATH, SAVED_MODEL_NAME))
            print(f"{SAVED_MODEL_NAME} is saved successfully")

        if STATE_TRAIN_EXPORT:
            gw.export_state('TRAIN')
    training_end_time = time.time()

    # Evaluate the trained model
    mean_reward, std_reward = gw.evaluate_model(model, env, N_EVAL_EPISODES)
    print(
        f"Mean reward over {N_EVAL_EPISODES} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")
    # Calculate computation time and print it
    end_time = time.time()
    print(f"Computation time: {(end_time - start_time)/60:.2f} minutes \n",
          f"Training time: {(training_end_time - start_time)/60:.2f} minutes \n ",
          f"Test time:{(end_time - training_end_time)/60:.2f} minutes")


# Optionally render the environment
env.render()