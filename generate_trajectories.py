import os
import time
import numpy as np
import pickle
import yaml
from tqdm import tqdm 

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.utilities.arg_parser import arg_parser
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, profit_maximization, SimpleReward
from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads
from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible, RandomAgent
from ev2gym.baselines.gurobi_models.PST_V2G_profit_max_mo import mo_PST_V2GProfitMaxOracleGB
from utils import PST_V2G_ProfitMax_reward, PST_V2G_ProfitMaxGNN_state,PST_V2G_ProfitMax_state

from ev2gym.baselines.mpc.eMPC_v2 import eMPC_V2G_v2


if __name__ == "__main__":

    args = arg_parser()

    # Define the directory where to save and load models
    checkpoint_dir = args.save_dir + args.env
    args.config_file = "./config_files/PST_V2G_ProfixMax_25.yaml"
    # reward_function = SquaredTrackingErrorReward
    reward_function = PST_V2G_ProfitMax_reward
    state_function = PST_V2G_ProfitMax_state
    problem = args.config_file.split("/")[-1].split(".")[0]

    env = EV2Gym(config_file=args.config_file,                            
                            state_function=state_function,
                            reward_function=reward_function,
                            )
    
    temp_env = EV2Gym(config_file=args.config_file,
                 save_replay=True,
                 reward_function=reward_function,
                 state_function=state_function,
                 )
    
    n_trajectories = args.n_trajectories
        
    config = yaml.load(open(args.config_file, 'r'), Loader=yaml.FullLoader)
    number_of_charging_stations = config["number_of_charging_stations"]
    n_transformers = config["number_of_transformers"]
    steps = config["simulation_length"]
    timescale = config["timescale"]
    
    trajectories = []

    trajecotries_type = "random" #args.dataset

    file_name = f"{problem}_{trajecotries_type}_{number_of_charging_stations}_{n_trajectories}.pkl"
    save_folder_path = f"./trajectories/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    epoch = 0
    # use tqdm with a fancy bar
    for i in tqdm(range(n_trajectories)):

        trajectory_i = {"observations": [],
                        "actions": [],
                        "rewards": [],
                        "dones": []}

        epoch_return = 0

        if trajecotries_type == "random":
            agent = RandomAgent(env)
            
        elif trajecotries_type == "optimal":
            _, _ = temp_env.reset()
            agent = ChargeAsFastAsPossible()

            for _ in range(temp_env.simulation_length):
                actions = agent.get_action(temp_env)
                new_state, reward, done, truncated, stats = temp_env.step(
                    actions)  # takes action
                if done:
                    break
                
            new_replay_path = f"./replay/replay_{temp_env.sim_name}.pkl"
            
            agent = mo_PST_V2GProfitMaxOracleGB(new_replay_path,
                                                timelimit=60,
                                                MIPGap=None,
                                                )
            
            # delete the new_replay_path file
            os.remove(new_replay_path)
        
        elif trajecotries_type == "mpc":
            agent = eMPC_V2G_v2(env, 
                        control_horizon=10,
                        MIPGap = 0.1,
                        time_limit=30,
                        verbose=False)
        else:
            raise ValueError(f"Trajectories type {trajecotries_type} not supported")
        
                
        state, _ = env.reset()
        
        while True:

            actions = agent.get_action(env)

            new_state, reward, done, truncated, _ = env.step(actions)

            trajectory_i["observations"].append(state)
            trajectory_i["actions"].append(actions)
            trajectory_i["rewards"].append(reward)
            trajectory_i["dones"].append(done)            
            state = new_state

            if done:
                break

        trajectory_i["observations"] = np.array(trajectory_i["observations"])
        trajectory_i["actions"] = np.array(trajectory_i["actions"])
        trajectory_i["rewards"] = np.array(trajectory_i["rewards"])
        trajectory_i["dones"] = np.array(trajectory_i["dones"])

        trajectories.append(trajectory_i)

        if i % 100 == 0:
            print(f'Saving trajectories to {save_folder_path+file_name}')
            f = open(save_folder_path+file_name, 'wb')
            # source, destination
            pickle.dump(trajectories, f)

    env.close()
    print(trajectories[:1])

    print(f'Saving trajectories to {save_folder_path+file_name}')
    f = open(save_folder_path+file_name, 'wb')
    # source, destination
    pickle.dump(trajectories, f)
    f.close()
