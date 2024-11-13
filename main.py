"""
This script is used to evaluate the performance of the ev2gym environment.
"""
from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.gurobi_models.tracking_error import PowerTrackingErrorrMin
from ev2gym.baselines.gurobi_models.PST_V2G_profit_max import PST_V2GProfitMaxOracleGB
from ev2gym.utilities.utils import print_statistics
# from ev2gym.baselines.gurobi_models.profit_max import V2GProfitMaxOracleGB
# from ev2gym.baselines.mpc.ocmf_mpc import OCMF_V2G, OCMF_G2V
from ev2gym.baselines.mpc.eMPC import eMPC_V2G, eMPC_G2V
# from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxOracle

from ev2gym.baselines.heuristics import RoundRobin, ChargeAsLateAsPossible, ChargeAsFastAsPossible
# from ev2gym.baselines.heuristics import RoundRobin_GF_off_allowed, RoundRobin_GF
# from ev2gym.baselines.heuristics import ChargeAsFastAsPossibleToDesiredCapacity
# from GNN.state import PublicPST_GNN
from ev2gym.rl_agent.reward import SimpleReward, ProfitMax_TrPenalty_UserIncentives
from ev2gym.rl_agent.state import PublicPST

from GNN.state import PublicPST_GNN_full_graph, PublicPST_GNN

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm
import yaml
import os

# from TD3.TD3_ActionGNN import TD3_ActionGNN


def eval():
    """
    Runs an evaluation of the ev2gym environment.
    """

    save_plots = True

    replay_path = "./replay/replay_sim_2024_02_21_056441.pkl"
    replay_path = None
    
    config_file = "./config_files/PST_V2G_ProfixMax_25.yaml"

    # env = EV2Gym(config_file=config_file,
    #              load_from_replay_path=replay_path,
    #              verbose=False,
    #              save_replay=True,
    #              save_plots=save_plots,
    #              state_function=PublicPST_GNN,
    #              reward_function=SimpleReward,
    #              )

    env = EV2Gym(config_file=config_file,
                 generate_rnd_game=True,
                 save_replay=True,
                 save_plots=save_plots,    
                #  lightweight_plots=True,
                # reward_function = ProfitMax_TrPenalty_UserIncentives,
                state_function= PublicPST,
                 verbose=False,             
                 )
    
    _,_ = env.reset()
    # agent = RoundRobin_GF(env, verbose=False)
    agent = ChargeAsFastAsPossible()
    # agent = eMPC_V2G(env, control_horizon=15, verbose=False)
    # agent = RoundRobin_GF_off_allowed(env, verbose=False)
    
    # env = FailedActionCommunication(env, p_fail=0)
    # env = DelayedObservation(env, p_delay=0)
    
    # env = Rescale_RepairLayer(env)
    # env = MinMax_RepairLayer(env,verbose=True)

    for _ in range(env.simulation_length):
        # actions = np.ones(env.cs*env.number_of_ports_per_cs)
        actions = agent.get_action(env)
        # actions = np.random.rand(env.action_space.shape[0])
        
        new_state, reward, done, truncated, stats = env.step(actions)  # takes action

        # input("Press Enter to continue...")
        if done:
            # print(stats)
            print_statistics(env)
            break
    # return 

    new_replay_path = f"replay/replay_{env.sim_name}.pkl"

    env = EV2Gym(config_file=config_file,
                 load_from_replay_path=new_replay_path,
                 save_plots=save_plots,
                 # state_function=PublicPST_GNN,
                 # reward_function=SimpleReward,
                 )

    # env = BinaryAction(env)

    state, _ = env.reset()

    ev_profiles = env.EVs_profiles
    max_time_of_stay = max([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])
    min_time_of_stay = min([ev.time_of_departure - ev.time_of_arrival
                            for ev in ev_profiles])

    print(f'Number of EVs: {len(ev_profiles)}')
    print(f'Max time of stay: {max_time_of_stay}')
    print(f'Min time of stay: {min_time_of_stay}')
    # exit()
    # agent = OCMF_V2G(env, control_horizon=30, verbose=True)
    # agent = OCMF_G2V(env, control_horizon=25, verbose=True)
    # agent = eMPC_V2G(env, control_horizon=25, verbose=True)
    # agent = V2GProfitMaxOracle(env,verbose=True)
    # agent = PowerTrackingErrorrMin(new_replay_path)
    agent = PST_V2GProfitMaxOracleGB(new_replay_path)
    # agent = eMPC_G2V(env, control_horizon=15, verbose=False)
    # agent = RoundRobin(env, verbose=False)
    # agent = ChargeAsLateAsPossible(verbose=False)
    # agent = ChargeAsFastAsPossible()
    # agent = ChargeAsFastAsPossibleToDesiredCapacity()

    for _ in range(env.simulation_length):
        # actions = np.random.rand(env.action_space.shape[0])
        actions = agent.get_action(env)
        
        new_state, reward, done, truncated, stats = env.step(
            actions)  # takes action

        if done:
            print(stats)
            break
            

    exit()
    algorithm = "TD3_ActionGNN_run_0_32590-628564"

    load_model_path = f'./eval_models/{algorithm}/'
    # Load kwargs.yaml as a dictionary
    with open(f'{load_model_path}kwargs.yaml') as file:
        kwargs = yaml.load(file, Loader=yaml.FullLoader)

    if "ActionGNN" in algorithm:
        agent = TD3_ActionGNN(**kwargs)
        agent.load(filename=f'{load_model_path}model.best')

    avg_reward = 0.
    # use tqdm to show progress bar
    stats_list = []
    eval_stats = {}
    eval_episodes = 1

    evaluation_name = "test_eval"
    os.makedirs(f"replay/{evaluation_name}/", exist_ok=True)

    for _ in tqdm(range(eval_episodes)):

        env = EV2Gym(config_file=config_file,
                     generate_rnd_game=True,
                     save_replay=True,
                     replay_save_path=f"replay/{evaluation_name}/",
                     )

        for _ in range(env.simulation_length):
            actions = np.ones(env.cs)

            new_state, reward, done, truncated, _ = env.step(
                actions, visualize=False)  # takes action

            if done:
                break

        replay_path = f"replay/{evaluation_name}/replay_{env.sim_name}.pkl"

        env = EV2Gym(config_file=config_file,
                     load_from_replay_path=replay_path,
                     state_function=PublicPST_GNN,
                     reward_function=SimpleReward,
                     )

        state, _ = env.reset()

        done = False
        while not done:
            action = agent.select_action(state, return_mapped_action=True)
            state, reward, done, _, stats = env.step(action)
            avg_reward += reward

        stats_list.append(stats)

    avg_reward /= eval_episodes
    print(f"Average reward: {avg_reward}")


if __name__ == "__main__":
    # while True:
        eval()
