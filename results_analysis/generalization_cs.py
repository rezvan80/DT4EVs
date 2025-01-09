import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import time
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
import matplotlib.patches

# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE


def plot_optimality_gap_in_generalization():
    config_files = [
        "./results/eval_25cs_1tr_PST_V2G_ProfixMax_25_6_algos_1_exp_2025_01_09_265600/",
        "./results/eval_25cs_1tr_PST_V2G_ProfixMax_25_6_algos_1_exp_2025_01_09_265600/",
        "./results/eval_25cs_1tr_PST_V2G_ProfixMax_25_6_algos_1_exp_2025_01_09_265600/",
        "./results/eval_25cs_1tr_PST_V2G_ProfixMax_25_6_algos_1_exp_2025_01_09_265600/",
        "./results/eval_25cs_1tr_PST_V2G_ProfixMax_25_6_algos_1_exp_2025_01_09_265600/",
    ]

    columns_to_keep = [
        'run',
        'Algorithm',
        'total_reward',
        'total_energy_charged',
        'total_energy_discharged',
        'power_tracker_violation',
        'energy_user_satisfaction',
        'total_profits',
    ]

    figure = plt.figure(figsize=(6, 4))

    all_data = pd.DataFrame()

    for index, config_file in enumerate(config_files):
        with open(config_file + "data.csv", "r") as f:
            # read as pandas dataframe
            data = pd.read_csv(f)

        data = data[columns_to_keep]
        print(data.columns)
        print(data.Algorithm.unique())

        # ChargeAsFastAsPossible

        PowerTrackingErrorrMin = data[data.Algorithm ==
                                      "PowerTrackingErrorrMin"]
        # print(PowerTrackingErrorrMin.head(20))

        # find the mean and std of the optimality gap for each algorithm

        data = data[data.Algorithm != "ChargeAsFastAsPossible"]
        for i, row in data.iterrows():
            run = row.run
            # optimal = PowerTrackingErrorrMin.iloc[run].total_reward
            reward = row.total_reward
            # data.at[i, 'G'] = ((reward - optimal) / optimal) * 100

            # optimal_energy_tracking_error = PowerTrackingErrorrMin.iloc[run].energy_tracking_error
            # energy_tracking_error = row.energy_tracking_error
            # data.at[i, 'G_E'] = (
            #     (energy_tracking_error - optimal_energy_tracking_error) / optimal_energy_tracking_error) * 100

            data.at[i, 'profits'] = row.total_profits
            data.at[i, 'energy_user_satisfaction'] = row.energy_user_satisfaction
            data.at[i, 'power_tracker_violation'] = row.power_tracker_violation

            data.at[i, 'reward'] = reward
            if index == 0:
                case_name = "5 CS"
            elif index == 1:
                case_name = "25 CS\n(Trained)"
            elif index == 2:
                case_name = "50 CS"
            elif index == 3:
                case_name = "75 CS"                
            elif index == 4:
                case_name = "100 CS"
                
            data.at[i, 'case'] = case_name

        data['Algorithm'] = data['Algorithm'].replace({"<class 'ev2gym.baselines.heuristics.ChargeAsFastAsPossible'>": 'AFAP',
                                                       "ChargeAsFastAsPossible": 'AFAP',
                                                       "RoundRobin": 'RR',
                                                       "<class 'ev2gym.baselines.heuristics.RoundRobin_GF'>": 'RR_GF',
                                                       "<class 'ev2gym.baselines.heuristics.RoundRobin'>": 'RR',
                                                       "<class 'ev2gym.baselines.gurobi_models.tracking_error.PowerTrackingErrorrMin'>": 'Optimal',
                                                       "PowerTrackingErrorrMin": 'Optimal'}
                                                      )

        # change Algorithm if string is in Algorithm
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: 'GNN-DT' if 'GNN_act_emb_DT' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: 'Q-DT' if 'QT' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: 'Optimal\n(Oracle)' if 'mo_PST_V2GProfitMaxOracleGB' in x else x)
        data['Algorithm'] = data['Algorithm'].apply(
            lambda x: 'BaU' if 'RoundRobin_GF' in x else x)

        all_data = pd.concat([all_data, data])

    plt.rcParams.update({'font.size': 12})

    plt.rcParams['font.family'] = 'serif'
    data = all_data[all_data.Algorithm != 'Q-DT']
    data = all_data[all_data.Algorithm != 'DT']

    # make subplots

    # sns.boxplot(x="case",
    #             y="reward",
    #             hue="Algorithm",
    #             # remove outliers
    #             showfliers=False,
    #             notch=True,
    #             hue_order=[
    #                 'BaU','DT', 'Q-DT', 'GNN-DT',
    #                 'Optimal\n(Oracle)'],
    #             data=all_data,
    #             # alpha=0.9,
    #             saturation=1,
    #             # palette=custom_palette,
    #             )
    
    sns.barplot(x="case",
                y="reward",
                hue="Algorithm",
                # remove outliers
                # showfliers=False,
                # notch=True,
                # hue_order=[
                #     'BaU','DT', 'Q-DT', 'GNN-DT',
                #     'Optimal\n(Oracle)'],
                data=all_data,
                # alpha=0.9,
                # saturation=1,
                # palette=custom_palette,
                zorder=2,
                )
    
    # sns.catplot(x="case",
    #             y="reward",
    #             hue="Algorithm",
    #             kind='bar',
    #             hue_order=[
    #                 'BaU','DT', 'Q-DT', 'GNN-DT',
    #                 'Optimal\n(Oracle)'],
    #             data=all_data,

    #             # alpha=0.9,
    #             # saturation=1,
    #             # palette=custom_palette,
    #             zorder=2,
    #             )

    # show grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.grid(
    #     which='major', linestyle='-', alpha=0.5)
    # make scientific notation
    plt.ticklabel_format(axis='y',
                         style='sci',
                         scilimits=(5, 5))
    
    plt.ylabel('Reward [-]', fontsize=12)
    plt.xlabel('', fontsize=12)
    
    #move lgend outside the plot
    plt.legend(loc='upper center',
               fontsize=10,
               ncol=5,
               title=' ',
               title_fontsize=12,
               frameon=False,
               bbox_to_anchor=(0.45, -0.15),
               )
    
    plt.tight_layout()

    plt.savefig("./results_analysis/figs/generalization_eda_CS.png")
    plt.savefig("./results_analysis/figs/generalization_eda_CS.pdf")


if __name__ == "__main__":
    plot_optimality_gap_in_generalization()
