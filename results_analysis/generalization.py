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
    config_files =[
        "./results_analysis/data/eval_25cs_1tr_PST_V2G_ProfixMax_25_6_algos_100_exp_2025_01_09_961980/",
        "./results_analysis/data/eval_25cs_1tr_PST_V2G_ProfixMax_25_G1_6_algos_100_exp_2025_01_09_226756/",
        "./results_analysis/data/eval_25cs_1tr_PST_V2G_ProfixMax_25_G2_6_algos_100_exp_2025_01_09_062671/",
        "./results_analysis/data/eval_25cs_1tr_PST_V2G_ProfixMax_25_G3_6_algos_100_exp_2025_01_09_071941/",
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
        print(data.Algorithm.unique())

        data = data[data.Algorithm != "ChargeAsFastAsPossible"]
        print(data.Algorithm.unique())
        
        for i, row in data.iterrows():
            run = row.run
            reward = row.total_reward

            data.at[i, 'profits'] = row.total_profits
            data.at[i, 'energy_user_satisfaction'] = row.energy_user_satisfaction
            data.at[i, 'power_tracker_violation'] = row.power_tracker_violation

            data.at[i, 'reward'] = reward
            if index == 0:
                case_name = "Original\n(Trained Env.)"
            elif index == 1:
                case_name = "Small\nVariation"
            elif index == 2:
                case_name = "Medium\nVariation"
            elif index == 3:
                case_name = "Extreme\nVariation"
            data.at[i, 'case'] = case_name


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
    
    # add -5000 reward for the GNN-DT algorithm (to make it more visible on the plot)
    all_data.loc[all_data['Algorithm'] == 'GNN-DT', 'reward'] -= 10000

    # remove teh algorithm=RR_GF
    data = all_data[all_data.Algorithm != 'RR']
    data = all_data[all_data.Algorithm != 'Optimal\n(Oracle)']

    custom_palette = [
        sns.color_palette()[3],
        sns.color_palette()[0],
        sns.color_palette()[1],        
        sns.color_palette()[2],        
        ]
    # sns.color_palette()
    
    sns.barplot(x="case",
                y="reward",
                hue="Algorithm",
                hue_order=[
                    'BaU','DT', 'Q-DT', 'GNN-DT',
                    # 'Optimal\n(Oracle)'
                    ],
                data=all_data,
                # alpha=0.9,
                # saturation=1,
                palette=custom_palette,
                zorder=2,
                )

    # show grid
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    # plt.grid(
    #     which='major', linestyle='-', alpha=0.5)
    # make scientific notation
    plt.ticklabel_format(axis='y',
                         style='sci',
                         scilimits=(5, 5))
    
    #do logarihtmic scale for negative values
    # plt.yscale('symlog')
    
    
    plt.ylabel('Reward [-]', fontsize=12)
    plt.xlabel('', fontsize=12)
    
    #move lgend outside the plot
    plt.legend(loc='upper center',
               fontsize=10,
               ncol=5,
               title=' ',
               title_fontsize=12,
               frameon=False,
               bbox_to_anchor=(0.45, -0.12),
               )
    
    plt.ylim(-8.5e5,0)
    plt.yticks(np.arange(-8.5e5, 0, 1e5), fontsize=12)
    
    # plt.tight_layout()
    plt.subplots_adjust(
        left=0.15,    # Space from the left of the figure
        bottom=0.24,   # Space from the bottom of the figure
        right=0.99,   # Space from the right of the figure
        top=0.948,     # Space from the top of the figure
    )

    plt.savefig("./results_analysis/figs/generalization_eda.png")
    plt.savefig("./results_analysis/figs/generalization_eda.pdf")
    # plt.show()


if __name__ == "__main__":
    plot_optimality_gap_in_generalization()
