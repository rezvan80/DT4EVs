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
                case_name = "Original\n(Trained)"
            elif index == 1:
                case_name = "Small\nVariation"
            elif index == 2:
                case_name = "Medium\nVariation"
            elif index == 3:
                case_name = "Extreme\nVariation"
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

    # remove teh algorithm=RR_GF
    data = all_data[all_data.Algorithm != 'RR']

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
                hue_order=[
                    'BaU','DT', 'Q-DT', 'GNN-DT',
                    'Optimal\n(Oracle)'],
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

    plt.savefig("./results_analysis/figs/generalization_eda.png")
    plt.savefig("./results_analysis/figs/generalization_eda.pdf")

    exit()

    for i, case in enumerate(data.case.unique()):

        ax = plt.subplot(1, 4, i+1)
        # all_data = all_data[all_data.case == 'Original']
        # all_data = all_data[all_data.case == 'Small']
        # all_data = all_data[all_data.case == 'Medium']
        all_data = data[data.case == case]

        custom_palette = [
            '#8C8C8C',
            '#b6db97', '#64B5CD',
            '#C44E52', '#DD8452',
            '#55A868', '#4C72B0',
        ]

        # use a boxplot instead
        ax1 = sns.boxplot(x="Algorithm",
                          y="reward",
                            # remove outliers
                            showfliers=False,
                            notch=True,
                            # order=['AFAP',
                            #        'SAC', 'TD3',
                            #        '  SAC  \nFX-GNN', '  TD3  \nFX-GNN',
                            #        '  SAC  \nEV-GNN', '  TD3  \nEV-GNN'],
                            data=all_data,
                            # alpha=0.9,
                            saturation=1,
                            ax=ax,
                            palette=custom_palette,
                          )
        # show the x-ticks
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6],
                      ['']*7,
                      fontsize=11)

        # add grid
        # plt.grid(axis='y', linestyle='--', alpha=0.5)
        ax.grid(which='minor', linestyle='--', alpha=0.3)
        ax.grid(
            which='major', linestyle='-', alpha=0.5)
        ax.grid(
            axis="x",
            which='major', linestyle='--', alpha=0.2)

        plt.yticks(fontsize=11)
        plt.xlabel(f'Algorithm', fontsize=11)

        # if i == 0:
        plt.ylabel('Episode Reward [-]', fontsize=11)
        plt.ticklabel_format(axis='y',
                             style='sci',
                             scilimits=(5, 5))

        if i == 0:
            plt.title(f'{case}\nDistribution', fontsize=12)
        else:
            plt.title(f'{case}\nVariation', fontsize=12)

        if i == 2:
            size = 12
            # make legedn outside the plot for the 7 algorithms used
            custom_patches = [
                mpatches.Patch(facecolor=custom_palette[0],
                               edgecolor='black', label='AFAP'),
                mpatches.Patch(facecolor=custom_palette[1],
                               edgecolor='black', label='SAC'),
                mpatches.Patch(facecolor=custom_palette[2],
                               edgecolor='black', label='TD3'),
                mpatches.Patch(facecolor=custom_palette[3],
                               edgecolor='black', label='   SAC\nFX-GNN'),
                mpatches.Patch(facecolor=custom_palette[4],
                               edgecolor='black', label='   TD3\nFX-GNN'),
                # mpatches.Patch(facecolor=custom_palette[5],
                #                edgecolor='black', label='  SAC\nEV-GNN'),
                # mpatches.Patch(facecolor=custom_palette[6],
                #                edgecolor='black', label='  TD3\nEV-GNN')
            ]

            # Add the legend with the custom patches
            plt.legend(handles=custom_patches,
                       loc='upper center',
                       fontsize=10,
                       ncol=7,
                       title='',
                       title_fontsize=12,
                       frameon=False,
                       bbox_to_anchor=(-0.305, -0.12),
                       )

    # set left and right margins
    plt.subplots_adjust(left=0.089, right=0.992, top=0.867, bottom=0.207)

    # set wspace and hspace
    plt.subplots_adjust(wspace=0.372, hspace=0.3)

    plt.savefig("./results_analysis/figs/generalization_eda.png")
    plt.savefig("./results_analysis/figs/generalization_eda.pdf")


if __name__ == "__main__":
    plot_optimality_gap_in_generalization()
