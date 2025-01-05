import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import dataset_info, parse_string_to_list

fig, axs = plt.subplots(3, 6, figsize=(12, 8))
all_data = pd.read_csv("./results_analysis/results.csv")

print(axs)
for index in range(6):
    for alg_index, algorithm in enumerate(['dt', 'QT', 'gnn_act_emb']):

        if alg_index == 0:
            algo_name = 'DT'
        elif alg_index == 1:
            algo_name = 'Q-DT'
        else:
            algo_name = 'GNN-DT'

        if index < 3:
            K = 2
        else:
            K = 10

        if index % 3 == 0:
            dataset_name = 'Optimal'
            datasets_list = [
                'optimal_100',
                'optimal_1000',
                'optimal_10000',
            ]

        elif index % 3 == 1:
            dataset_name = 'Random'
            datasets_list = [
                'random_100',
                'random_1000',
                'random_10000',
            ]
        else:
            dataset_name = 'BaU'
            datasets_list = [
                'bau_100',
                'bau_1000',
                'bau_10000',
            ]

        print(
            f'Index: {index}, Alg_index: {alg_index}, K: {K}, Algorithm: {algorithm}')

        data = all_data[(all_data["K"] == K)]
        data = data[(data["algorithm"] == algorithm)]

        data = data[data["dataset"].isin(datasets_list)]
        # dataset_info(data)
        # For every row in the data create a new dataframe with epoch as the index and the reward as the value, keep also, the seed, algorithm and dataset

        new_df = pd.DataFrame()
        for i, row in data.iterrows():
            # print(row)
            # parse the string to a list
            rewards = parse_string_to_list(row["eval_reward"])

            for j in range(250):
                # if there is no value for the epoch, use the last value

                reward = rewards[j] if j < len(rewards) else rewards[-1]
                entry = {
                    "epoch": j,
                    "reward": reward,
                    "seed": row["seed"],
                    "algorithm": row["algorithm"],
                    "dataset": row["dataset"]
                }
                new_df = pd.concat([new_df, pd.DataFrame([entry])])

        new_df["algorithm"] = new_df["algorithm"].replace("dt", "DT")
        # from QT to Q-DT
        new_df["algorithm"] = new_df["algorithm"].replace("QT", "Q-DT")
        # from gnn_act_emb to GNN-DT
        new_df["algorithm"] = new_df["algorithm"].replace(
            "gnn_act_emb", "GNN-DT")

        # plot the data
        sns.set_theme(style="whitegrid")
        plt.rcParams['font.family'] = 'serif'

        print(f' Data ready to plot')
        
        if index == 0 and alg_index == 2:
            #rename dataset names to show the number of samples
            new_df["dataset"] = new_df["dataset"].replace("optimal_100", "100")
            new_df["dataset"] = new_df["dataset"].replace("optimal_1000", "1000")
            new_df["dataset"] = new_df["dataset"].replace("optimal_10000", "10000")

        sns.lineplot(data=new_df,
                     x="epoch",
                     y="reward",
                     hue="dataset",
                     # hue_order=hue_order,
                     ax=axs[alg_index][index],
                     )

        
        if alg_index == 0 and index == 0:
            plot_title = f"   {dataset_name} | K={K}"
            axs[alg_index][index].set_title(plot_title, fontsize=15)
        elif alg_index == 0 and index != 0:
            plot_title = f"{dataset_name} | K={K}"
            axs[alg_index][index].set_title(plot_title, fontsize=15)

        axs[alg_index][index].axhline(
            y=-2405, color='r', linestyle='--')

        # show grid lines
        axs[alg_index][index].grid(True)

        if index == 0 and alg_index == 2:
            axs[alg_index][index].legend(loc='upper center',
                                         bbox_to_anchor=(3.5, -0.22),
                                         ncol=3,
                                         fontsize=12,
                                         title="Number of Trajectory Samples",
                                         title_fontsize=12)
        
        else:
            axs[alg_index][index].get_legend().remove()

        if index != 0:
            axs[alg_index][index].set_ylabel("")
        else:
            axs[alg_index][index].set_ylabel(f"{algo_name}\nReward [-]",
                                             fontsize=14)

        if alg_index != 2:
            axs[alg_index][index].set_xlabel("")
        else:
            axs[alg_index][index].set_xlabel("Epoch", fontsize=14)

        # Set xticks and yticks font size
        axs[alg_index][index].tick_params(axis='x', labelsize=12)
        # show ticks on the y-axis onl the first time
        if index == 0:
            axs[alg_index][index].tick_params(axis='y', labelsize=13)
            axs[alg_index][index].ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0))
        else:
            # remove yticks labels but keep the ticks
            axs[alg_index][index].set_yticklabels([])

        # Set xlim and ylim
        axs[alg_index][index].set_xlim(0, 150)
        axs[alg_index][index].set_ylim(-400_000, 10_000)
        print(f'Plot ready')

        # if alg_index == 1 and index == 1:
        #     plt.tight_layout()
        #     plt.show()
        #     exit()

# Adjust layout
fig.tight_layout()
plt.subplots_adjust(
    left=0.07,    # Space from the left of the figure
    bottom=0.138,   # Space from the bottom of the figure
    right=0.986,   # Space from the right of the figure
    top=0.964,     # Space from the top of the figure
    wspace=0.15,    # Width space between subplots
    hspace=0.214     # Height space between subplots
)
plt.savefig(f"results_analysis/figs/performance_all.pdf",
            dpi=60)
plt.savefig(f"results_analysis/figs/performance_all.png",
            dpi=60)
plt.show()
plt.clf()
