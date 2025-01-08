import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from res_utils import dataset_info, parse_string_to_list

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

K = 2

name = "mixed_opt_performance"
datasets_list = [
    'random_1000',
    'optimal_25_1000',
    'optimal_50_1000',
    'optimal_75_1000',
    'optimal_1000',
]

hue_order = [
    'Random 100%',
    'Optimal 25% + Random 75%',
    'Optimal 50% + Random 50%',
    'Optimal 75% + Random 25%',
    'Optimal 100%',
]

# name = "mixed_bau_performance"
# datasets_list = [
#         'random_1000',
#         'bau_25_1000',
#         'bau_50_1000',
#         'bau_75_1000',
#         'bau_1000',
#     ]

# hue_order = [
#     'Random 100%',
#     'BAU 25% + Random 75%',
#     'BAU 50% + Random 50%',
#     'BAU 75% + Random 25%',
#     'BAU 100%',
#     ]

print(axs)
for index in range(2):

    if index == 0:
        K = 2
    else:
        K = 10

    data = pd.read_csv("./results_analysis/results.csv")
    data = data[(data["K"] == K)]

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

    # from dt to DT
    new_df["algorithm"] = new_df["algorithm"].replace("dt", "DT")

    # plot the data
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'

    # plt.figure(figsize=(6,4))
    sns.set_context("paper", font_scale=1.5)
    # make subplots

    # keep the dataset that are in the datasets_list
    new_df = new_df[new_df["dataset"].isin(datasets_list)]

    change_dataset = {
        'random_1000': 'Random 100%',
        'optimal_25_1000': 'Optimal 25% + Random 75%',
        'optimal_50_1000': 'Optimal 50% + Random 50%',
        'optimal_75_1000': 'Optimal 75% + Random 25%',
        'optimal_1000': 'Optimal 100%',
        'bau_25_1000': 'BAU 25% + Random 75%',
        'bau_50_1000': 'BAU 50% + Random 50%',
        'bau_75_1000': 'BAU 75% + Random 25%',
        'bau_1000': 'BAU 100%',
    }

    new_df["dataset"] = new_df["dataset"].replace(change_dataset)
    print(new_df["dataset"].value_counts())

    print(f' Data ready to plot')
    # plt.figure(figsize=(4, 5))
    sns.lineplot(data=new_df,
                 x="epoch",
                 y="reward",
                 hue="dataset",
                 hue_order=hue_order,
                 ax=axs[index],
                 )

    # plt.title(f"K=10",
    #           fontsize=17)
    # Add a horizontal line for the optimal reward
    # put title
    if index == 0:
        axs[index].set_title("K=2",
                             fontsize=17)
    else:
        axs[index].set_title("K=10",
                             fontsize=17)

    axs[index].axhline(y=-2405, color='r', linestyle='--', label="Oracle")
    
    # Create a new legend for the optimal reward and the algorithms
    # axs[index].legend(loc='lower right',
    #                   title="Dataset",
    #                   title_fontsize=15,
    #                   ncol=2,
    #                   columnspacing=0.4,
    #                   fontsize=14.5
    #                   )

    # move the legend under the plot
    axs[index].legend(loc='upper center',
                      title="Dataset",
                      title_fontsize=15,
                      bbox_to_anchor=(1, -0.15),
                      shadow=True,
                      fontsize=15,
                      ncol=3)

    # show grid lines
    axs[index].grid(True)

    if index == 1:
        # remove the legend
        axs[index].get_legend().remove()

    # Set x and y labels font size
    axs[index].set_xlabel("Epoch", fontsize=17)
    axs[index].set_ylabel("Reward [-]", fontsize=17)

    # Set xticks and yticks font size
    axs[index].tick_params(axis='x', labelsize=15)
    axs[index].tick_params(axis='y', labelsize=15)

    # Put scientific notation in the y-axis
    axs[index].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    # Set xlim and ylim
    axs[index].set_xlim(0, 150)
    axs[index].set_ylim(-350_000, 10_000)
    print(f'Plot ready')

# Adjust layout
# fig.tight_layout()
plt.subplots_adjust(
    left=0.074,    # Space from the left of the figure
    bottom=0.288,   # Space from the bottom of the figure
    right=0.983,   # Space from the right of the figure
    top=0.952,     # Space from the top of the figure
    wspace=0.2,    # Width space between subplots
    hspace=0.2     # Height space between subplots
)
# plt.show()
plt.savefig(f"results_analysis/figs/{name}.pdf",
            dpi=60)
plt.savefig(f"results_analysis/figs/{name}.png",
            dpi=60)
plt.show()
plt.clf()
