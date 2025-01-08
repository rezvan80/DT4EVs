'''Find the best reward for each dataset and algorithm together with the corresponding epoch'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from res_utils import dataset_info, parse_string_to_list


data = pd.read_csv("./results_analysis/max_rewards.csv")

datasets_list = [
    'optimal_1000',
    'bau_1000',
    'random_1000',
]

print(data.head())

# keep rows with aalgoritm = gnn_act_emb
data = data[(data["algorithm"] == "gnn_act_emb")]

# keep rows with dataset in datasets_list
data = data[data["dataset"].isin(datasets_list)]
# multiply max_reward by 10_0000
data["max_reward"] = data["max_reward"]*10_0000

# remove rows with K = 100
data = data[data["K"] != 100]
data = data[data["K"] != 50]

print(data.head())

plt.figure(figsize=(6, 4))
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

sns.lineplot(x="K",
             y="max_reward",
             hue="dataset",
             hue_order=datasets_list,
             data=data,
             ci=None)

plt.scatter(data["K"],
            data["max_reward"],
            color='black',
            zorder=10)

plt.xlabel("Context Length (K)", fontsize=14)
# add a horizontal line for the optimal reward
plt.axhline(y=-2405,
            color='r',
            linestyle='--',
            label="Oracle")

plt.xticks(data["K"].unique(), fontsize=13)
# ticklabel_format(
# style='sci', axis='y', scilimits=(0, 0))
plt.ylabel("Max Reward [-]", fontsize=14)
# make scientific notation the y axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlim(1.5, 45.5)

plt.legend([
    'Optimal',
    'BaU',
    'Random',
],
    loc='lower center',
    title="Dataset",
    title_fontsize=14,
    bbox_to_anchor=(0.5, 1.03),
    ncol=3,
    fontsize=13
)

# plt.show()
plt.savefig("./results_analysis/figs/max_rewards_K.png",
            dpi=300,
            bbox_inches='tight')

plt.savefig("./results_analysis/figs/max_rewards_K.pdf",
            bbox_inches='tight')
