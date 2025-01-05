# results = {
#     "algorithm": algorithm,
#     "K": K,
#     "dataset": dataset,
#     "seed": seed,
#     "best": np.array(history["best"])[-1],
#     "best_reward": np.array(history["best"]),
#     "eval_reward": np.array(history["test/total_reward"]),
#     "eval_profits": np.array(history["test/total_profits"]),
#     "eval_power_tracker_violation": np.array(history["test/power_tracker_violation"]),
#     "eval_user_satisfaction": np.array(history["test/average_user_satisfaction"]),
#     "opt_reward": np.array(history["opt/total_reward"])[-1],
#     "opt_profits": np.array(history["opt/total_profits"])[-1],
#     "opt_power_tracker_violation": np.array(history["opt/power_tracker_violation"])[-1],
#     "opt_user_satisfaction": np.array(history["opt/average_user_satisfaction"])[-1],
# }

# # Plot the results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils import dataset_info, parse_string_to_list

data = pd.read_csv("./results_analysis/results.csv")


datasets_list = [
    'random_100',
    'random_1000',
    'random_10000',
    'optimal_100',
    'optimal_1000',
    'optimal_10000',
    'bau_100',
    'bau_1000',
    'bau_10000',
]

# filter the data that have:
# data = data[(data["K"] == 2) & (data["dataset"].str.contains("optimal"))]
# data = data[(data["K"] == 10) & (data["dataset"].str.contains("optimal"))]
data = data[(data["K"] == 10)]
dataset_info(data)

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

# print(new_df.head())
# print(new_df.describe())

datasets_list = [
    
    'random_1000',
    'optimal_25_1000',
    'optimal_50_1000',
    'optimal_75_1000',
    'optimal_1000',
]


# from dt to DT
new_df["algorithm"] = new_df["algorithm"].replace("dt", "DT")

# plot the data
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

plt.figure(figsize=(4, 5))
# keep the dataset that are in the datasets_list
new_df = new_df[new_df["dataset"].isin(datasets_list)]
dataset_info(new_df)

# plt.figure(figsize=(4, 5))
sns.lineplot(data=new_df,
                x="epoch",
                y="reward",
                hue="dataset",                
                )

# plt.title(f"K=10",
#           fontsize=17)

# add a horizontal line for the optimal reward
plt.axhline(y=-2405, color='r', linestyle='--',
            label="Oracle")

# create a new legend for the optimal reward and the algorithms
plt.legend(loc='lower right',
            title="Algorithm",
            title_fontsize=15,
            ncol=2,
            columnspacing=0.4,
            fontsize=14.5)
# plt.legend(loc='upper left')

# set x and y labels font size
plt.xlabel("Epoch", fontsize=17)

plt.ylabel("Reward [-]", fontsize=17)

# set xticks and yticks font size
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# put scientific notation in the y axis
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# set xlim
plt.xlim(0, 150)
plt.ylim(-450_000, 10_000)
plt.tight_layout()

plt.savefig(f"results_analysis/figs/mixed_opt_performance.pdf",
            dpi=60)
plt.savefig(f"results_analysis/figs/mixed_opt_performance.png",
            dpi=60)
plt.clf()
