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

def dataset_info(data):
    print("=====================================")
    print(f' data shape: {data.shape}')
    print(data["dataset"].value_counts()) 
    print(data["K"].value_counts())
    print(data["algorithm"].value_counts())
    print(data["seed"].value_counts())
    print("=====================================")


data = pd.read_csv("./results_analysis/results.csv")
dataset_info(data)

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
data = data[(data["K"] == 10) & (data["dataset"].str.contains("optimal"))]
dataset_info(data)

# For every row in the data create a new dataframe with epoch as the index and the reward as the value, keep also, the seed, algorithm and dataset

new_df = pd.DataFrame()
for i, row in data.iterrows():
    # print(row)
    # parse the string to a list
    rewards = row["eval_reward"].replace("[", "").replace("]", "").replace("\n", " ")
    rewards = rewards.replace("'", " ")
    rewards = rewards.replace("  ", " ")
    rewards = rewards.replace("  ", " ")
    rewards = rewards.replace("  ", " ")
    rewards = rewards.replace("  ", " ")
    # remove the last space
    if rewards[-1] == " ":
        rewards = rewards[:-1]
    rewards = rewards.replace("  ", " ").split(" ")
    # print(rewards)
    rewards = np.array(rewards).astype(float)
    
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
    
print(new_df.head())    
print(new_df.describe())
    

# plot the data
sns.set_theme(style="whitegrid")
# plt.figure(figsize=(12, 6))
sns.relplot(data=new_df,
             x="epoch",
             y="reward",
             col="dataset",
             kind="line",
             hue="algorithm")

plt.show()

sns.relplot(data=new_df,
                x="epoch",
                y="reward",
                col="algorithm",
                kind="line",
                hue="dataset")
plt.show()