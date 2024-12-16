import pickle
import gzip
import os
import numpy as np

# datasets = [
#     'PST_V2G_ProfixMax_25_optimal_25_2000.pkl',
# ]

# check all files in the directory
datasets = os.listdir("./trajectories/")
datasets = [dataset for dataset in datasets if dataset.endswith(".pkl.gz")]
for dataset_path in datasets:
    print('=' * 50)
    # print(f"Loading dataset from {dataset_path}")
    dataset_path = f"./trajectories/{dataset_path}"
    
    if "gz" in dataset_path:
        with gzip.open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    
    env_size = dataset_path.split('/')[-1].split('_')[3]
    dataset_type = dataset_path.split('/')[-1].split('_')[4]
    
    print(f"Environment {env_size} CS | Dataset type: {dataset_type}")
    traj_lens, returns = [], []
    for path in trajectories:
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens = np.array(traj_lens)
    num_timesteps = sum(traj_lens)

    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)
    