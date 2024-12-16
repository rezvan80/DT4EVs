import pickle
import gzip

datasets = [
    'PST_V2G_ProfixMax_25_optimal_25_2000.pkl',
]


for dataset_path in datasets:
    print(f"=========================")
    print(f"Loading dataset from {dataset_path}")
    dataset_path = f"./trajectories/{dataset_path}"
    
    if "gz" in dataset_path:
        with gzip.open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    else:
        with open(dataset_path, 'rb') as f:
            trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories)} trajectories")