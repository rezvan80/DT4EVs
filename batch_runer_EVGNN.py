'''
This script is used to run the batch of training sciprts for every algorithms evaluated'''
import os
import random

seeds = [10, 20]
# seeds = [30]
algorithms = ["TD3_GNN", "TD3_ActionGNN",
              "TD3", "SAC", "SAC_GNN", "SAC_ActionGNN"]

algorithms = ["TD3_ActionGNN",
              "TD3", "SAC",
              "SAC_ActionGNN"]
dscr_actions = 1

config = "PST_V2G_ProfixMax_25.yaml"

# SBATCH --mail-type=BEGIN,END
# SBATCH --mail-user=s.orfanoudakis@tudelft.nl

# Extra arguments for the python script
fx_dim = 32  # 8
fx_GNN_hidden_dim = 64
fx_num_heads = 2
mlp_hidden_dim = 512  # 256
actor_num_gcn_layers = 3
critic_num_gcn_layers = 3

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

# create a dataframe with index the run name and the other columns the hyperparameters
# import pandas as pd

# #check if file already exists
# if os.path.exists('runs_logger.csv'):
#     #then load it
#     runs_logger = pd.read_csv('runs_logger.csv',index_col=0)
# else:
#     #create a new one
#     runs_logger = pd.DataFrame(columns=['fx_dim',
#                                         'fx_GNN_hidden_dim',
#                                         'fx_num_heads',
#                                         'mlp_hidden_dim',
#                                         'actor_num_gcn_layers',
#                                         'critic_num_gcn_layers',
#                                         'batch_size',
#                                         'memory',
#                                         'total_time',
#                                         'discrete_actions',
#                                         'config',
#                                         'algorithm',
#                                         'seeds',
#                                         'counter',
#                                         'train_hours_done',
#                                         'finished_training'])


for algorithm in algorithms:
    for counter, seed in enumerate(seeds):

        if "1000" in config or "500" in config:
            batch_size = 64
            memory = 128
            time = 47*4

            fx_dim = 64
            fx_GNN_hidden_dim = 128
            mlp_hidden_dim = 512
            actor_num_gcn_layers = 4
            critic_num_gcn_layers = 4

            if "1000" in config:
                actor_num_gcn_layers = 5
                critic_num_gcn_layers = 5

        elif "100" in config:
            batch_size = 128
            memory = 30
            time = 2*47

            fx_dim = 32
            fx_GNN_hidden_dim = 64
            mlp_hidden_dim = 512

        else:
            batch_size = 256
            memory = 16
            time = 47

            fx_dim = 32
            fx_GNN_hidden_dim = 64
            mlp_hidden_dim = 512

        # if 'GNN' not in algorithm:
        #     time = int(time/2)

        run_name = f'{algorithm}_run_{counter}_{random.randint(0, 100000)}'

        # "gpu" "gpu-a100"

        total_time = int(time)
        max_allowed_time = 47
        time_threshold = 10

        # if total_time - time_threshold > max_allowed_time:
        #     save_replay_buffer = " --save_replay_buffer"
        #     # add a new row to the dataframe
        #     # runs_logger.loc[run_name] = [fx_dim,
        #     #                              fx_GNN_hidden_dim,
        #     #                              fx_num_heads,
        #     #                              mlp_hidden_dim,
        #     #                              actor_num_gcn_layers,
        #     #                              critic_num_gcn_layers,
        #     #                              batch_size,
        #     #                              memory,
        #     #                              total_time,
        #     #                              dscr_actions,
        #     #                              config,
        #     #                              algorithm,
        #     #                              seed,
        #     #                              0,
        #     #                              0,
        #     #                              False]
        # else:
        #     save_replay_buffer = ""

        time = max_allowed_time
        command = '''#!/bin/sh

#!/bin/bash
#SBATCH --job-name="Cl_DT"
#SBATCH --partition=gpu
''' + \
            f'#SBATCH --time={time}:00:00' + \
            '''
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=1
''' + \
            f'#SBATCH --mem-per-cpu={memory}G' + \
            '''
#SBATCH --account=research-eemcs-ese

''' + \
            f'#SBATCH --output=./slurm_logs/{run_name}.out' + \
            '''
''' + \
            f'#SBATCH --error=./slurm_logs/{run_name}.err' + \
            '''

module load 2024r1 openmpi miniconda3 py-pip

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate dt
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

''' + \
            f'srun python train_RL_GNN.py --seed {seed}' +\
            f' --fx_dim {fx_dim} --fx_GNN_hidden_dim {fx_GNN_hidden_dim} --fx_num_heads {fx_num_heads} --mlp_hidden_dim {mlp_hidden_dim}' + \
            f' --policy {algorithm} --name {run_name} --project_name DT4EVs --config {config} --batch_size {batch_size} --discrete_actions {dscr_actions}' + \
            f' --actor_num_gcn_layers {actor_num_gcn_layers} --critic_num_gcn_layers {critic_num_gcn_layers}' + \
            f' --time_limit_hours {time-1}' + \
            '''

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
'''

        with open(f'run_tmp.sh', 'w') as f:
            f.write(command)

        with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
            f.write(command)

        os.system('sbatch run_tmp.sh')

        # save the dataframe
        # runs_logger.to_csv('runs_logger.csv')
