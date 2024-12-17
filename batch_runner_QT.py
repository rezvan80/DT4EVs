'''
This script is used to run the batch of training sciprts for every algorithms evaluated
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
'''
import os
import random

seeds = [10]
config = "PST_V2G_ProfixMax_25.yaml"
eval_replay_path = "./eval_replays/PST_V2G_ProfixMax_25_optimal_25_50/"

# config = "PST_V2G_ProfixMax_250.yaml"
# eval_replay_path = "./eval_replays/PST_V2G_ProfixMax_250_optimal_250_50/"


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

mixed_datasets_list = [
    'bau_25_1000',
    'bau_50_1000',
    'bau_75_1000',
    'optimal_25_1000',
    'optimal_50_1000',
    'optimal_75_1000',
]

num_steps_per_iter = 1000
max_iters = 200
num_eval_episodes = 30
batch_size = 128

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

grad_norm = 15
eta = 0.05
embed_dim = 128

for K in [10]:
    for dataset in datasets_list:
        for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)
            for counter, seed in enumerate(seeds):

                if "250" in config:
                    memory = 24
                    time = 20
                    cpu_cores = 2

                    max_iters = 400
                    batch_size = 64
                    num_steps_per_iter = 3000

                elif "25" in config:

                    if "10000" in dataset:
                        memory = 16
                    else:
                        memory = 8

                    time = 10

                    if K > 10 and K <= 20:
                        time = int(time*1.5)
                    elif K > 20:
                        time = int(time*2)
                    elif K > 30:
                        time = int(time*3)

                    if time > 46:
                        time = 46

                    cpu_cores = 1
                    max_iters = 250
                    batch_size = 128
                    num_steps_per_iter = 1000
                else:
                    raise ValueError("Invalid config file")

                # run_name = f'{algorithm}_run_{counter}_{random.randint(0, 100000)}'
                run_name = f'QT_run_{seed}_K={K}'
                run_name += "_" + str(random.randint(0, 100000))

                command = '''#!/bin/sh
#!/bin/bash
#SBATCH --job-name="EV_Exps"
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

conda activate dt3
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

''' + 'srun python train_QT.py' + \
                    ' --dataset ' + dataset + \
                    ' --K ' + str(K) + \
                    ' --device cuda:0' + \
                    ' --embed_dim ' + str(embed_dim) + \
                    ' --n_layer ' + str(n_layer) + \
                    ' --n_head ' + str(n_head) + \
                    ' --seed ' + str(seed) + \
                    ' --max_iters=' + str(max_iters) + \
                    ' --batch_size=' + str(batch_size) + \
                    ' --num_steps_per_iter=' + str(num_steps_per_iter) + \
                    ' --num_eval_episodes=' + str(num_eval_episodes) + \
                    ' --log_to_wandb True' + \
                    ' --eta ' + str(eta) + \
                    ' --grad_norm ' + str(grad_norm) + \
                    ' --group_name ' + '"QT_tests_"' + \
                    ' --config_file ' + config + \
                    ' --eval_replay_path ' + eval_replay_path + \
                    ' --name ' + str(run_name) + \
                    '' + \
                    '''
        
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
'''

                with open(f'run_tmp.sh', 'w') as f:
                    f.write(command)

                with open(f'./slurm_logs/{run_name}.sh', 'w') as f:
                    f.write(command)

                os.system('sbatch run_tmp.sh')

                # command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_QT.py' + \
                #     ' --dataset ' + dataset + \
                #     ' --K ' + str(K) + \
                #     ' --device cuda:0' + \
                #     ' --embed_dim ' + str(embed_dim) + \
                #     ' --n_layer ' + str(n_layer) + \
                #     ' --n_head ' + str(n_head) + \
                #     ' --seed ' + str(seed) + \
                #     ' --max_iters=' + str(1) + \
                #     ' --batch_size=' + str(batch_size) + \
                #     ' --num_steps_per_iter=' + str(1) + \
                #     ' --eta ' + str(eta) + \
                #     ' --grad_norm ' + str(grad_norm) + \
                #     ' --group_name ' + '"2ndTests_"' + \
                #     ' --config_file ' + config + \
                #     ' --eval_replay_path ' + eval_replay_path + \
                #     ' --name ' + str(run_name) + \
                #     '" Enter'
                    
                # os.system(command=command)
                # print(command)       
                # import time as timer                      
                # timer.sleep(5)