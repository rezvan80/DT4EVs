'''
This script is used to run the batch of training sciprts for every algorithms evaluated
srun --mpi=pmix --job-name=interactive-gpu --partition=gpu --gres=gpu:1 --qos=normal --time=01:00:00 --mem-per-cpu=4096 --pty /bin/bash -il
'''
import os
import random

seeds = [10]
config = "PST_V2G_ProfixMax_25.yaml"


num_steps_per_iter = 1000
max_iters = 200
num_eval_episodes = 30
batch_size = 128

# if directory does not exist, create it
if not os.path.exists('./slurm_logs'):
    os.makedirs('./slurm_logs')

for grad_norm in [25]:
    for K in [2]:
        for eta in [0.001]:
            for dataset in ["optimal_2000"]:  # optimal_2000, random_100
                for embed_dim in [128]:  # 128, 512
                    for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)
                        for counter, seed in enumerate(seeds):

                            memory = 10
                            time = 8

                            # run_name = f'{algorithm}_run_{counter}_{random.randint(0, 100000)}'
                            run_name = f'QT_run_{seed}_K={K}_grad_norm={grad_norm}_eta={eta}_embed_dim={embed_dim}_n_layer={n_layer}_n_head={n_head}'
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
