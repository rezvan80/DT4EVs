"""
This file is used to run various experiments in different tmux panes each.
"""

import os
import time

# run train_DT.py in a tmux pane for each K and dataset

# batch_size = 64
num_steps_per_iter = 1000
max_iters = 350
num_eval_episodes = 30

counter = 0
for model_type in ["dt","gnn_dt","gnn_in_out_dt"]: #dt, gnn_dt, gnn_in_out_dt
    for action_mask in [True]:
        for K in [10]:
            for batch_size in [128]:
                # "RR_400_000", "optimal_100000", "RR_10_000"
                # "RR_10_000", "RR_10_000", 'RR_400_000' RR_SimpleR_10_000
                for dataset in ["random_100"]:  # optimal_5000, suboptimal_10000
                    for embed_dim in [128]:  # 128, 512
                        #   ' --device cuda:0' + str(counter % 2) + \
                        for n_layer, n_head in [(3, 4)]:  # (3, 1),(3,4)

                            # a10 machine config
                            # command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                            # iepg machine config
                            # command = 'tmux new-session -d \; send-keys "  /home/sorfanoudakis/.conda/envs/dt3/bin/python train_DT.py' + \
                            
                            command = 'tmux new-session -d \; send-keys " /home/sorfanouda/anaconda3/envs/dt/bin/python train_DT.py' + \
                                ' --dataset ' + dataset + \
                                ' --K ' + str(K) + \
                                ' --device cuda:0' + \
                                ' --model_type ' + model_type + \
                                ' --embed_dim ' + str(embed_dim) + \
                                ' --n_layer ' + str(n_layer) + \
                                ' --n_head ' + str(n_head) + \
                                ' --max_iters=' + str(max_iters) + \
                                ' --batch_size=' + str(batch_size) + \
                                ' --num_steps_per_iter=' + str(num_steps_per_iter) + \
                                ' --num_eval_episodes=' + str(num_eval_episodes) + \
                                ' --log_to_wandb True' + \
                                ' --action_masking ' + str(action_mask) + \
                                ' --group_name ' + '"2ndTests_"' + \
                                ' --name ReLu_LastLayer_RTG_K=' + str(K) + \
                                ",action_mask=" + str(action_mask) + \
                                "," + str(model_type) + "," + \
                                'dataset=' + dataset + \
                                ',embed_dim=' + str(embed_dim) + \
                                ',n_layer=' + str(n_layer) +\
                                ',max_iters=' + str(max_iters) + \
                                ',num_steps_per_iter=' + str(num_steps_per_iter) + \
                                ',batch_size=' + str(batch_size) + \
                                ',n_head=' + str(n_head) + \
                                ',num_eval_episodes=' + str(num_eval_episodes) + \
                                ',scale=1' +\
                                '" Enter'
                            os.system(command=command)
                            print(command)
                            # wait for 20 seconds before starting the next experiment
                            time.sleep(15)
                            counter += 1
