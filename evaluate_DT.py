'''
This file is used to evaluate the decision transformer model
'''
import gymnasium as gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import yaml

from DT.evaluation.evaluate_episodes import evaluate_episode_rtg
from DT.models.decision_transformer import DecisionTransformer

from ev2gym.models import ev2gym_env

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='ev-city')
parser.add_argument('--name', type=str, default='ev')
parser.add_argument('--group_name', type=str, default='')
parser.add_argument('--seed', type=int, default=42)

# medium, medium-replay, medium-expert, expert
parser.add_argument('--dataset', type=str, default='RR_20')
# normal for standard setting, delayed for sparse
parser.add_argument('--mode', type=str, default='normal')
parser.add_argument('--K', type=int, default=24)
parser.add_argument('--pct_traj', type=float, default=1.)
parser.add_argument('--batch_size', type=int, default=128)
# dt for decision transformer, bc for behavior cloning
parser.add_argument('--model_type', type=str, default='dt')
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--n_layer', type=int, default=3)
parser.add_argument('--n_head', type=int, default=1)
parser.add_argument('--activation_function', type=str, default='relu')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--num_eval_episodes', type=int, default=2)
parser.add_argument('--max_iters', type=int, default=500)
parser.add_argument('--num_steps_per_iter', type=int, default=1000)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
parser.add_argument('--config_file', type=str, default="PublicPST.yaml")
variant = vars(parser.parse_args())

device = torch.device(variant['device'])
if device.type == 'cuda' and not torch.cuda.is_available():
    device = torch.device('cpu')
log_to_wandb = variant.get('log_to_wandb', False)

env_name, dataset = variant['env'], variant['dataset']
model_type = variant['model_type']
# group_name = f'{exp_prefix}-{env_name}'

run_name = variant['name']

exp_prefix = f'{run_name}-{dataset}-{random.randint(int(1e5), int(1e6) - 1)}'

# # seed everything
# seed = variant.get('seed', 0)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

env = "ev_city-v1"
scale = 1
env_targets = [0]  # evaluation conditioning targets

if model_type == 'bc':
    # since BC ignores target, no need for different evaluations
    env_targets = env_targets[:1]

config = yaml.load(open(variant.get("config_file"), 'r'),
                   Loader=yaml.FullLoader)

number_of_charging_stations = config["number_of_charging_stations"]
n_transformers = config["number_of_transformers"]
steps = config["simulation_length"]

env = ev2gym_env.EV2Gym(config_file=variant["config_file"],
                        generate_rnd_game=True,)

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
max_ep_len = steps

# exp_prefix = "K=12,embed_dim=128,n_layer=3,max_iters=2000,num_steps_per_iter=20000,batch_size=128,n_head=4,num_eval_episodes=30,scale=1-RR_10_000-231028"
exp_prefix = "Not_RTG_K=12,embed_dim=128,n_layer=3,max_iters=2000,num_steps_per_iter=1000,batch_size=128,n_head=4,num_eval_episodes=30,scale=1-RR_SimpleR_10_000-877286"
K = 12 #int(exp_prefix.split(",")[0].split("=")[1])
embed_dim = 128 #int(exp_prefix.split(",")[1].split("=")[1])
n_layer =3 # int(exp_prefix.split(",")[2].split("=")[1])
batch_size = 128 #int(exp_prefix.split(",")[5].split("=")[1])
n_head = 4 # int(exp_prefix.split(",")[6].split("=")[1])

print(f"K={K}, embed_dim={embed_dim}, n_layer={n_layer}, batch_size={batch_size}, n_head={n_head}")

model = DecisionTransformer(
    state_dim=state_dim,
    act_dim=act_dim,
    max_length=K,
    max_ep_len=max_ep_len,
    hidden_size=embed_dim,
    n_layer=n_layer,
    n_head=n_head,
    n_inner=4*embed_dim,
    activation_function=variant['activation_function'],
    resid_pdrop=variant['dropout'],
    attn_pdrop=variant['dropout'],
)


load_path = f"saved_models/{exp_prefix}"
model_path = f"{load_path}/model.best"
model.load_state_dict(torch.load(model_path))
model.eval()

target = 0
num_eval_episodes = variant['num_eval_episodes']

state_mean = np.load(f'{load_path}/state_mean.npy')
state_std = np.load(f'{load_path}/state_std.npy')

stats = evaluate_episode_rtg(
    exp_prefix,
    state_dim,
    act_dim,
    model,
    model_type='dt',
    max_ep_len=max_ep_len,
    scale=scale,
    target_return=target/scale,
    # mode=mode,
    state_mean=state_mean,
    state_std=state_std,
    device=device,
    n_test_episodes=500,
    config_file=variant["config_file"],
)

print(stats)