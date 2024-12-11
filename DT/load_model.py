'''
This file is used to load the model
'''
import torch
import numpy as np
import yaml
from DT.models.decision_transformer import DecisionTransformer
from DT.models.gnn_In_Out_decision_transformer import GNN_IN_OUT_DecisionTransformer


def load_GNN_IN_OUT_DecisionTransformer_model(model_path, max_ep_len, env, config,  device):

    # load model config yaml file
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    model_path = f"saved_models/{model_path}"
    vars_path = f"{model_path}/vars.yaml"

    vars = yaml.load(open(vars_path), Loader=yaml.FullLoader)

    model = GNN_IN_OUT_DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=vars['K'],
        max_ep_len=max_ep_len,
        hidden_size=vars['embed_dim'],
        n_layer=vars['n_layer'],
        n_head=vars['n_head'],
        n_inner=4*vars['embed_dim'],
        activation_function=vars['activation_function'],
        n_positions=1024,
        resid_pdrop=vars['dropout'],
        attn_pdrop=vars['dropout'],
        action_tanh=True,
        action_masking=vars['action_masking'],
        fx_node_sizes={'ev': 5, 'cs': 4, 'tr': 2, 'env': 6},
        feature_dim=vars['feature_dim'],
        GNN_hidden_dim=vars['GNN_hidden_dim'],
        num_gcn_layers=vars['num_gcn_layers'],
        config=config,
        device=device,
    )
    
    model.load_state_dict(torch.load(f"{model_path}/model.best"))
    
    return model


def load_DT_model(model_path, max_ep_len, env,  device):
    '''
    Load the Decision Transformer model using the model path and device'''

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    load_path = f"saved_models/{model_path}"
    state_mean = np.load(f'{load_path}/state_mean.npy')
    state_std = np.load(f'{load_path}/state_std.npy')

    load_model_path = f"{load_path}/model.best"

    K = int(model_path.split(",")[0].split("=")[1])
    embed_dim = int(model_path.split(",")[1].split("=")[1])
    n_layer = int(model_path.split(",")[2].split("=")[1])
    batch_size = int(model_path.split(",")[5].split("=")[1])
    n_head = int(model_path.split(",")[6].split("=")[1])

    print(
        f"K={K}, embed_dim={embed_dim}, n_layer={n_layer}, batch_size={batch_size}, n_head={n_head}")

    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=embed_dim,
        n_layer=n_layer,
        n_head=n_head,
        n_inner=4*embed_dim,
        activation_function='relu',
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    )

    model.load_state_dict(torch.load(load_model_path))
    model.to(device=device)

    return model, state_mean, state_std
