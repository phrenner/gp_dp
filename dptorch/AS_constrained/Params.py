import importlib
import torch

def dynamic_params(cfg):
    """Dynamic parameter transforms"""
    model = importlib.import_module(
            cfg["model"]["MODEL_NAME"] + ".Model"
        )
    upper_shock = cfg["model"]["params"]["upper_shock"]
    lower_shock = cfg["model"]["params"]["lower_shock"]
    n_types = cfg["model"]["params"]["n_types"]
    shoch_vec = torch.linspace(lower_shock,upper_shock,n_types)
    cfg["model"]["params"]["shock_vec"] = shoch_vec
    beta = cfg["model"]["params"]["beta"]

    upper_shock_length = cfg["model"]["params"]["upper_shock_length"]
    lower_shock_length = cfg["model"]["params"]["lower_shock_length"]
    trans_mat = torch.tensor([[1 - 1/lower_shock_length, 1/lower_shock_length],[1/upper_shock_length, 1 - 1/upper_shock_length]])
    cfg["model"]["params"]["stationary_distribution"] = torch.tensor(
        [(1/upper_shock_length)/(1/upper_shock_length + 1/lower_shock_length),(1/lower_shock_length)/(1/upper_shock_length + 1/lower_shock_length)])
    cfg["model"]["params"]["expected_shock"] = torch.sum(cfg["model"]["params"]["stationary_distribution"]*shoch_vec)
    L,V = torch.linalg.eig(trans_mat)
    reg_c = cfg["model"]["params"]["reg_c"]
    sigma = cfg["model"]["params"]["sigma"]
    long_run = torch.unsqueeze(1/(1 - beta * L.real),-1)
    right_side = torch.matmul(torch.linalg.inv(V.real),torch.unsqueeze(model.utility_ind(shoch_vec, reg_c, sigma),-1))
    reseveration_util_vec = torch.ones(n_types)
    for indxt in range(n_types):
        reseveration_util_vec[indxt] = torch.matmul((V.real)[indxt:indxt+1,:],right_side*long_run)[0]
    cfg["model"]["params"]["reseveration_util_vec"] = (reseveration_util_vec)
    cfg["model"]["params"]["trans_mat"] = trans_mat
    cfg["model"]["params"]["trans_mat_inv"] = torch.inverse(cfg["model"]["params"]["trans_mat"])
    upper_trans = cfg["model"]["params"]["upper_trans"]
    lower_trans = cfg["model"]["params"]["lower_trans"]
    cfg["model"]["params"]["discrete_state_dim"] = n_types

    upper_w = model.utility_ind(upper_trans, reg_c, sigma)/(1-beta)
    cfg["model"]["params"]["upper_w"] = upper_w * torch.ones(n_types)

    lower_w = model.utility_ind(lower_trans, reg_c, sigma)/(1-beta)
    cfg["model"]["params"]["lower_w"] = lower_w * torch.ones(n_types)

    # cfg["model"]["params"]["lower_w"] = reseveration_util_vec
    cfg["model"]["params"]["max_points"] = 0.5 * torch.unsqueeze(upper_w + cfg["model"]["params"]["lower_w"],0) * torch.ones([n_types,n_types])

    cfg["model"]["params"]["lower_V"] =  torch.tensor(-upper_trans + lower_shock)#/(1-beta)
    cfg["model"]["params"]["upper_V"] =  torch.tensor(cfg["model"]["params"]["upper_shock"])#/(1-beta)
    # cfg["model"]["params"]["upper_V"] =  torch.tensor(cfg["model"]["params"]["upper_trans"])#/(1-beta)

    cfg["model"]["params"]["max_penalty"] = 1.0
    cfg["model"]["params"]["GP_offset"] = cfg["model"]["params"]["lower_V"] - cfg["model"]["params"]["max_penalty"] #/(1-beta) #translate the gp by this amount

    if not "no_samples" in cfg:
        cfg["no_samples"] = 16

    # used for expectations
    if cfg["model"]["EXPECT_OP"]["name"] != "SinglePoint":
        cfg["model"]["EXPECT_OP"]["config"]["mc_std"] = [
            cfg["model"]["EXPECT_OP"]["config"]["sigma"]
        ] * cfg["model"]["params"]["n_types"]

    return cfg
