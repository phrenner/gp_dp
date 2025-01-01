import importlib
import torch

def set_params(cfg):

    cfg["warm_start"] = False #warm start for gp fitting
    cfg["use_fixed_noise"] = True #use fixed noise for likelyhood
    cfg["drop_non_converged"] = True #drop non converged points from the training set

    cfg["torch_optim"]["relative_ll_change_tol_vf"] = 0
    cfg["torch_optim"]["relative_ll_grad_change_tol_vf"] = 0
    cfg["torch_optim"]["relative_error_tol_vf"] = 0
    cfg["torch_optim"]["parameter_change_tol_vf"] = 1e-3

    cfg["torch_optim"]["relative_ll_change_tol_pol"] = 0
    cfg["torch_optim"]["relative_ll_grad_change_tol_pol"] = 0
    cfg["torch_optim"]["relative_error_tol_pol"] = 0
    cfg["torch_optim"]["parameter_change_tol_pol"] = 1e-3

    ### Define constants
    cfg["model"]["params"]["n_types"] = 10
    cfg["model"]["params"]["beta"] = 0.9
    cfg["model"]["params"]["upper_trans"] = 1.
    cfg["model"]["params"]["lower_trans"] = 0.
    cfg["model"]["params"]["upper_shock"] = 0.35
    cfg["model"]["params"]["lower_shock"] = 0.1
    cfg["model"]["params"]["sigma"] = 0.5
    cfg["model"]["params"]["reg_c"] = 0.0001
    cfg["model"]["params"]["pen_opt_vf"] = 50.
    cfg["model"]["params"]["pen_vf"] = 10.0
    cfg["model"]["params"]["n_Howard_steps"] = 0    

    ### Compute derived parameters
    model = importlib.import_module(
            cfg["model"]["MODEL_NAME"] + ".Model"
        )
    upper_shock = cfg["model"]["params"]["upper_shock"]
    lower_shock = cfg["model"]["params"]["lower_shock"]
    n_types = cfg["model"]["params"]["n_types"]
    shoch_vec = torch.linspace(lower_shock,upper_shock,n_types)
    cfg["model"]["params"]["shock_vec"] = shoch_vec
    beta = cfg["model"]["params"]["beta"]

    trans_mat = torch.zeros([n_types,n_types])
    upper_triangular_mat = 0.1*torch.cat([torch.cat([torch.zeros([n_types-1,1]),torch.eye(n_types-1)],1),torch.zeros([1,n_types])],0)
    trans_mat += upper_triangular_mat #upper diagonal set to 0.1
    trans_mat += torch.transpose(upper_triangular_mat,0,1) #lower diagonal set to 0.1
    tmp_mat = 0.8*torch.eye(n_types)
    tmp_mat[0,0] = 0.9
    tmp_mat[-1,-1] = 0.9
    trans_mat = trans_mat + tmp_mat #set diagonal to 0.8 except for the corner pts
    cfg["model"]["params"]["trans_mat"] = trans_mat
    cfg["model"]["params"]["trans_mat_inv"] = torch.inverse(cfg["model"]["params"]["trans_mat"])
    upper_trans = cfg["model"]["params"]["upper_trans"]
    lower_trans = cfg["model"]["params"]["lower_trans"]
    cfg["model"]["params"]["discrete_state_dim"] = n_types

    reg_c = cfg["model"]["params"]["reg_c"]
    sigma = cfg["model"]["params"]["sigma"]
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
    cfg["model"]["params"]["GP_offset"] = (cfg["model"]["params"]["lower_V"] - cfg["model"]["params"]["max_penalty"]) #/(1-beta) #translate the gp by this amount
    cfg["model"]["params"]["GP_offset_feas"] = torch.tensor(-1.0)

    if not "no_samples" in cfg:
        cfg["no_samples"] = 16

    # used for expectations
    if cfg["model"]["EXPECT_OP"]["name"] != "SinglePoint":
        cfg["model"]["EXPECT_OP"]["config"]["mc_std"] = [
            cfg["model"]["EXPECT_OP"]["config"]["sigma"]
        ] * cfg["model"]["params"]["n_types"]

    return cfg

