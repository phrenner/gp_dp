import importlib
import torch

def dynamic_params(cfg):

    cfg["no_samples"] = 64
    cfg["num_epochs"] = 2999  #fixed number of epsiodes for reproducibility
    cfg["warm_start"] = False #warm start for gp fitting
    cfg["use_fixed_noise"] = True #use fixed noise for likelyhood
    cfg["drop_non_converged"] = True #drop non converged points from the training set
    cfg["CHECKPOINT_INTERVAL"] = 1
    cfg["num_iterations"] = 2999

    cfg["BAL"] = {"enabled" : cfg.get("ENABLE_BAL", True)}
    cfg["BAL"]["epoch_freq"] = 30
    cfg["BAL"]["n_bal_sims"] = 64
    cfg["BAL"]["max_points"] = 100000
    cfg["BAL"]["bal_rho"] = 1
    cfg["BAL"]["bal_beta"] = 1
    
    cfg["torch_optim"] = {}
    cfg["torch_optim"]["LR_vf"] = 1e-2
    cfg["torch_optim"]["LR_pol"] = 1e-2

    cfg["torch_optim"]["iter_per_cycle_vf"] = 10000
    cfg["torch_optim"]["relative_ll_change_tol_vf"] = 0
    cfg["torch_optim"]["relative_ll_grad_change_tol_vf"] = 0
    cfg["torch_optim"]["relative_error_tol_vf"] = 0
    cfg["torch_optim"]["parameter_change_tol_vf"] = 1e-3

    cfg["torch_optim"]["iter_per_cycle_pol"] = 10000
    cfg["torch_optim"]["relative_ll_change_tol_pol"] = 0
    cfg["torch_optim"]["relative_ll_grad_change_tol_pol"] = 0
    cfg["torch_optim"]["relative_error_tol_pol"] = 0
    cfg["torch_optim"]["parameter_change_tol_pol"] = 5e-4

    cfg["scipyopt"] = {"maxiter": 400}
    cfg["scipyopt"]["no_restarts"] = 3
    cfg["scipyopt"]["method"] = "SLSQP"
    cfg["scipyopt"]["tol"] = 1e-6

    ### Define constants
    cfg["model"] = {"params":{}}
    cfg["model"]["params"]["n_types"] = 2
    cfg["model"]["params"]["beta"] = 0.9
    cfg["model"]["params"]["upper_trans"] = 1.
    cfg["model"]["params"]["lower_trans"] = 0.
    cfg["model"]["params"]["upper_shock"] = 0.35
    cfg["model"]["params"]["lower_shock"] = 0.1
    cfg["model"]["params"]["sigma"] = 0.5
    cfg["model"]["params"]["reg_c"] = 0.0001
    cfg["model"]["params"]["pen_opt_vf"] = 50.
    cfg["model"]["params"]["pen_vf"] = 10.0
    if cfg.get("ENABLE_HOWARD", True):
        cfg["model"]["params"]["n_Howard_steps"] = 50   
    else:
        cfg["model"]["params"]["n_Howard_steps"] = 0  

    model = importlib.import_module(
            cfg["MODEL_NAME"] + ".Model"
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
    reg_c = cfg["model"]["params"]["reg_c"]
    sigma = cfg["model"]["params"]["sigma"]
    cfg["model"]["params"]["trans_mat"] = trans_mat
    cfg["model"]["params"]["trans_mat_inv"] = torch.inverse(cfg["model"]["params"]["trans_mat"])
    cfg["model"]["params"]["trans_mat_inv_left"] = torch.inverse((torch.eye(n_types) - beta * trans_mat))
    upper_trans = cfg["model"]["params"]["upper_trans"]
    lower_trans = cfg["model"]["params"]["lower_trans"]
    cfg["model"]["params"]["discrete_state_dim"] = n_types

    upper_w = model.utility_ind(upper_trans, reg_c, sigma)/(1-beta)
    cfg["model"]["params"]["upper_w"] = upper_w * torch.ones(n_types)

    lower_w = model.utility_ind(lower_trans, reg_c, sigma)/(1-beta)
    cfg["model"]["params"]["lower_w"] = lower_w * torch.ones(n_types)

    cfg["model"]["params"]["max_points"] = 0.5 * torch.unsqueeze(upper_w + cfg["model"]["params"]["lower_w"],0) * torch.ones([n_types,n_types])

    cfg["model"]["params"]["lower_V"] =  torch.tensor(-upper_trans + lower_shock)
    cfg["model"]["params"]["upper_V"] =  torch.tensor(cfg["model"]["params"]["upper_shock"])

    cfg["model"]["params"]["autarky_state"] =  torch.tensor([[8.09419, 9.66737]])

    cfg["model"]["params"]["min_GP_value"] = cfg["model"]["params"]["lower_V"] - 1.0
    cfg["model"]["params"]["GP_offset"] = cfg["model"]["params"]["min_GP_value"]


    return cfg
