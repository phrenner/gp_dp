import importlib
import torch

def dynamic_params(cfg):

    cfg["no_samples"] = 32 - 1
    cfg["num_epochs"] = 10000    
    cfg["warm_start"] = False #warm start for gp fitting
    cfg["use_fixed_noise"] = True #use fixed noise for likelyhood
    cfg["drop_non_converged"] = True #drop non converged points from the training set
    cfg["CHECKPOINT_INTERVAL"] = 1
    cfg["iter_per_cycle"] = 10000

    cfg["BAL"] = {"enabled" : cfg.get("ENABLE_BAL", False)}
    cfg["BAL"]["epoch_freq"] = 30
    cfg["BAL"]["max_points"] = 100000
    cfg["BAL"]["n_bal_sims"] = 64
    cfg["BAL"]["bal_rho"] = 1
    cfg["BAL"]["bal_beta"] = 2

    cfg["torch_optim"] = {}
    cfg["torch_optim"]["LR"] = 1e-3

    cfg["torch_optim"]["iter_per_cycle_vf"] = 10000
    cfg["torch_optim"]["relative_ll_change_tol_vf"] = 0
    cfg["torch_optim"]["relative_ll_grad_change_tol_vf"] = 0
    cfg["torch_optim"]["relative_error_tol_vf"] = 0
    cfg["torch_optim"]["parameter_change_tol_vf"] = 1e-3

    cfg["torch_optim"]["iter_per_cycle_pol"] = 10000
    cfg["torch_optim"]["relative_ll_change_tol_pol"] = 0
    cfg["torch_optim"]["relative_ll_grad_change_tol_pol"] = 0
    cfg["torch_optim"]["relative_error_tol_pol"] = 0
    cfg["torch_optim"]["parameter_change_tol_pol"] = 1e-3

    cfg["scipyopt"] = {"maxiter": 400}
    if "N_RESTARTS" in cfg:
        cfg["scipyopt"]["no_restarts"] = cfg["N_RESTARTS"]
    else:    
        cfg["scipyopt"]["no_restarts"] = 3
    cfg["scipyopt"]["method"] = "SLSQP"
    cfg["scipyopt"]["method_gpopt"] = "L-BFGS-B"
    cfg["scipyopt"]["tol"] = 1e-4
    cfg["scipyopt"]["tol_gpopt"] = 1e-4

    ### Define constants
    cfg["model"] = {"params":{}}
    cfg["model"]["params"]["n_types"] = 4
    cfg["model"]["params"]["beta"] = 0.99
    cfg["model"]["params"]["upper_trans"] = 2.
    cfg["model"]["params"]["lower_trans"] = 0.
    cfg["model"]["params"]["upper_shock"] = 1.35
    cfg["model"]["params"]["lower_shock"] = 0.65
    cfg["model"]["params"]["sigma"] = 0.5
    cfg["model"]["params"]["reg_c"] = 0.0001
    cfg["model"]["params"]["pen_opt_vf"] = 100.
    cfg["model"]["params"]["pen_vf"] = 10.0
    if cfg.get("ENABLE_HOWARD", True):
        cfg["model"]["params"]["n_Howard_steps"] = 100   
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

    upper_shock_length = 12
    lower_shock_length = 6
    alpha_param = 0.75
    trans_mat = torch.tensor([[alpha_param * (1 - 1/lower_shock_length),       (1 - alpha_param) * (1 - 1/lower_shock_length), alpha_param/lower_shock_length,                 (1 - alpha_param)/lower_shock_length],
                              [(1 - alpha_param) * (1 - 1/lower_shock_length), alpha_param * (1 - 1/lower_shock_length),       (1-alpha_param)/(lower_shock_length),           alpha_param/(lower_shock_length)],
                              [(alpha_param)/upper_shock_length,               (1 - alpha_param)/upper_shock_length,           (alpha_param) * (1 - 1/upper_shock_length),     (1 - alpha_param) * (1 - 1/upper_shock_length)],
                              [(1 - alpha_param)/upper_shock_length,           (alpha_param)/upper_shock_length ,              (1 - alpha_param) * (1 - 1/upper_shock_length), (alpha_param) * (1 - 1/upper_shock_length)]])    

    cfg["model"]["params"]["autarky_state"] =  torch.tensor([[204.868, 205.118, 206.262, 206.477]])

    reg_c = cfg["model"]["params"]["reg_c"]
    sigma = cfg["model"]["params"]["sigma"]
    cfg["model"]["params"]["trans_mat"] = trans_mat
    cfg["model"]["params"]["trans_mat_inv"] = torch.inverse(cfg["model"]["params"]["trans_mat"])
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

    cfg["model"]["params"]["min_GP_value"] = cfg["model"]["params"]["lower_V"] - 0.0
    cfg["model"]["params"]["GP_offset"] = (cfg["model"]["params"]["min_GP_value"]) #translate the gp by this amount

    return cfg
