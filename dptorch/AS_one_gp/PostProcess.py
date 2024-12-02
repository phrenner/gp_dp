from matplotlib import pyplot as plt
import torch
import gpytorch
import matplotlib
import importlib
import logging
import pandas as pd
import numpy as np
from Utils import NonConvergedError

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

def grad_V(m,d,state):
    """Caclulate the gradient wrt. control of the expectation of V"""
    V_fun = m.M[d][0].eval()
    inputs = state.clone().detach().requires_grad_(True)
    outputs = V_fun(inputs).mean
    dummy_out = torch.sum(outputs)
    dummy_out.backward()
    return inputs.grad

def process(m, cfg):

    n_types = m.cfg["model"]["params"]["discrete_state_dim"]
    lower_V = m.cfg["model"]["params"]["lower_V"]
    if m.epoch > m.cfg["n_feas_set_it"]:
        gp_offset = m.cfg["model"]["params"]["GP_offset"]
    else:
        gp_offset = torch.zeros(n_types)

    sigma_util = m.cfg["model"]["params"]["sigma"]
    beta = m.cfg["model"]["params"]["beta"]

    # deleting the error file
    f = open("V_func_error.txt", 'w')
    n_plot_pts = 4000

    lower_w = m.cfg["model"]["params"]["lower_w"]
    upper_w = m.cfg["model"]["params"]["upper_w"]
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")  
    top_out = "# "
    for key, val in m.S.items():
        top_out = top_out + key + " "

    top_out_with_ds = top_out + "DS VF "
    top_out = top_out + "VF "


    file_out = open(f"V_raw_{m.epoch}.txt", 'w')
    file_out.write(top_out_with_ds + "\n")
    file_out.close()
    file_out = open(f"V_raw_{m.epoch}.txt", 'a')

    pts_out = torch.cat(
        (
            (m.state_sample),
            torch.unsqueeze(m.V_sample , dim=-1),
            m.policy_sample
        ), dim=1)
    np.savetxt(
        file_out,
        pts_out.numpy(),
        fmt='%.18e',
        delimiter=' ',
        newline='\n',
        header='',
        footer='',
        comments='# ',
        encoding=None)

    file_out.close()

    for param_name, param in m.M[0][0].named_parameters():
        print(f'Parameter name: {param_name:42} value = {param}')


def compare(m1, m2, cfg):
    """Compare two GP-s"""

    n_test_pts = 1000
    state_sample,dummy = m1.sample(n_test_pts)
    state_sample = state_sample[:,:-1]
    n_types = m2.cfg["model"]["params"]["discrete_state_dim"]

    error_vec = np.zeros((1, 2))

    for indxd in range(m1.cfg["model"]["params"]["discrete_state_dim"]):
        for indx in range(state_sample.shape[0]):
            tmp_state = torch.zeros(state_sample.shape[1] + 1)
            tmp_state[:-1] = state_sample[indx, :]
            tmp_state[-1] = 1. * indxd
            state_sample[indx, :] = tmp_state[:-1]

        V1 = m1.M[indxd][0].eval()
        V2 = m2.M[indxd][0].eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            V_p1 = V1(state_sample).mean
            V_p2 = V2(state_sample).mean

        error_vec += np.array([[(1 / V_p1.shape[0]**0.5 * torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p1)))).numpy(),
                                (torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p1)),ord=float('inf'))).numpy()]])

    error_out = np.zeros((1, 4))
    error_out[0, 2:] = error_vec[0, :]/m1.cfg["model"]["params"]["discrete_state_dim"]
    error_out[0,0] = m1.epoch
    error_out[0,1] = m2.epoch

    with open("V_func_error.txt", 'a') as csvfile:
        np.savetxt(
            csvfile,
            error_out,
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

    n_disc_states = m2.cfg["model"]["params"]["discrete_state_dim"]
    if m2.epoch > m1.cfg["n_feas_set_it"]:
        gp_offset = m2.cfg["model"]["params"]["GP_offset"]
    else:
        gp_offset = torch.zeros(n_types)

    max_v = torch.zeros(n_disc_states)
    min_v = torch.zeros(n_disc_states)
    mean_v = torch.zeros(n_disc_states)
    for indxs in range(n_disc_states):
        mask_state = m2.state_sample[:,-1] == 1.*indxs
        max_v[indxs] = torch.max(m2.V_sample[mask_state] + gp_offset[indxs])
        min_v[indxs] = torch.min(m2.V_sample[mask_state] + gp_offset[indxs])
        mean_v[indxs] = torch.mean(m2.V_sample[mask_state] + gp_offset[indxs])
    
    logger.info(
        f"Difference epochs {m1.epoch} and {m2.epoch} in prediction (norm): L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p1)))}; L_inf: {torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p1)),ord=float('inf'))} max by state {max_v}"
    )
    # logger.info(
    #f"Difference in T+1 iteration vs interpolated: L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm(V_p2 - m2.V_sample)}; L_inf: {torch.linalg.norm(V_p2 - m2.V_sample,ord=float('inf'))}"
    # )


def simulate(m, m_prev, cfg, model):

    torch.manual_seed(1054211) #set seed for reproducible results
    optimize = False
    n_sim_steps = 10
    n_types = m.cfg["model"]["params"]["n_types"]
    trans_mat = m.cfg["model"]["params"]["trans_mat"]
    lower_V = m.cfg["model"]["params"]["lower_V"]
    beta = m.cfg["model"]["params"]["beta"]
    sigma_util = m.cfg["model"]["params"]["sigma"]

    # fitting policy in case we have not done so yet
    if m.cfg.get("DISABLE_POLICY_FIT"):
        m.policy_fit(m.cfg["torch_optim"]["iter_per_cycle"])
        
    logger.info("Training of policies done now simulate")

    # setting to evaluate
    for d in range(m.discrete_state_dim):
        for p in range(m.policy_dim + 1):
            m.M[d][p].eval()
            m_prev.M[d][p].eval()

    max_ind = torch.argmax(m.V_sample)
    start_pt = torch.unsqueeze(m.state_sample[max_ind, :], 0)

    # deleting previous file
    sim_out = open(f"simulation_{m.epoch}.txt", 'w')
    sim_out.close()

    # opening for appending
    sim_out = open(f"simulation_{m.epoch}.txt", 'a')
    top_out = "# "
    for key, val in m.S.items():
        top_out = top_out + key + " "

    top_out = top_out + "DS VF CB_L CB_U R_DIFF "
    for key, val in m.P.items():
        top_out = top_out + key + " "

    sim_out.write(top_out + "\n")
    gp_offset = m.cfg["model"]["params"]["GP_offset"]

    current_state = start_pt 

    dim_state = start_pt.shape[1]
    out_np = np.zeros([1, m.policy_dim + 4 + dim_state])
    abs_diff_vec = np.zeros([n_sim_steps])
    for indxt in range(n_sim_steps):

        #if indxt < 6:
       #     current_state[0, -1] = 0.
        #else:
        #    current_state[0, -1] = 1.

        params = m.get_params(current_state[0, :],None)

        out_np[0, 0:dim_state] = (current_state[0, :]).numpy()  # current state
        current_disc_state = int(current_state[0, -2].item())
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = m.M[0][0](
                current_state[:, :-1])
            out_np[0, dim_state] = (pred.mean + gp_offset[current_disc_state]).numpy()  # value function

            lower,upper = pred.confidence_region() #confidence interval
            out_np[0, dim_state+1] = lower.numpy()
            out_np[0, dim_state+2] = upper.numpy()

            abs_diff_vec[indxt] = (
                np.abs((m_prev.M[0][0](
                current_state[:, :-1]).mean + gp_offset[current_disc_state]).numpy() - out_np[0, dim_state])/
                (1 + np.abs(out_np[0, dim_state])))  # relative abs of difference of vf in this and the prior epoch

            out_np[0, dim_state+3] = abs_diff_vec[indxt]

            pol_out = torch.zeros(m.control_dim)
            LB_pol = m.lb(current_state[0, :],params)
            UB_pol = m.ub(current_state[0, :],params)
            for indxp in range(1, m.policy_dim + 1):
                pol_out[indxp - 1] = torch.minimum(
                    torch.tensor(UB_pol[indxp - 1]),
                    torch.maximum(
                        torch.tensor(LB_pol[indxp - 1]),
                        (m.M[0][indxp](current_state[:, :-1]).mean)))[0]
                out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy



        #scale back to Fernandes Pheland's values
        out_np[0, dim_state] = out_np[0, dim_state] #/ (1 - beta)
        out_np[0, :dim_state - 1] = out_np[0,:dim_state - 1]
        np.savetxt(
            sim_out,
            out_np,
            fmt='%.18e',
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

        target = m.cfg["BAL"]["targets"][0]
        bal_util = m.bal_utility_func(current_state,0,target.get("rho"),target.get("beta"))
        logger.info(f"iteration {indxt} output {out_np[0,0:]}")        
        sim_out.flush()

        cat_dist = torch.distributions.categorical.Categorical(
            torch.tensor(trans_mat[current_disc_state]))
        next_disc_state = int((cat_dist.sample()).item())
        current_state = torch.unsqueeze(m.state_next(
            current_state[0, :], params, pol_out, next_disc_state), 0)

    l2_err = np.linalg.norm(abs_diff_vec/n_sim_steps,ord=2)
    linf_err = np.quantile(abs_diff_vec,0.99)
    logger.info(f"Relative error along simulatin path: L2 error {l2_err} Linf error {linf_err}")
    sim_out.write(f"#Relative error along simulatin path: L2 error {l2_err} Linf error {linf_err}")
    sim_out.close()  