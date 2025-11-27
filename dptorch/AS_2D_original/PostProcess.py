from matplotlib import pyplot as plt
import torch
import gpytorch
import matplotlib
import logging
import numpy as np
from Utils import NonConvergedError

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

def process(m, cfg, checkpoint_indx_start, checkpoint_indx):

    lower_V = m.cfg["model"]["params"]["lower_V"]
    gp_offset = m.cfg["model"]["params"]["GP_offset"]
    sigma_util = m.cfg["model"]["params"]["sigma"]
    beta = m.cfg["model"]["params"]["beta"]

    # deleting the error file
    f = open(f"V_func_error_{checkpoint_indx_start}_{checkpoint_indx}.txt", 'w')
    n_plot_pts = 10000

    plot_sample,dummy = m.sample(n_plot_pts)

    n_types = m.cfg["model"]["params"]["discrete_state_dim"]
    lower_w = m.cfg["model"]["params"]["lower_w"]
    upper_w = m.cfg["model"]["params"]["upper_w"]
    logger.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")  

    top_out = "# "
    for key, val in m.S.items():
        top_out = top_out + key + " "

    top_out_with_ds = top_out + "DS VF "


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
    
    out_lst = m.only_fit_VF()# + m.only_fit_trans_pol()
    for indxd in range(len(out_lst)):
        mean = m.M[out_lst[indxd][0]][out_lst[indxd][1]].mean
        var = m.M[out_lst[indxd][0]][out_lst[indxd][1]].var
        print(f"State {out_lst[indxd]} Mean: {mean}; Variance: {var}")        
        for param_name, param in m.M[out_lst[indxd][0]][out_lst[indxd][1]].named_parameters():
            print(f"State {out_lst[indxd]} Name: {param_name}; value: {param}")



def compare(m1, m2, cfg):
    """Compare two GP-s"""

    n_test_pts = 1000
    state_sample,dummy = m1.sample(n_test_pts)
    state_sample = state_sample[:,:-1]

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

        error_vec += np.array([[(1 / V_p1.shape[0]**0.5 * torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)))).numpy(),
                                (torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)),ord=float('inf'))).numpy()]])

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
    gp_offset = m2.cfg["model"]["params"]["GP_offset"]
    max_v = torch.zeros(n_disc_states)
    min_v = torch.zeros(n_disc_states)
    mean_v = torch.zeros(n_disc_states)
    error_at_sample = np.zeros([n_disc_states,2])
    error_at_sample_sum = np.zeros(2)
    for indxs in range(n_disc_states):
        mask_state = m2.state_sample[:,-1] == 1.*indxs
        mask_state_prev = m1.state_sample[:,-1] == 1.*indxs
        V1 = m1.V_sample[mask_state_prev]
        n_pts = V1.shape[0]
        V2 = m2.V_sample[mask_state][:n_pts]
        error_tmp = np.array([
            (1 / V1.shape[0]**0.5 * torch.linalg.norm((V1 - V2)/(1 + torch.abs(V2)))).numpy(),
            (torch.linalg.norm((V1 - V2)/(1 + torch.abs(V2)),ord=float('inf'))).numpy()])        
        
        error_at_sample_sum += error_tmp/n_disc_states
        error_at_sample[indxs,:] = error_tmp

        max_v[indxs] = torch.max(m2.V_sample[mask_state] + gp_offset)
        min_v[indxs] = torch.min(m2.V_sample[mask_state] + gp_offset)
        mean_v[indxs] = torch.mean(m2.V_sample[mask_state] + gp_offset)
    
    # logger.info(
    #     f"Difference epochs {m1.epoch} and {m2.epoch} in prediction (norm): L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)))}; L_inf: {torch.linalg.norm((V_p1 - V_p2)/(1 + torch.abs(V_p2)),ord=float('inf'))} max by state {max_v} min {min_v}"
    # )
    logger.info(
        f"Difference epochs {m1.epoch} and {m2.epoch} in prediction (norm): L2: {error_at_sample_sum[0]}; L_inf: {error_at_sample_sum[1]} max by state {max_v} min {min_v}"
    )

    # logger.info(
    #f"Difference in T+1 iteration vs interpolated: L2: {1 / V_p1.shape[0]**0.5 * torch.linalg.norm(V_p2 - m2.V_sample)}; L_inf: {torch.linalg.norm(V_p2 - m2.V_sample,ord=float('inf'))}"
    # )


def simulate(m, m_prev, cfg, model):

    torch.manual_seed(1054211) #set seed for reproducible results
    n_sim_steps = 10000
    n_types = m.cfg["model"]["params"]["n_types"]
    trans_mat = m.cfg["model"]["params"]["trans_mat"]
    lower_V = m.cfg["model"]["params"]["lower_V"]
    beta = m.cfg["model"]["params"]["beta"]
    sigma_util = m.cfg["model"]["params"]["sigma"]

    # fitting policy in case we have not done so yet
    m.policy_fit(m.cfg["torch_optim"]["iter_per_cycle"])
        
    logger.info("Training of policies done now simulate")

    # setting to evaluate
    for d in range(m.discrete_state_dim):
        for p in range(m.policy_dim + 1):
            m.M[d][p].eval()

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
    l1_err = 0
    l2_err = 0
    for indxt in range(n_sim_steps):

        params = m.get_params(current_state[0, :],torch.zeros([m.combined_sample_all.shape[1]-1]))

        out_np[0, 0:dim_state] = (current_state[0, :]).numpy()  # current state
        current_disc_state = int(current_state[0, -1].item())
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = m.M[current_disc_state][0](
                current_state[:, :-1])
            out_np[0, dim_state] = (pred.mean + gp_offset).numpy()  # value function

            lower,upper = pred.confidence_region() #confidence interval
            out_np[0, dim_state+1] = lower.numpy()
            out_np[0, dim_state+2] = upper.numpy()

            abs_diff = (
                np.abs((m_prev.M[current_disc_state][0](
                current_state[:, :-1]).mean + gp_offset).numpy() - out_np[0, dim_state])/
                (1 + np.abs(out_np[0, dim_state])))  # relative abs of difference of vf in this and the prior epoch

            l1_err += abs_diff
            l2_err += abs_diff**2

            out_np[0, dim_state+3] = abs_diff

            pol_out = torch.zeros(m.control_dim)
            LB_pol = m.lb(current_state[0, :],params)
            UB_pol = m.ub(current_state[0, :],params)
            scale_vec = m.scale_policy()
            for indxp in range(1, m.policy_dim + 1):
                pol_out[indxp - 1] = torch.minimum(
                    torch.tensor(UB_pol[indxp - 1]),
                    torch.maximum(
                        torch.tensor(LB_pol[indxp - 1]),
                        (m.M[current_disc_state][indxp](current_state[:, :-1]).mean) / scale_vec[indxp - 1]))[0]
                out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy


        for indxp in range(1, m.policy_dim + 1):
            out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy
            
        #scale back to Fernandes Pheland's values
        out_np[0, dim_state] = out_np[0, dim_state] 
        out_np[0, :dim_state - 1] =  (model.unscale_state_func(torch.from_numpy(out_np[0,:dim_state - 1]),m.cfg)).numpy()
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

        bal_util = m.bal_utility_func(current_state,0)
        logger.info(f"iteration {indxt} output {out_np[0,:n_types+2]}")        
        sim_out.flush()

        cat_dist = torch.distributions.categorical.Categorical(
            torch.tensor(trans_mat[current_disc_state]))
        next_disc_state = int((cat_dist.sample()).item())
        current_state = torch.unsqueeze(m.state_next(
            current_state[0, :], params, pol_out, next_disc_state), 0)

    l2_err = np.sqrt(l2_err/n_sim_steps)
    l1_err = l1_err/n_sim_steps
    logger.info(f"Relative error along simulatin path: L2 error {l2_err} L1 error {l1_err}")
    sim_out.write(f"#Relative error along simulatin path: L2 error {l2_err} L1 error {l1_err}")
    sim_out.close()  
