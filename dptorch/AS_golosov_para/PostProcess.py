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
    for indxd in range(n_types):
        mask = plot_sample[:,-1] == 1. * indxd
        eval_samples = plot_sample[mask,:-1]
        # grad_V_fun = grad_V(m,indxd,eval_samples)
        # mat_of_dd = (grad_V_fun @ grad_V_fun.T)/n_plot_pts
        # E_vals,E_vecs = torch.linalg.eigh(mat_of_dd)
        # print(f"largest Eigenvalues in state {indxd} {E_vals[-n_types:]}")

        d = indxd
        mask = m.state_sample[:, -1] == d * torch.tensor(1.)
        max_v_ind = torch.argmax(m.V_sample[mask] + gp_offset)
        min_v_ind = torch.argmin(m.V_sample[mask] + gp_offset)
        V = m.V_sample[mask] + gp_offset
        sample = m.state_sample_all[mask,:]
        logger.info(f">>>  maximal value in state {d} is {V[max_v_ind]} min is {V[min_v_ind]} state {sample[max_v_ind,:]}")    
        #index_lst = (mask.type(torch.IntTensor))

        V_fun = m.M[d][0].eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            V_pred = (V_fun(eval_samples).mean + gp_offset)

        top_out = "# "
        
        for key, val in m.S.items():
            top_out = top_out + key + " "

        top_out = top_out + "VF "

        #deleting prev file and write out the header
        file_out = open(f"V_func_all_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_func_all_{m.epoch}_{indxd}.txt", 'a')

        plot_pts = torch.cat(
            ((
                torch.cat(
                    (eval_samples,
                     indxd * torch.ones(
                         (eval_samples.shape[0],1))
                     ),
                    dim=1)[:,:-1]) *
                torch.tensor((1 - sigma_util)/(1-beta)), #scale back to Fernandes Pheland's values
                torch.unsqueeze(V_pred,1)),
            dim=1)

        np.savetxt(
            file_out,
            plot_pts.numpy(),
            fmt='%.18e',
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

        file_out.close()

        #deleting prev file and write out the header
        file_out = open(f"V_func_feas_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_func_feas_{m.epoch}_{indxd}.txt", 'a')

        mask_feas = V_pred >= lower_V
        feas_samples = eval_samples[mask_feas, :]
        V_feas = V_pred[mask_feas]

        plot_pts = torch.cat(
            ((
                torch.cat(
                    (feas_samples,
                     indxd * torch.ones((feas_samples.shape[0],1))),
                    dim=1)[:,:-1]) *
                torch.tensor((1 -sigma_util)/(1-beta)), #scale back to Fernandes Pheland's values
                torch.unsqueeze(V_feas,1)),
            dim=1)

        np.savetxt(
            file_out,
            plot_pts.numpy(),
            fmt='%.18e',
            delimiter=' ',
            newline='\n',
            header='',
            footer='',
            comments='# ',
            encoding=None)

        file_out.close()

        top_out = "# "
        for key, val in m.S.items():
            top_out = top_out + key + " "

        top_out_with_ds = top_out + "DS VF "
        top_out = top_out + "VF "

        for key, val in m.P.items():
            top_out = top_out + key + " "
            top_out_with_ds = top_out_with_ds + key + " "

        #deleting prev file and write out the header
        file_out = open(f"V_comp_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_comp_{m.epoch}_{indxd}.txt", 'a')

        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            V_rr = torch.max(torch.abs(V_fun(m.state_sample[mask, :][:, :-1]).mean - m.V_sample[mask]))

        print(f"Interpolation error {V_rr}")

        pts_out = torch.cat(
            (
                (m.state_sample[mask, :][:, :-1]) * torch.tensor((1 - sigma_util)/(1-beta)), #scale back to Fernandes Pheland's values
                torch.unsqueeze((m.V_sample[mask] + gp_offset) , dim=-1),
                m.policy_sample[mask, :]
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

        #deleting prev file and write out the header
        file_out = open(f"V_raw_{m.epoch}_{indxd}.txt", 'w')
        file_out.write(top_out + "\n")
        file_out.close()
        file_out = open(f"V_raw_{m.epoch}_{indxd}.txt", 'a')

        pts_out = torch.cat(
            (
                (m.state_sample[mask, :][:, :-1]),
                torch.unsqueeze(m.V_sample[mask] , dim=-1),
                m.policy_sample[mask, :]
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

        # diff_all_pts = torch.unsqueeze(eval_samples[mask_feas, :],1) - torch.unsqueeze(eval_samples[mask_feas, :],0)
        # diff_all_pts = torch.norm(diff_all_pts, dim=2)
        # diff_all_pts = diff_all_pts[diff_all_pts > 0.0001]
        # print(f"Minimal distance between points in state {indxd} is {torch.min(diff_all_pts)}")
        # print(f"Maximal distance between points in state {indxd} is {torch.max(diff_all_pts)}")

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
    
    out_lst = m.only_fit_VF() + m.only_fit_trans_pol()
    for indxd in range(len(out_lst)):
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
    optimize = False
    n_sim_steps = 1000
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

        # if indxt > 500:
        #    current_state[0, -1] = 0.
        # else:
        #    current_state[0, -1] = 1.

        params = m.get_params(current_state[0, :],None)

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
            
        bellman_eq_err = abs( (m.eval_f(current_state[0, :],params,pol_out)).detach().numpy() - out_np[0, dim_state])/(1 + np.abs(out_np[0, dim_state]))

        if optimize:
            if bellman_eq_err > 0.01:
                try:

                    m.cfg["ipopt"]["no_restarts"] = 5
                    v, p = m.solve(current_state[0, :],pol_out)
                    out_np[0, dim_state + 4:] = p[:]
                    bellman_eq_err = abs( v.detach().numpy()  + gp_offset.detach().numpy() - out_np[0, dim_state])/(1 + np.abs(out_np[0, dim_state]))
                    out_np[0, dim_state] = v.detach().numpy()  + gp_offset.detach().numpy()
                    pol_out = (p)


                except NonConvergedError as e:
                    for indxp in range(1, m.policy_dim + 1):
                        out_np[0, dim_state + 4 + indxp - 1] = pol_out[indxp - 1].numpy()  # indxp th policy
                    bellman_eq_err = abs( (m.eval_f(current_state[0, :],params,pol_out)).detach().numpy() - out_np[0, dim_state])/(1 + np.abs(out_np[0, dim_state]))



        #scale back to Fernandes Pheland's values
        out_np[0, dim_state] = out_np[0, dim_state] #/ (1 - beta)
        out_np[0, :dim_state - 1] = out_np[0,:dim_state - 1] * (1 - sigma_util) / (1 - beta)
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
