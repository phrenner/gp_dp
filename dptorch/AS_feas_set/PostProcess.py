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

    lower_V = m.cfg["model"]["params"]["lower_V"]
    gp_offset = m.cfg["model"]["params"]["GP_offset"]
    sigma_util = m.cfg["model"]["params"]["sigma"]
    beta = m.cfg["model"]["params"]["beta"]

    # deleting the error file
    f = open("V_func_error.txt", 'w')
    n_plot_pts = 50000

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

        diff_all_pts = torch.unsqueeze(m.state_sample[mask, :][:, :-1],1) - torch.unsqueeze(m.state_sample[mask, :][:, :-1],0)
        diff_all_pts = torch.norm(diff_all_pts, dim=2)
        diff_all_pts = diff_all_pts[diff_all_pts > 0.0001]
        print(f"Minimal distance between points in state {indxd} is {torch.min(diff_all_pts)}")
        print(f"Maximal distance between points in state {indxd} is {torch.max(diff_all_pts)}")

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
        vec = m.M[out_lst[indxd][0]][out_lst[indxd][1]].covar_module.base_kernel.lengthscale
        scale = m.M[out_lst[indxd][0]][out_lst[indxd][1]].covar_module.outputscale
        # alpha = m.M[out_lst[indxd][0]][out_lst[indxd][1]].covar_module.base_kernel.alpha
        print(f"{out_lst[indxd]} {vec} {scale}")


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

    return None
