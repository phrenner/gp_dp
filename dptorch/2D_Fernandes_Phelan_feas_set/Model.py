import torch
import gpytorch
import numpy as np
# from DPGPIpoptModel import DPGPIpoptModel,IpoptModel
from DPGPScipyModel import DPGPScipyModel
# import cyipopt
from typing import List, Tuple
import copy
from Utils import NonConvergedError
import logging
from math import ceil

from Utils import NonConvergedError
import os
from mpi_dist import dist

torch.set_default_dtype(torch.float64)

rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))
if rank != -1:
    logger = logging.getLogger(f"DPGPModel.Rank{rank}")
else:
    logger = logging.getLogger(f"DPGPModel")

# ======================================================================
#
#     sets the parameters for the model
#     "APS_nocharge"
#
#     Philipp Renner, 09/21
# ======================================================================

# ======================================================================


def utility_ind(cons, reg_c, sigma): #utility function
   return (cons + reg_c)**(1 - sigma)/(1 - sigma) - (reg_c)**(1 - sigma)/(1 - sigma)

def inv_utility_ind(util, reg_c, sigma): #inverse utility function
   return ((1-sigma)*(util + (reg_c)**(1 - sigma)/(1 - sigma)))**(1/(1 - sigma)) - reg_c

n_rand_its = 1

def scale_state_func(unscaled_state,cfg):
    beta = cfg["model"]["params"]["beta"]
    n_types = cfg["model"]["params"]["n_types"]
    sigma = cfg["model"]["params"]["sigma"]
    reg_c = cfg["model"]["params"]["reg_c"]
    upper_w = cfg["model"]["params"]["upper_w"]
    lower_w = cfg["model"]["params"]["lower_w"]
    state_scaled = unscaled_state.clone()
    state_scaled[...,:] = unscaled_state[...,:] * (1-beta)
    return state_scaled

def unscale_state_func(state_scaled,cfg):
    beta = cfg["model"]["params"]["beta"]
    n_types = cfg["model"]["params"]["n_types"]
    sigma = cfg["model"]["params"]["sigma"]
    reg_c = cfg["model"]["params"]["reg_c"]
    upper_w = cfg["model"]["params"]["upper_w"]
    lower_w = cfg["model"]["params"]["lower_w"]
    unscaled_state = state_scaled.clone()
    unscaled_state[...,:] = state_scaled[...,:] / (1-beta)
    return unscaled_state

# ======================================================================


# V infinity
def V_INFINITY(model, scaled_state):
    state = model.unscale_state(scaled_state)
    n_types=model.cfg["model"]["params"]["n_types"]

    # max_val = (0.5 * (upper_w[disc_state] - lower_w[disc_state]))**2
    # adjust_fac = (0.5 * shock_vals[disc_state] - (lower_V - max_pen)) / max_val
    # sum_tmp = torch.tensor(0.5 * shock_vals[disc_state])
    # for indxa in range(n_types):
    #     sum_tmp -=  trans_mat[disc_state,indxa]*(
    #          (0.5*(lower_w[disc_state]+upper_w[disc_state]) - state[model.S[f"w_{indxa+1}"]])**2) * adjust_fac

    v_infinity = torch.tensor(0.)


    return v_infinity


# ======================================================================
#   Equality constraints during the VFI of the model


def EV_G_ITER(model, scaled_state, params, control):
    state = model.unscale_state(scaled_state)
    n_types = model.cfg["model"]["params"]["n_types"]
    trans_mat=model.cfg["model"]["params"]["trans_mat"]
    shock_vec=model.cfg["model"]["params"]["shock_vec"]
    reg_c=model.cfg["model"]["params"]["reg_c"]
    sigma=model.cfg["model"]["params"]["sigma"]
    beta=model.cfg["model"]["params"]["beta"]
    P=model.P
    S=model.S

    # M = 2*n_types + n_types - 1  # number of constraints
    M = 2*n_types + n_types*(n_types-1) // 2  # number of constraints
    disc_state = state[-1].type(torch.IntTensor)

    G = torch.empty(M)

    #equality constraints
    counter = 0
    for indxs in range(n_types):
        G[counter] = utility_ind(control[P[f"c_{indxs+1}"]], reg_c, sigma) - control[P[f"u_{indxs+1}"]]
        counter += 1

    #promise keeping
    for indx in range(n_types):
        G[counter] = state[S[f"w_{indx+1}"]] + control[P[f"pen_{indx+1}"]] - control[P[f"pen_u_{indx+1}"]] - \
                sum(
                    [
                        (control[P[f"u_{indxs+1}"]] + beta*(state[S[f"w_{indxs+1}"]] + control[P[f"fut_util_{indxs+1}_{indxs+1}"]]))*trans_mat[indx,indxs]  
                        for indxs in range(n_types) ])
        # G[counter] = state[S[f"w_{indx+1}"]] + control[P[f"pen_{indx+1}"]] - control[P[f"pen_u_{indx+1}"]] - \
        #         sum(
        #             [
        #                 (control[P[f"u_{indxs+1}"]] + beta*(control[P[f"fut_util_{indxs+1}_{indxs+1}"]]))*trans_mat[indx,indxs]  
        #                 for indxs in range(n_types) ])
        counter += 1


    #inequality constraints
    #incentive constraints
    for indx_true in range(1,n_types): # true state in
        for indx_false in range(indx_true): # false state in
            G[counter] = control[P[f"u_{indx_true+1}"]] + beta*(state[S[f"w_{indx_true+1}"]] + control[P[f"fut_util_{indx_true+1}_{indx_true+1}"]]) - \
                (utility_ind(shock_vec[indx_true] + control[P[f"c_{indx_false+1}"]] - shock_vec[indx_false], reg_c, sigma) + beta*(state[S[f"w_{indx_true+1}"]] + control[P[f"fut_util_{indx_false+1}_{indx_true+1}"]]))
            # G[counter] = control[P[f"u_{indx_true+1}"]] + beta*(control[P[f"fut_util_{indx_true+1}_{indx_true+1}"]]) - \
            #     (utility_ind(shock_vec[indx_true] + control[P[f"c_{indx_false+1}"]] - shock_vec[indx_false], reg_c, sigma) + beta*(control[P[f"fut_util_{indx_false+1}_{indx_true+1}"]]))
            counter += 1


    return G


# ======================================================================
class SpecifiedModel(DPGPScipyModel):
    def __init__(self, V_guess=V_INFINITY, cfg={}, **kwargs):
        policy_names = []
        policy_names += [f"c_{i+1}" for i in range(cfg["model"]["params"]["n_types"])]
        policy_names += [f"u_{i+1}" for i in range(cfg["model"]["params"]["n_types"])]
        for indxr in range(cfg["model"]["params"]["n_types"]):
            for indxc in range(cfg["model"]["params"]["n_types"]):
                policy_names += [f"fut_util_{indxr+1}_{indxc+1}"]


        policy_names += [f"pen_{i+1}" for i in range(cfg["model"]["params"]["n_types"])]
        policy_names += [f"pen_u_{i+1}" for i in range(cfg["model"]["params"]["n_types"])]

        state_names = [f"w_{i+1}" for i in range(cfg["model"]["params"]["n_types"])]

        # for faster indexing
        self.fut_util_mask = torch.tensor(
            [x.startswith("fut_util_") for x in policy_names]
        )
        self.pen_mask = torch.tensor(
            [x.startswith("pen_") for x in policy_names]
        )
        super().__init__(
            V_guess=lambda x: V_INFINITY(
                self,
                x,
            ),
            cfg=cfg,
            policy_names=policy_names,
            state_names=state_names,
            policy_dim=4 * cfg["model"]["params"]["n_types"] + cfg["model"]["params"]["n_types"]**2,
            discrete_state_dim=cfg["model"]["params"].get("discrete_state_dim", 1),
            control_dim=4 * cfg["model"]["params"]["n_types"] + cfg["model"]["params"]["n_types"]**2,
            **kwargs
        )

    def get_fit_precision(self,d,p):
        if p == 0:
            rel_ll_change_tol = self.cfg["torch_optim"].get("relative_ll_change_tol_vf",1e-4)
            relative_ll_grad_tol = self.cfg["torch_optim"].get("relative_ll_grad_change_tol_vf",1e-2)
            relative_error_tol = self.cfg["torch_optim"].get("relative_error_tol_vf", 0)
            parameter_change_tol= self.cfg["torch_optim"].get("parameter_change_tol_vf", 0)

        else:
            rel_ll_change_tol = self.cfg["torch_optim"].get("relative_ll_change_tol_pol",1e-4)
            relative_ll_grad_tol = self.cfg["torch_optim"].get("relative_ll_grad_change_tol_pol",1e-2)
            relative_error_tol = self.cfg["torch_optim"].get("relative_error_tol_pol", 0)
            parameter_change_tol= self.cfg["torch_optim"].get("parameter_change_tol_pol", 0)

        return rel_ll_change_tol, relative_ll_grad_tol, relative_error_tol, parameter_change_tol   

    def create_optimizer(self, d, p):
        train_iter,lr = self.get_solver_config(d,p)
        return  torch.optim.Adam(self.M[d][p].parameters(),lr = lr)
                

    def get_solver_config(self,d,p):
        if p == 0:
            training_iter = self.cfg["torch_optim"].get("iter_per_cycle_vf",100)
            lr = self.cfg["torch_optim"].get("LR_vf",1e-3)
        else:
            training_iter = self.cfg["torch_optim"].get("iter_per_cycle_pol",100)
            lr = self.cfg["torch_optim"].get("LR_pol",1e-3)

        return training_iter, lr

    def what_pol_to_fit(self): #per default only fit the VF in each iteration; customize if we need some policies
        n_types = self.cfg["model"]["params"]["n_types"]
        fit_lst = []
        for indxs in range(n_types):
            for indxa1 in range(n_types):
                for indxa2 in range(n_types):
                    fit_lst.append((indxs,1 + self.P[f"fut_util_{indxa1+1}_{indxa2+1}"]))

        return fit_lst

    def sample(self,no_samples_=None):
        torch.manual_seed(self.cfg["seed"] + 55*self.epoch)
        beta = self.cfg["model"]["params"]["beta"]
        n_types = self.cfg["model"]["params"]["n_types"]
        if no_samples_ is None:
            no_samples_ = self.cfg["no_samples"]

        #sample on [0,1] and later scale to right shape
        state_sample = (
            torch.rand(
                [int(no_samples_), n_types]
            )
        )

        lower_w = scale_state_func(self.cfg["model"]["params"]["lower_w"],self.cfg)
        upper_w = scale_state_func(self.cfg["model"]["params"]["upper_w"],self.cfg)
        no_samples = state_sample.shape[0]
        LB_state = torch.zeros(n_types)
        UB_state = torch.zeros(n_types)

        for indxt in range(n_types):
            LB_state[self.S[f"w_{indxt+1}"]] = lower_w[indxt]
            UB_state[self.S[f"w_{indxt+1}"]] = upper_w[indxt]

        state_sample = torch.unsqueeze(UB_state - LB_state,dim=0)*state_sample + torch.unsqueeze(LB_state,dim=0)

        no_samples = state_sample.shape[0]

        #copy sample points for each state and add the corresponding discrete state in the last entry of each row
        state_sample = torch.cat(
            (
                state_sample.repeat_interleave(self.discrete_state_dim,0),
                torch.unsqueeze((torch.arange(self.discrete_state_dim)).repeat(no_samples),-1),
            ),
            dim=1,
        )

        ## manually add points for plotting the feasible set
        additional_pts = (1-beta) * 2 * torch.tensor([
            [9,10]])
        
        n_additional_pts = additional_pts.shape[0]

        manual_points = (1-beta) * torch.tensor([
            [3.5, 7.0, 0],   
            [3., 6.0, 0],   
            [4.0, 10.0, 0],  
            [4.0, 12.0, 0],  
            [4.0, 11.0, 0],  
            [4.0, 13.0, 0],  
            [7.0, 18.0, 0],  
            [10.0, 18.3, 0], 
            [13.0, 19.0, 0], 
            [17.0, 20.0, 0], 
            [17.5, 19.0, 0], 
            [16.0, 17.0, 0], 
            [14.5, 15.5, 0], 
            [13.0, 14.5, 0], 
            [11.0, 12.5, 0], 
            [11.0, 18.5, 0], 
            [9.5, 10.5, 0],  
            [8.0, 9.5, 0],   
            [8.0, 18.0, 0],   
            [9.0, 18.0, 0],   
            [13.5, 18.0, 0],   
            [12.0, 18.75, 0],   
            [15.0, 19.0, 0],   
            [15.5, 18.8, 0],   
            [17.0, 19.5, 0],   
            [17.0, 19.75, 0],   
            [6.5, 8.0, 0],   
            [6.0, 17.5, 0],   
            [5.5, 7.0, 0],   
            [5.25, 17.5, 0],   
            [3.5, 7., 0], 
            [3., 7., 0], 
            [3.5, 8., 0],    
            [3.9, 9.0, 0],
            [4.,12.0,0],
            [4.25,14.0,0],
            [4.,6.5,0],
            [4.,6.0,0],
            [4.,8.0,0],
            [4.,9.0,0],
            [4.5,14.75,0],
            [5.0,16.0,0],
            [5.0,17.0,0],
            [7.0,17.5,0],
            [3.0, 6.0,0], 
            [7.0, 15.0,0], 
            [4.137998707602195, 6.683973820051117,0], 
            [19.54561396845826, 20.25631197363793,0]            
        ])               
        state_sample = torch.cat(
            (
                torch.cat(
                    (
                        additional_pts,
                        torch.zeros([n_additional_pts,1]))  # we only plot the type 0 feasible set so plotting points are only needed there
                        ,dim=1),
                state_sample,
                manual_points
            ),dim=0
        )

        feasible = torch.ones(state_sample.shape[0])

        return (state_sample), feasible

    def scale_state(self,unscaled_state):
        n_types = self.cfg["model"]["params"]["n_types"]
        state_out = unscaled_state.clone()
        state_out[...,:n_types] = scale_state_func(unscaled_state[...,:n_types],self.cfg)
        return state_out

    def unscale_state(self,state_scaled):
        n_types = self.cfg["model"]["params"]["n_types"]
        state_out = state_scaled.clone()
        state_out[...,:n_types] = unscale_state_func(state_scaled[...,:n_types],self.cfg)
        return state_out
    
    def scale_policy(self):
        n_controls = self.control_dim
        scale_vec = torch.ones(n_controls)
        n_types = self.cfg["model"]["params"]["n_types"]
        beta = self.cfg["model"]["params"]["beta"]
        for indxt1 in range(n_types):
            for indxt2 in range(n_types):
                scale_vec[self.P[f"fut_util_{indxt1+1}_{indxt2+1}"]] = 1

        return scale_vec

    @torch.no_grad()
    def sample_start_pts(self, scaled_state, params, policy, n_restarts):
        torch.manual_seed(self.cfg["seed"] + 33*self.epoch)
        state = self.unscale_state(scaled_state)
        S = self.S
        P = self.P
        w_vec = state[:len(S)]
        beta = self.cfg["model"]["params"]["beta"]
        reg_c = self.cfg["model"]["params"]["reg_c"]
        sigma = self.cfg["model"]["params"]["sigma"]
        n_types = self.cfg["model"]["params"]["n_types"]

        disc_state = int(state[-1].item())

        inv_trans_mat = self.cfg["model"]["params"]["trans_mat_inv"]

        LB = torch.from_numpy(self.lb(scaled_state, params))
        UB = torch.from_numpy(self.ub(scaled_state, params))
        n_pts = 1*n_restarts
        policy_sample = (
            torch.rand(
                [n_pts, self.control_dim]
            )
            * (
                UB - LB
            )
            + LB
        )
        upper_transfer = self.cfg["model"]["params"]["upper_trans"]
        for indxa in range(n_types):
            UB[self.P[f"u_{indxa+1}"]] = utility_ind(upper_transfer, reg_c, sigma)
            UB[self.P[f"c_{indxa+1}"]] = upper_transfer    

        for indxp in range(n_pts):
            control = torch.zeros(policy_sample.shape[-1])
            control[:] = policy_sample[indxp,:]
            if indxp == 0:  #first batch assume we stay at current util
                for indxt1 in range(n_types):
                    for indxt2 in range(n_types):
                        control[P[f"fut_util_{indxt1+1}_{indxt2+1}"]] = 0.
                        # control[P[f"fut_util_{indxt1+1}_{indxt2+1}"]] = state[S[f"w_{indxt1+1}"]]

            for indxt in range(n_types): #we assume penalty zero as initial guess
                control[P[f"pen_{indxt+1}"]] = 0.
                control[P[f"pen_u_{indxt+1}"]] = 0.

            for indxr in range(n_types):

                control[self.P[f"c_{indxr+1}"]] = inv_utility_ind(control[P[f"u_{indxr+1}"]] ,reg_c,sigma)

            policy_sample[indxp,:] = control[:]
        
        policy_sample_out = policy_sample
        scale_vec = self.scale_policy()
        policy_sample_out[-1,:] = policy*scale_vec

        return policy_sample_out
        
    @torch.no_grad()
    def get_params(self,scaled_state,policy):
        params = {}

        return params

    def is_feasible(self, scaled_state,value,control):
        return 1.0

    def post_process_optimization(self, scaled_state, params, control, value):
        state = self.unscale_state(scaled_state)
        disc_state = int(state[-1].item())     
        error_tol = 1.0e-2
        n_types = self.cfg["model"]["params"]["n_types"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        pen_base = self.cfg["model"]["params"]["pen_vf"]


        total_pen = sum([ control[self.P[f"pen_{indxt}"]] for indxt in range(1,n_types+1)]) + \
            sum([ control[self.P[f"pen_u_{indxt}"]] for indxt in range(1,n_types+1)])
        
        pen_vf = pen_base

        out_val = value + total_pen * pen_opt_vf
        if total_pen * pen_opt_vf >= error_tol:

            out_val -= total_pen * pen_vf

        else:
            for indxa in range(n_types):
                control[self.P[f"pen_{indxa+1}"]] = 0.
                control[self.P[f"pen_u_{indxa+1}"]] = 0.

        # out_val_adj = torch.min(out_val, self.cfg["model"]["params"]["upper_V"])
        out_val_adj = torch.clip(out_val, min=self.cfg["model"]["params"]["lower_V"], max=self.cfg["model"]["params"]["upper_V"])
        scale_vec = self.scale_policy()
        return control*scale_vec,  out_val_adj


    def u(self, scaled_state, params, control):
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state_in = (state[-1]).type(torch.IntTensor)
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        shock_vals = self.cfg["model"]["params"]["shock_vec"]
        beta = self.cfg["model"]["params"]["beta"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        
        total = torch.tensor(0.0)
        for indxa in range(n_types):
            total += (1 - beta) * (0. * trans_mat[disc_state_in,indxa]*(shock_vals[indxa] - control[self.P[f"c_{indxa+1}"]])) - control[self.P[f"pen_{indxa+1}"]]*pen_opt_vf - control[self.P[f"pen_u_{indxa+1}"]]*pen_opt_vf
        
        return total  #lowest value will be greater than zero, useful for GP approximation of VF

    def E_V(self, scaled_state, params, control):
        state = self.unscale_state(scaled_state)

        e_v_next = 0
        lower_V = self.cfg["model"]["params"]["lower_V"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]

        weights, points = self.state_iterate_exp(scaled_state, params, control)
        for i in range(len(weights)):
            p_i = torch.unsqueeze(points[i, :-1], 0)
            obj_val = (self.M[int(points[i, -1].item())][0](p_i).mean)
            e_v_next += (obj_val)  * weights[i]

        return e_v_next


    def state_next(self, scaled_state, params, control, zpy, opt=False):
        """Return next periods states, given the controls of today and the random discrete realization"""
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        S = self.S

        s = state.clone()
        
        # update discrete state
        s[-1] = 1.*zpy
        
        beta = self.cfg["model"]["params"]["beta"]
        for indx in range(n_types):
            s[S[f"w_{indx+1}"]] = state[self.S[f"w_{indx+1}"]] + control[self.P[f"fut_util_{int(zpy)+1}_{indx+1}"]]
            # s[S[f"w_{indx+1}"]] = control[self.P[f"fut_util_{int(zpy)+1}_{indx+1}"]]

        return self.scale_state(s)
 
    def state_iterate_exp(self, scaled_state, params, control):
        """How are future states generated from today state and control"""
        state = self.unscale_state(scaled_state)
        
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state = state[-1].type(torch.IntTensor)
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        weights = torch.tensor([trans_mat[disc_state,indx]  for indx in range(n_types)])
        
        points = torch.cat(
            tuple(
                torch.unsqueeze(self.state_next(scaled_state, params, control, z), dim=0)
                for z in range(self.discrete_state_dim)
            ),
            dim=0,
        )
  

        return weights, points


    def lb(self, scaled_state, params):
        state = self.unscale_state(scaled_state)
        S = self.S
        disc_state = int(state[-1].item())
        n_types = self.cfg["model"]["params"]["n_types"]
        beta = self.cfg["model"]["params"]["beta"]
        lower_w = self.cfg["model"]["params"]["lower_w"]

        X_L = np.zeros(self.control_dim)

        for indxa in range(n_types):
            for indxa2 in range(n_types):
                X_L[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (lower_w[indxa2] - state[self.S[f"w_{indxa2 + 1}"]])
                # X_L[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (lower_w[indxa2])

        return X_L

    def ub(self, scaled_state, params):
        state = self.unscale_state(scaled_state)
        S = self.S
        disc_state = int(state[-1].item())
        n_types = self.cfg["model"]["params"]["n_types"]
        upper_transfer = self.cfg["model"]["params"]["upper_trans"]
        reg_c = self.cfg["model"]["params"]["reg_c"]
        sigma = self.cfg["model"]["params"]["sigma"]
        upper_w = self.cfg["model"]["params"]["upper_w"]
        beta = self.cfg["model"]["params"]["beta"]
        X_U = np.empty(self.control_dim)
        for indxa in range(n_types):
            X_U[self.P[f"u_{indxa+1}"]] = utility_ind(upper_transfer, reg_c, sigma)
            X_U[self.P[f"pen_{indxa+1}"]] = 10*upper_w[indxa]
            X_U[self.P[f"pen_u_{indxa+1}"]] = 10*upper_w[indxa]
            X_U[self.P[f"c_{indxa+1}"]] = upper_transfer
            for indxa2 in range(n_types):
                X_U[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (upper_w[indxa2] - state[self.S[f"w_{indxa2 + 1}"]])
                # X_U[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (upper_w[indxa2])

        return X_U

    def cl(self, scaled_state, params):
        n_types = self.cfg["model"]["params"]["n_types"]
        M = 2*n_types + n_types*(n_types-1) // 2
        G_L = np.empty(M)
        G_L[:] =  0.0
        return G_L

    def cu(self, scaled_state, params):
        n_types = self.cfg["model"]["params"]["n_types"]
        M = 2*n_types + n_types*(n_types-1) // 2
        G_U = np.empty(M)
        G_U[:] = 0.0
        G_U[2*n_types:] = 1.0e10
        return G_U

    def eval_g(self, scaled_state, params, control):

        return EV_G_ITER(
            self,
            scaled_state,
            params,
            control,
        )

##############################################################################################################################
###########              Functions used for Bayesian active learning                                            ##############
##############################################################################################################################


    def bayesian_opt_criterion(self, eval_pt,discrete_state, target_p, pen_val=torch.tensor([-1.0e10])):

        rho = self.cfg["BAL"].get("bal_rho",1.0)
        beta = self.cfg["BAL"].get("bal_beta",10.0)
        # #compute bayesian optimization criteria
        pred = self.M[discrete_state][target_p](
                        eval_pt
                    )

        var_v = pred.variance
        mean_v = pred.mean

        #Deisenroth criterion
        is_v_ok = (var_v > 0.0000001)
        out_vec = is_v_ok*(rho * (mean_v) + beta / 2.0 * torch.log(var_v + 1e-15)) + torch.logical_not(is_v_ok)*pen_val

        return out_vec

    def mean_squared_error_GP(self, eval_pt, discrete_state, target_p):

        try:
            eval_pt = self.M[discrete_state][target_p].feature_extractor(eval_pt)
        except:
            pass

        #compute the mean squared error MSE according to Dario Azzimonti Gaussian processes and sequential design of experiments (lecture)
        if eval_pt.shape[0] > 1:             
            eval_pt = torch.unsqueeze(eval_pt,1) #doing a batch of points needs to be done as batches of single points

        train_inputs = self.M[discrete_state][target_p].train_inputs[0]
        try:
            train_inputs = self.M[discrete_state][target_p].feature_extractor(train_inputs)
        except:
            pass

        kxx = self.M[discrete_state][target_p].covar_module(eval_pt).evaluate()
        kXx = self.M[discrete_state][target_p].covar_module(train_inputs,eval_pt).evaluate()
        kXX = self.M[discrete_state][target_p].covar_module(train_inputs,train_inputs).evaluate()
        try:
            U = torch.linalg.cholesky(kXX)
        except:
            U = torch.linalg.cholesky(kXX + 1e-2*torch.eye(kXX.shape[0]))
        kXXinv_kXx = torch.cholesky_solve(kXx, U)
        out_vec_ = kxx - torch.tensordot(kXx.transpose(-1,-2),kXXinv_kXx,[[-2,-1],[-1,-2]])

        out_vec = torch.diagonal(out_vec_)
        if out_vec.ndim == 2:
            out_vec = torch.diag(out_vec)

        return out_vec
    
    def bal_utility_func(self, scaled_state,target_p,pen_val=torch.tensor([-1.0e10])):
        if scaled_state.ndim == 1:
            discrete_state = int(scaled_state[-1].item())
            eval_pt = torch.unsqueeze(scaled_state[:-1],0)
        elif scaled_state.ndim == 2:
            discrete_state = int(scaled_state[0,-1].item())
            assert torch.min(scaled_state[:,-1] == scaled_state[0,-1]), f"Something wrong with discrete states {scaled_state}"
            eval_pt = scaled_state[:,:-1]

        eval_pt = (eval_pt[:,self.get_d_cols(discrete_state)])

        # out_vec = self.mean_squared_error_GP(eval_pt,discrete_state,target_p)
        out_vec = self.bayesian_opt_criterion(eval_pt,discrete_state,target_p)

        return out_vec


    def BAL(self):

        n_types = self.cfg["model"]["params"]["n_types"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        pen_val = -1.0e10
        dim_state = self.state_sample_all.shape[1]
        torch.manual_seed(self.cfg["seed"] + 77*self.epoch)

        if not self.cfg.get("distributed") or dist.get_rank() == 0 and self.epoch > 4:
            n_sample_pts = 1000
            rand_sample, dummy = self.sample(n_sample_pts)
            cand_pts_gather = torch.zeros([0,1 + dim_state])
            for indxt in range(n_types):
                mask_type = rand_sample[:,-1] == 1.*indxt
                with torch.no_grad():
                    V_val_vec = self.M[indxt][0](rand_sample[mask_type,:-1]).mean
                mask_feas = V_val_vec > - 0.05
                sample_tmp = rand_sample[mask_type,:][mask_feas,:]
                n_pts = sample_tmp.shape[0]
                bal_util = torch.zeros([n_pts])
                for indxp in range(n_pts):
                    with torch.no_grad():
                        bal_util[indxp] = self.bal_utility_func(sample_tmp[indxp,:],0, pen_val=torch.tensor([pen_val]))
                pts_tmp = torch.zeros([n_pts,1 + dim_state])
                pts_tmp[:,:-1] = sample_tmp
                pts_tmp[:,-1] = bal_util
                cand_pts_gather = torch.cat(
                    (
                        cand_pts_gather,
                        pts_tmp
                    ),
                    dim=0
                )


            #we only compute and return the results for rank 0
            cand_pts = torch.zeros([n_types,1 + dim_state])
            cand_pts[:,-1] = pen_val
            for indxtype in range(n_types):
                for indxp in range(cand_pts_gather.shape[0]):
                    if cand_pts_gather[indxp,-2].item() == 1.*indxtype:
                        if cand_pts_gather[indxp,-1] > cand_pts[indxtype,-1]:
                            cand_pts[indxtype,:] = cand_pts_gather[indxp,:]

            n_pts = 0
            for indxd in range(self.cfg["model"]["params"]["discrete_state_dim"]):
                if cand_pts[indxd,-1] > pen_val:
                    n_pts += 1

            out = torch.zeros([n_pts,dim_state])
            indxp = 0
            for indxd in range(self.cfg["model"]["params"]["discrete_state_dim"]):
                if cand_pts[indxd,-1] > pen_val:
                    out[indxp,:] = cand_pts[indxd,:-1]
                    indxp+=1

           
            logger.info(f"BAL added points {out} after iteration {indxt}")
            new_sample = out
            self.state_sample = torch.cat(
                (
                        self.prev_state_sample,
                        new_sample,
                ),
                dim=0,
            )
            self.feasible = torch.cat(
                (
                        self.prev_feasible,
                        torch.ones(new_sample.shape[0]),
                ),
                dim=0,
            )
            self.combined_sample = torch.cat(
                (
                        self.prev_combined_sample,
                        torch.zeros([new_sample.shape[0],1+self.policy_dim]),
                ),
                dim=0,
            )
        
        self.warm_start = False

    def only_fit_trans_pol(self):
        n_types = self.cfg["model"]["params"]["n_types"]
        return  [
                    (d, 1+self.P[f"fut_util_{p1}_{p2}"])
                    for p1 in range(1, n_types + 1)
                    for p2 in range(1, n_types + 1)
                    for d in range(self.discrete_state_dim)
                ]
    def only_fit_VF(self):
        n_types = self.cfg["model"]["params"]["n_types"]
        return [(d, 0) for d in range(self.discrete_state_dim)]


    def iterate(self):

        # fit pattern for GPs
        if self.epoch == 0:
            fit_lst = self.only_fit_VF()
            howard_lst = []

            # fit pol and vf in first iteration
            self.fit_GP(fit_lst)
            self.save()
            self.warm_start = False

        else:
            fit_lst = self.only_fit_VF()
            howard_lst = self.only_fit_VF()
            

        self.epoch += 1
        logger.info(f"Starting EPOCH #{self.epoch} - current sample size is {self.state_sample_all.shape[0]}")

        # generate new sample
        self.sample_all()

        # update V_sample for next-step by solving the VF iteration
        self.solve_all(self.cfg["scipyopt"].get("no_restarts", 1))

        # gather estimated VF to all processes
        self.allgather()

        #compute difference to previous iteration at sample points
        self.metrics[str(self.epoch)] = {
            "l2": self.convergence_error(fit_lst = fit_lst,ord=2),
            "l_inf": self.convergence_error(fit_lst = fit_lst),
        }

        logger.info(
            f"Difference between previous interpolated values & next iterated values: {self.metrics[str(self.epoch)]['l_inf']} (L_inf) {self.metrics[str(self.epoch)]['l2']} (L2) for state policy pairs {fit_lst}"
        )

        self.fit_GP(fit_lst, self.warm_start) 

        self.warm_start = self.cfg.get("warm_start", False)

        # perform Howard improvment step
        vf_lst = self.only_fit_VF()

        indx_it = 0
        error_old = 1
        while indx_it < self.cfg["model"]["params"]["n_Howard_steps"] and error_old > 1e-5 and self.epoch > 30:

            error_new = self.Howard_step(vf_lst)
            
            logger.info(
                f"Howard iteration {indx_it} error: {error_new} (L2)"
            )            

            if error_new >= error_old:
                logger.info("No improvement in Howard it detected aborting iteration")
                break

            indx_it += 1
            error_old = error_new


        if self.cfg.get("distributed") and dist.is_initialized():
            dist.barrier()
            for indxd in range(self.cfg["model"]["params"]["discrete_state_dim"]):
                self.sync_GP(indxd, 0, source_rank=0)


        if indx_it > 0:
            self.fit_GP(vf_lst, self.warm_start, compute_metrics=False)  #fit GPs after Howard steps


        self.save()

    @torch.no_grad()
    def Howard_step(self,vf_lst):
        self.V_sample = torch.zeros(self.state_sample.shape[0])
        policies = []
        for s in range(self.state_sample.shape[0]):
            state = self.state_sample[s, :]
            policy = self.combined_sample[s,1:]
            if (int(self.state_sample[s, -1]),0) in vf_lst:
                params = self.get_params(state,policy)
                control = policy

                value = self.eval_f(state, params, control)
                pol_new,v = self.post_process_optimization(state, params, control, value)

                self.V_sample[s] = v
                policies.append(torch.unsqueeze(pol_new,0))
            else:
                pol = self.combined_sample[s,1:]
                policies.append(torch.unsqueeze(pol, 0))
                self.feasible[s] = 0            



        # get the policy dimension
        if self.state_sample.shape[0] == 0:
            policies.append(torch.zeros([0,self.combined_sample.shape[-1] - 1]))

        self.policy_sample = torch.cat(policies, dim=0)
        self.combined_sample = torch.cat(
            (torch.unsqueeze(self.V_sample, dim=1), self.policy_sample), dim=1
        )

        self.allgather()

        metrics_how = {
                "l2": self.convergence_error(fit_lst = vf_lst,ord=2),
                "l_inf": self.convergence_error(fit_lst = vf_lst),
            }
        
        n_disc_states = self.cfg["model"]["params"]["discrete_state_dim"]
        error_new = torch.sum(metrics_how['l2'], dtype=torch.double)/n_disc_states

        error_new = dist.reduce_mean(error_new, dst=0)

        for indxd in range(n_disc_states):
            train_sample, train_v = self.set_train_data(indxd, 0, True) #update training data to update GP without refitting
            self.M[indxd][0].eval()
            self.likelihood[indxd][0].eval()

        return error_new


    def create_model(self, d, p, train_x, train_y, warm_start=False):
        if self.cfg.get('use_fixed_noise',True):
            if p == 0:
                noise_vec = torch.ones(train_y.shape[0]) * 1e-5
            else:
                noise_vec = torch.ones(train_y.shape[0]) * 1e-5

            self.likelihood[d][p] = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                        noise_vec,
                        learn_additional_noise=False
                    ).to(self.device)

        else:
            self.likelihood[d][p] = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-7)
                    ).to(self.device)

        import importlib
        gp_model = importlib.import_module(self.cfg["MODEL_NAME"] + ".GPModel")
        if p == 0:
            model = gp_model.GPModel(
                            d,
                            p,
                            train_x,
                            train_y,
                            self.likelihood[d][p],
                            self.cfg,
                        ).to(self.device)
        else:
            model = gp_model.GPModel_pol(
                            d,
                            p,
                            train_x,
                            train_y,
                            self.likelihood[d][p],
                            self.cfg,
                        ).to(self.device)

        self.mll[d][p] = gpytorch.mlls.ExactMarginalLogLikelihood(
                    self.likelihood[d][p], model
                )

        if warm_start:
            state_dict = copy.deepcopy(self.M[d][p].state_dict())
            model.load_state_dict(state_dict)
        else:
            n_types = self.cfg["model"]["params"]["n_types"]
        
            if p == 0:
                upper_w = self.cfg["model"]["params"]["upper_w"]
                scale_bound = self.scale_state(upper_w)
                length_scale_init = 1*scale_bound * torch.sqrt(torch.tensor(1.*n_types))
            else:
                upper_w = self.cfg["model"]["params"]["upper_w"]
                scale_bound = self.scale_state(upper_w)
                length_scale_init = 0.01*scale_bound * torch.sqrt(torch.tensor(1.*n_types))

            # try:
            #     module_kernel = model.covar_module.kernels[0].base_kernel
            # except:
            #     module_kernel = model.covar_module.base_kernel

            # module_kernel.lengthscale = length_scale_init


        return model

    def set_train_data(self, d, p, warm_start=False):
        # update the training data - last column is the discrete state
        train_sample_rows = self.state_sample_all[
            self.get_d_rows(
                d, drop_non_converged=self.cfg.get("drop_non_converged")
            ),
            :-1,
        ]
        train_sample = train_sample_rows[:,self.get_d_cols(d)].clone().contiguous()
        train_v = self.combined_sample_all[
            self.get_d_rows(
                d, drop_non_converged=self.cfg.get("drop_non_converged")
            ),
            p,
        ].clone().contiguous()

        self.M[d][p] = self.create_model(d, p, train_sample, train_v, warm_start)

        self.optimizer[d][p] = self.create_optimizer(d,p)


        self.M[d][p].set_train_data(
            train_sample,
            train_v,
            strict=False,
        )


        return train_sample, train_v       
    

    def fit_GP(self, dp: List[Tuple] = [()], warm_start=False, howard_iteration=False):
        tasks_per_worker = len(dp)
        if self.cfg.get("distributed"):
            # allocate fitting across workers
            tasks_per_worker = len(dp) / dist.get_world_size()
            worker_slice = [
                A
                for A in range(len(dp))
                if int(A / tasks_per_worker) == dist.get_rank()
            ]
        else:
            worker_slice = [
                A
                for A in range(len(dp))
            ]
        for w, W in enumerate(dp):
            d, p = W

            train_sample, train_v = self.set_train_data(d, p, warm_start)

            # fit first the GP
            self.M[d][p].train()
            self.likelihood[d][p].train()

            training_iter, lr = self.get_solver_config(d,p)
            if howard_iteration:
                training_iter = 0

            if w in worker_slice:
                ll_first_try = 0.
                rel_ll_change,Lmax_ll_grad = self.optimize_ll(training_iter,d,p,train_sample,train_v)

            # set to eval mode
            self.M[d][p].eval()
            self.likelihood[d][p].eval()

        # synchronize results
        if self.cfg.get("distributed"):
            for A, W in enumerate(dp):
                d, p = W
                self.sync_GP(d, p, source_rank=int(A / tasks_per_worker))

        metrics_int = {
            "l2": self.convergence_error(fit_lst = dp,ord=2),
            "l_inf": self.convergence_error(fit_lst = dp),
        }

        logger.info(
            f"Interpolation error: {metrics_int['l_inf']} (L_inf) {metrics_int['l2']} (L2) for state policy pairs {dp}"
        )