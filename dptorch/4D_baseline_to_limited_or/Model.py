import torch
import gpytorch
import numpy as np
from DPGPScipyModel import DPGPScipyModel
from scipy.stats import qmc
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
# ======================================================================

from numpy.linalg import det
from scipy.stats import dirichlet
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import factorial

def sample_on_convex_hull(points, n):
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / factorial(dims)    
    sample = np.random.choice(len(vols), size = n, p = vols / vols.sum())

    return np.einsum('ijk, ij -> ik', deln[sample], dirichlet.rvs([1]*(dims + 1), size = n))

def sample_from_nd_ball(n, r, num_samples=1):
    """
    Draws uniform samples from an n-dimensional ball of radius r.

    Parameters:
        n (int): Dimension of the ball.
        r (float): Radius of the ball.
        num_samples (int): Number of samples to generate.

    Returns:
        numpy.ndarray: An array of shape (num_samples, n) containing the samples.
    """
    samples = []
    for _ in range(num_samples):
        # Step 1: Generate a random point from a standard normal distribution
        point = np.random.normal(0, 1, n)
        
        # Step 2: Normalize the point to lie on the unit sphere
        point /= np.linalg.norm(point)
        
        # Step 3: Scale the radius by a random factor
        radius_scale = np.random.uniform(0, 1) ** (1 / n)
        scaled_point = radius_scale * r * point
        
        samples.append(scaled_point)
    
    return np.array(samples)

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
    trans_mat = model.cfg["model"]["params"]["trans_mat"]
    shock_vals = model.cfg["model"]["params"]["shock_vec"]
    gp_offset = model.cfg["model"]["params"]["GP_offset"]
    beta=model.cfg["model"]["params"]["beta"]
    reg_c=model.cfg["model"]["params"]["reg_c"]
    sigma=model.cfg["model"]["params"]["sigma"]
    lower_w = (model.cfg["model"]["params"]["lower_w"])
    upper_w = (model.cfg["model"]["params"]["upper_w"])
    max_pen = 0 #model.cfg["model"]["params"]["max_penalty"]
    lower_V = model.cfg["model"]["params"]["lower_V"]
    disc_state = state[-1].type(torch.IntTensor)


    sum_tmp = torch.tensor(0.)
    for indxa in range(n_types):
        sum_tmp += trans_mat[disc_state,indxa]*(
            shock_vals[indxa] - inv_utility_ind((1 - beta) * state[model.S[f"w_{indxa+1}"]],reg_c,sigma))

    v_infinity = -gp_offset + sum_tmp
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
                        (control[P[f"u_{indxs+1}"]] + beta*(control[P[f"fut_util_{indxs+1}_{indxs+1}"]]))*trans_mat[indx,indxs]  
                        for indxs in range(n_types) ])
        counter += 1


    #inequality constraints
    #incentive constraints
    for indx_true in range(1,n_types): # true state in
        for indx_false in range(indx_true): # false state in
            G[counter] = control[P[f"u_{indx_true+1}"]] + beta*(control[P[f"fut_util_{indx_true+1}_{indx_true+1}"]]) - \
                (utility_ind(shock_vec[indx_true] + control[P[f"c_{indx_false+1}"]] - shock_vec[indx_false], reg_c, sigma) + beta*(control[P[f"fut_util_{indx_false+1}_{indx_true+1}"]]))
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
        control_dim_loc = 4 * cfg["model"]["params"]["n_types"] + cfg["model"]["params"]["n_types"]**2
        super().__init__(
            V_guess=lambda x: V_INFINITY(
                self,
                x,
            ),
            cfg=cfg,
            policy_names=policy_names,
            state_names=state_names,
            policy_dim=control_dim_loc,
            discrete_state_dim=cfg["model"]["params"].get("discrete_state_dim", 1),
            control_dim=control_dim_loc,
            **kwargs
        )
        n_types = cfg["model"]["params"]["n_types"]    
        self.future_util_diag_mask = torch.zeros(control_dim_loc,dtype=torch.bool)
        self.future_util_mask = torch.zeros(control_dim_loc,dtype=torch.bool)
        for indx in range(n_types):
            self.future_util_diag_mask[self.P[f"fut_util_{indx+1}_{indx+1}"]] = True
            for indx2 in range(n_types):
                self.future_util_mask[self.P[f"fut_util_{indx+1}_{indx2+1}"]] = True


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

    def policy_fit(self):
        n_types = self.cfg["model"]["params"]["n_types"]
        fit_lst = []
        for indxs in range(n_types):
            for indxa1 in range(n_types):
                for indxa2 in range(n_types):
                    fit_lst.append((indxs,1 + self.P[f"fut_util_{indxa1+1}_{indxa2+1}"]))

            for indxa1 in range(n_types):
                fit_lst.append((indxs,1 + self.P[f"c_{indxa1+1}"]))

        self.fit_GP(
            fit_lst,
        )

    def sample(self,no_samples_=None):
        beta = self.cfg["model"]["params"]["beta"]
        n_types = self.cfg["model"]["params"]["n_types"]
        if no_samples_ is None:
            no_samples_ = self.cfg["no_samples"]

        sampler = qmc.Halton(d=n_types, scramble=False)
        
        lower_w = scale_state_func(self.cfg["model"]["params"]["lower_w"],self.cfg)
        upper_w = scale_state_func(self.cfg["model"]["params"]["upper_w"],self.cfg)
        LB_state = torch.zeros(n_types)
        UB_state = torch.zeros(n_types)

        for indxt in range(n_types):
            LB_state[self.S[f"w_{indxt+1}"]] = lower_w[indxt]
            UB_state[self.S[f"w_{indxt+1}"]] = upper_w[indxt]

        if self.epoch == 0 and n_types == 4:

            manual_pts = scale_state_func(torch.tensor([
                [114.953, 115.736, 118.169, 118.559], 
                [143.877, 144.296, 146.012, 146.311],
                [156.07, 156.435, 157.99, 158.267], 
                [183.373, 183.662, 184.955, 185.194], 
                [194.167, 194.915, 200.988, 201.261], 
                [206.651, 206.899, 208.032, 208.246], 
                [220.258, 222.105, 225.33, 225.801], 
                [227.398, 227.618, 228.641, 228.837], 
                [246.327, 246.528, 247.467, 247.649], 
                [261.308, 261.495, 262.378, 262.55], 
                [280.275, 280.448, 281.269, 281.43],
            ]),self.cfg)            
            convex_hull_pts = scale_state_func(torch.tensor([
                [114.953, 115.736, 118.169, 118.559], 
                [261.308, 261.495, 262.378, 262.55], 
                [220.258, 222.105, 225.33, 225.801], 
                [194.167, 194.915, 200.988, 201.261], 
                [152.183, 153.27, 156.737, 158.12], 
                [115.817, 116.569, 118.956, 119.34], 
                [116.985, 117.702, 120.035, 120.413], 
                [143.877, 144.296, 146.012, 146.311],
                [156.07, 156.435, 157.99, 158.267], 
                [183.373, 183.662, 184.955, 185.194], 
                [206.651, 206.899, 208.032, 208.246], 
                [227.398, 227.618, 228.641, 228.837], 
                [246.327, 246.528, 247.467, 247.649], 
                [263.861, 264.046, 264.921, 265.091], 
                [280.275, 280.448, 281.269, 281.43],
            ]),self.cfg)            

            state_sample_01 = torch.from_numpy(sampler.random(n=int(no_samples_)))
            # state_sample_all = torch.unsqueeze(UB_state - LB_state,dim=0)*state_sample_01 + torch.unsqueeze(LB_state,dim=0)
            epsilon = 0.005/(1-beta)
            ball_sample = torch.from_numpy(sample_from_nd_ball(n_types, epsilon, num_samples=no_samples_))
            state_sample_all = ball_sample + torch.from_numpy(sample_on_convex_hull(convex_hull_pts.detach().numpy(), no_samples_))
            state_sample_all = torch.clamp(state_sample_all, LB_state, UB_state)
            state_sample_ = torch.cat((manual_pts, state_sample_all),0)        
        else:
            state_sample_01 = torch.from_numpy(sampler.random(n=int(no_samples_)))

            state_sample_ = torch.unsqueeze(UB_state - LB_state,dim=0)*state_sample_01 + torch.unsqueeze(LB_state,dim=0)

        no_samples = state_sample_.shape[0]

        state_sample = torch.cat(
            (
                state_sample_.repeat_interleave(self.discrete_state_dim,0),
                torch.unsqueeze((torch.arange(self.discrete_state_dim)).repeat(no_samples),-1),
            ),
            dim=1,
        )

        feasible = torch.ones(state_sample.shape[0])
        # test = self.unscale_state(state_sample)
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
        upper_transfer = self.cfg["model"]["params"]["upper_trans"]
        for indxa in range(n_types):
            UB[self.P[f"u_{indxa+1}"]] = utility_ind(upper_transfer, reg_c, sigma)
            UB[self.P[f"c_{indxa+1}"]] = upper_transfer      

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

        for indxp in range(n_pts):
            control = torch.zeros(policy_sample.shape[-1])
            control[:] = policy_sample[indxp,:]
            if indxp == 0: 
                for indxt1 in range(n_types):
                    for indxt2 in range(n_types):
                        control[P[f"fut_util_{indxt1+1}_{indxt2+1}"]] = state[S[f"w_{indxt2+1}"]]

            for indxt in range(n_types): 
                control[P[f"pen_{indxt+1}"]] = 0.
                control[P[f"pen_u_{indxt+1}"]] = 0.

            for indxr in range(n_types):

                control[self.P[f"c_{indxr+1}"]] = inv_utility_ind(control[P[f"u_{indxr+1}"]] ,reg_c,sigma)

            policy_sample[indxp,:] = control[:]

        policy_sample_out = policy_sample
        if not (self.epoch == 1): 
            policy_sample_out[-1,:] = policy


        return policy_sample_out
        
    @torch.no_grad()
    def get_params(self,scaled_state,policy):
        params = {}
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        lower_V = self.cfg["model"]["params"]["lower_V"]

        p_i = torch.unsqueeze(scaled_state[:-1], 0)
        obj_val = (self.M[int(scaled_state[-1].item())][0](p_i).mean[0] + gp_offset)
        params["V_prior"] = obj_val
                     
        params["use_penalty"] = 0.0
        if torch.equal(policy, -123.0*torch.ones_like(policy)):
            params["use_penalty"] = 1.0

        n_types = self.cfg["model"]["params"]["n_types"]
        pen_vec = torch.zeros([2 * n_types])
        for indxt in range(n_types):
            pen_vec[indxt] = policy[self.P[f"pen_{indxt+1}"]]
            pen_vec[n_types + indxt] = policy[self.P[f"pen_u_{indxt+1}"]]

        if not torch.all(pen_vec == 0.0) or self.epoch == 1:
            params["use_penalty"] = 1.0

        if obj_val >= lower_V and self.epoch > 1:
            params["is_feas"] = 1.0
        else:
            params["is_feas"] = 0.0

        return params

    def is_feasible(self, scaled_state,value,control):
        n_types = self.cfg["model"]["params"]["n_types"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"] 
        lower_V = self.cfg["model"]["params"]["lower_V"]      
        gp_offset = self.cfg["model"]["params"]["GP_offset"] 
        min_GP_value = self.cfg["model"]["params"]["min_GP_value"]
        total_pen = sum([ control[self.P[f"pen_{indxt}"]]*pen_opt_vf for indxt in range(1,n_types+1)]) + \
            sum([ control[self.P[f"pen_u_{indxt}"]]*pen_opt_vf for indxt in range(1,n_types+1)])
        if value + gp_offset < min_GP_value:            
            return 0.0
        else:
            return 1.0

    def post_process_optimization(self, scaled_state, params, control, value):
        state = self.unscale_state(scaled_state)
        disc_state = int(state[-1].item())     
        max_points = self.cfg["model"]["params"]["max_points"]
        error_tol = 1.0e-2
        n_types = self.cfg["model"]["params"]["n_types"]
        beta = self.cfg["model"]["params"]["beta"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        pen_base = self.cfg["model"]["params"]["pen_vf"]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]

        upper_V = self.cfg["model"]["params"]["upper_V"]
        lower_V = self.cfg["model"]["params"]["lower_V"]

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

        min_GP_value = self.cfg["model"]["params"]["min_GP_value"]
        out_val_adj = torch.minimum(upper_V, torch.maximum(min_GP_value, out_val)) - gp_offset
        scale_vec = self.scale_policy()
        return control*scale_vec,  out_val_adj


    def u(self, scaled_state, params, control):
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state_in = (state[-1]).type(torch.IntTensor)
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        shock_vals = self.cfg["model"]["params"]["shock_vec"]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        beta = self.cfg["model"]["params"]["beta"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        
        total = torch.tensor(0.)
        for indxa in range(n_types):
            total += (1 - beta) * (trans_mat[disc_state_in,indxa]*(shock_vals[indxa] - control[self.P[f"c_{indxa+1}"]])) - control[self.P[f"pen_{indxa+1}"]]*pen_opt_vf - control[self.P[f"pen_u_{indxa+1}"]]*pen_opt_vf
        
        return total  

    def E_V(self, scaled_state, params, control):
        """Caclulate the expectation of V"""
        state = self.unscale_state(scaled_state)

        e_v_next = 0
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]

        weights, points = self.state_iterate_exp(scaled_state, params, control)
        for i in range(len(weights)):
            p_i = torch.unsqueeze(points[i, :-1], 0)
            obj_val = (self.M[int(points[i, -1].item())][0].predict_mean(p_i) + gp_offset)
            e_v_next += (obj_val)  * weights[i]

        return e_v_next


    def state_next(self, scaled_state, params, control, zpy, opt=False):
        """Return next periods states, given the controls of today and the random discrete realization"""
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        S = self.S

        s = state.clone()
        
        s[-1] = 1.*zpy
        
        for indx in range(n_types):
            s[S[f"w_{indx+1}"]] = control[self.P[f"fut_util_{int(zpy)+1}_{indx+1}"]]

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
                X_L[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (lower_w[indxa2])

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
            X_U[self.P[f"u_{indxa+1}"]] = utility_ind(1 * upper_transfer, reg_c, sigma)
            X_U[self.P[f"pen_{indxa+1}"]] = 10*upper_w[indxa] 
            X_U[self.P[f"pen_u_{indxa+1}"]] = 10*upper_w[indxa]
            X_U[self.P[f"c_{indxa+1}"]] = 1 * upper_transfer
            for indxa2 in range(n_types):

                X_U[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (upper_w[indxa2])

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

        if (self.epoch // self.cfg["BAL"]["epoch_freq"]) % 2 == 1:
            rho = self.cfg["BAL"].get("bal_rho",1.0)
            beta = self.cfg["BAL"].get("bal_beta",10.0)
        else:
            rho = 0
            beta = 1

        # #compute bayesian optimization criteria
        mean_v = self.M[discrete_state][target_p].predict_mean(
                        eval_pt
                    )
        var_v = self.M[discrete_state][target_p].predict_var(
                        eval_pt
                    )

        #Deisenroth criterion
        is_v_ok = (var_v > 0.0000001)
        out_vec = is_v_ok*(rho * (mean_v) + beta / 2.0 * torch.log(var_v + 1e-15)) + torch.logical_not(is_v_ok)*pen_val

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

        out_vec = self.bayesian_opt_criterion(eval_pt,discrete_state,target_p)

        return out_vec


    def BAL(self):

        n_types = self.cfg["model"]["params"]["n_types"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        pen_val = -1.0e10
        dim_state = self.state_sample_all.shape[1]
        torch.manual_seed(self.cfg["seed"] + 77*self.epoch)

        if (self.epoch // self.cfg["BAL"]["epoch_freq"]) % 10000 == 0: 
            if not self.cfg.get("distributed") or dist.get_rank() == 0:
                new_sample = torch.empty((0,self.state_sample_all.shape[1]+1))
                mask_feas = self.combined_sample_all[:,0] + gp_offset > lower_V + 0.4
                state_sample_feas = self.state_sample_all[mask_feas,:]
                combined_sample_feas = self.combined_sample_all[mask_feas,:]
                for indxp in range(state_sample_feas.shape[0]):

                    weighs, next_points = self.state_iterate_exp(state_sample_feas[indxp,:],{},combined_sample_feas[indxp,1:])
                    bal_utility = -1.0e10*torch.ones(next_points.shape[0])
                    for indx_eval in range(next_points.shape[0]):
                        eval_pt = next_points[indx_eval,:]
                        with torch.no_grad():
                            bal_utility[indx_eval] = self.bal_utility_func(eval_pt,0)

                    new_sample = torch.cat(
                        (
                            new_sample,
                            torch.cat(
                                (
                                    next_points,
                                    torch.unsqueeze(bal_utility,dim=1)
                                ),1
                            )        
                        ),0
                    )      

                cand_pts_gather = torch.zeros([0,1 + dim_state])
                for indx_t in range(n_types):
                    mask = new_sample[:,-2] == 1.*indx_t
                    bal_utility = new_sample[mask,-1]
                    max_indx = torch.argsort(bal_utility, descending=True)[ : 1]

                    cand_pts_gather = torch.cat(
                        (
                            cand_pts_gather,
                            new_sample[mask][max_indx,:]
                        ),
                        dim=0,
                    )   

        else:

            self.warm_start = False
            self.fit_GP(self.only_fit_trans_pol(), self.warm_start, compute_metrics=False) 
            n_sim_steps = 1000
            P = self.P
            trans_mat = self.cfg["model"]["params"]["trans_mat"]
            beta = self.cfg["model"]["params"]["beta"]

            lower_w = self.cfg["model"]["params"]["lower_w"]
            upper_w = self.cfg["model"]["params"]["upper_w"]
            LB_state = torch.zeros(dim_state - 1)
            UB_state = torch.zeros(dim_state - 1)

            for indxt in range(n_types):
                LB_state[self.S[f"w_{indxt+1}"]] = lower_w[indxt]
                UB_state[self.S[f"w_{indxt+1}"]] = upper_w[indxt]

            LB_state = scale_state_func(LB_state,self.cfg)
            UB_state = scale_state_func(UB_state,self.cfg)

            out_tmp = torch.zeros([n_types,1 + dim_state])
            out_tmp[:,-1] = pen_val

            for d in range(self.discrete_state_dim):
                self.M[d][0].eval()
                for p1 in range(1,n_types+1):
                    self.M[d][1+P[f"c_{p1}"]].eval()
                    for p2 in range(1,n_types+1):
                        self.M[d][1+P[f"fut_util_{p1}_{p2}"]].eval()


            start_pt = torch.zeros([n_types,dim_state])
            for indxd in range(n_types):

                mask = self.state_sample_all[:, -1] == indxd * torch.tensor(1.)
                V_sample = self.V_sample_all[mask]
                max_ind_opt = torch.argmax(V_sample)

                sample_tmp = self.state_sample_all[mask,:]
                try:
                    obj,point = self.find_max_of_vf(indxd, sample_tmp[max_ind_opt,:-1],LB_state,UB_state)    #run optimization to find maximum of current VF
                    test_pt = sample_tmp[max_ind_opt,:]        
                    test_pt[:-1] = torch.unsqueeze(torch.from_numpy(point),dim=0)
                    self.cfg["model"]["params"]["max_points"][indxd,:] = test_pt[:-1]
                    test_pt[-1] = indxd
                    bal_util = self.bal_utility_func(test_pt,0, pen_val=torch.tensor([pen_val]))
                    out_tmp[indxd,-1] = bal_util[0]
                    out_tmp[indxd,:(dim_state-1)] = test_pt[:-1]
                    out_tmp[indxd,-2] = 1.*indxd
                except:
                    logger.info(f"failed to converge when finding max gp val in state {indxd}")
                    obj = -1e10
                    point = torch.empty(1)

                start_pt[indxd,:-1] = sample_tmp[max_ind_opt,:-1]
                start_pt[indxd,-1] = indxd

            n_bal_sims = self.cfg["BAL"]["n_bal_sims"]
            if self.cfg.get("distributed"):

                worker_slice = [
                    A
                    for A in range(n_bal_sims)
                    if int(A % dist.get_world_size()) == dist.get_rank()
                ]
            else:
                worker_slice = list(range(n_bal_sims))

            for indx_ in worker_slice: 
                torch.manual_seed(105 + indx_ * 4211 + self.epoch*3654)
                indx_type = (indx_%int(n_types))
                current_state = torch.unsqueeze(start_pt[indx_type,:],dim=0)

                for indxt in range(n_sim_steps):
                    if indx_ < n_types and indxt < n_sim_steps/2:
                        current_state[0,-1] = 1.*indx_type

                    current_disc_state = int(current_state[0,-1].item())

                    #compute state transition
                    pol_out = torch.zeros(self.control_dim)
                    params = self.get_params(current_state[0,:],torch.zeros([self.combined_sample_all.shape[1]-1]))
                    try:
                        for p1 in range(1,n_types+1):

                            for p2 in range(1,n_types+1):
                                with torch.no_grad():
                                    next_pol_tmp = self.M[current_disc_state][1+P[f"fut_util_{p1}_{p2}"]].predict_mean(current_state[:,:-1])

                                pol_out[P[f"fut_util_{p1}_{p2}"]] = next_pol_tmp[0]
                    except:
                        logger.info(f"Evaluation failed in state {current_disc_state}; aborting simulation.")
                        break

                    with torch.no_grad():
                        bal_util = self.bal_utility_func(current_state,0, pen_val=torch.tensor([pen_val]))
                        v = self.M[current_disc_state][0].predict_mean(current_state[:,:-1])

                    if bal_util > out_tmp[current_disc_state,-1]:
                        out_tmp[current_disc_state,:-1] = current_state[0,:]
                        out_tmp[current_disc_state,-1] = bal_util[0]

                    if (v < lower_V - gp_offset and indxt > 0 and self.epoch > n_rand_its):                    
                        print(f"Break simulation in iteration {indxt} with value {v + gp_offset} at point {current_state[0,:]}")
                        break

                    cat_dist = torch.distributions.categorical.Categorical(trans_mat[current_disc_state,:])
                    next_disc_state = int((cat_dist.sample()).item())
                    scale_vec = self.scale_policy()
                    current_state = torch.unsqueeze(self.state_next(current_state[0,:], params, pol_out / scale_vec, next_disc_state),0)
                    current_state[:,:-1] = torch.maximum( #project simulation back into bounds
                        LB_state,
                        torch.minimum(
                            UB_state,
                            current_state[:,:-1]
                            ))

            #gather simulation results on rank 0
            if self.cfg.get("distributed"):
                cand_pts_gather = (
                    torch.cat(self.gather_tensors(out_tmp))
                    .clone()
                    .detach()
                    .to(self.device)
                )
            else:
                cand_pts_gather = out_tmp.clone().detach().to(self.device)

        #we only compute and return the results for rank 0
        if not self.cfg.get("distributed") or dist.get_rank() == 0:
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
            final_bal_util = torch.zeros([n_pts])
            indxp = 0
            for indxd in range(self.cfg["model"]["params"]["discrete_state_dim"]):
                if cand_pts[indxd,-1] > pen_val:
                    out[indxp,:] = cand_pts[indxd,:-1]
                    final_bal_util[indxp] = cand_pts[indxd,-1]
                    indxp+=1


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
                        -123.0 * torch.ones([new_sample.shape[0],1+self.policy_dim])                
                ),
                dim=0,
            )
            logger.info(f"BAL added points {out} with final bal util {final_bal_util}")
        
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

            # fit pol and vf in first iteration
            self.fit_GP(fit_lst)
            self.save()
            self.warm_start = False

        else:
            fit_lst = self.only_fit_VF()
            

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
        while indx_it < self.cfg["model"]["params"]["n_Howard_steps"] and error_old > 1e-7 and self.epoch > max(100,self.cfg["BAL"]["epoch_freq"]):

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
        model = gp_model.GPModel(
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

        if p > 0:
            train_vf = self.combined_sample_all[
                self.get_d_rows(
                    d, drop_non_converged=self.cfg.get("drop_non_converged")
                ),
                0,
            ].clone()                
            lower_V = self.cfg["model"]["params"]["lower_V"]
            gp_offset = self.cfg["model"]["params"]["GP_offset"]     
            mask_feas = train_vf + gp_offset >= lower_V
            train_v = train_v[mask_feas].contiguous()
            train_sample = train_sample[mask_feas,:].contiguous()
        if p == 0:
            n_zeropts_keep = 50
            mask_zero = (train_v == 0.0)
            mask_nonzero = torch.logical_not(mask_zero)
            nonzero_samples = train_sample[mask_nonzero,...]
            nonzero_train_v = train_v[mask_nonzero]
            zero_samples = train_sample[mask_zero,...][-n_zeropts_keep:,...]
            zero_train_v = train_v[mask_zero][-n_zeropts_keep:]

            train_sample = torch.cat((
                nonzero_samples,
                zero_samples
            ),dim=0).contiguous()
            train_v = torch.cat((
                nonzero_train_v,
                zero_train_v
            ),dim=0).contiguous()

        self.M[d][p] = self.create_model(d, p, train_sample, train_v, warm_start)

        self.optimizer[d][p] = self.create_optimizer(d,p)

        self.M[d][p].set_train_data(
            train_sample,
            (train_v - self.M[d][p].offset_y)/self.M[d][p].scale_y,
            strict=False,
        )

        return train_sample, train_v        

    def fit_GP(self, dp: List[Tuple] = [()], warm_start=False, howard_iteration=False, compute_metrics=True):
        world_size = dist.get_world_size() if self.cfg.get("distributed") else 1
        if self.cfg.get("distributed"):
            # allocate fitting across workers
            worker_slice = [
                A
                for A in range(len(dp))
                if (A % world_size) == dist.get_rank()
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
                rel_ll_change,Lmax_ll_grad = self.optimize_ll(training_iter,d,p,train_sample,(train_v - self.M[d][p].offset_y)/self.M[d][p].scale_y)

            # set to eval mode
            self.M[d][p].eval()
            self.likelihood[d][p].eval()

        # synchronize results
        if self.cfg.get("distributed"):
            for A, W in enumerate(dp):
                d, p = W
                self.sync_GP(d, p, source_rank=(A % world_size))

        if compute_metrics:
            metrics_int = {
                "l2": self.convergence_error(fit_lst = dp,ord=2),
                "l_inf": self.convergence_error(fit_lst = dp),
            }

            logger.info(
                f"Interpolation error: {metrics_int['l_inf']} (L_inf) {metrics_int['l2']} (L2) for state policy pairs {dp}"
            )

    @torch.no_grad()
    def convergence_error(self, ord=float("inf"), fit_lst = None):
        if fit_lst is None:
            fit_lst = self.what_to_fit()
        err = torch.zeros(len(fit_lst))
        with torch.no_grad():
            for indxd in range(len(fit_lst)):
                d,p = fit_lst[indxd][0], fit_lst[indxd][1]
                mask = self.state_sample_all[:, -1]  == 1.*d
                eval_pts = self.state_sample_all[mask, :-1]
                tmp_vec = (self.M[d][p].predict_mean(
                    (eval_pts[:,self.get_d_cols(d)])
                            ))
                tmp_vec_target = self.combined_sample_all[mask,p]
                err[indxd] = torch.linalg.norm(
                    (
                        tmp_vec - tmp_vec_target
                    )/(1+torch.abs(tmp_vec_target)),ord=ord)

                scale = (
                    1.0
                    if ord == float("inf")
                    else 1 / max(1,self.state_sample_all[mask,:].shape[0])
                )
                err[indxd] = scale * err[indxd]
        return err              