import torch
import gpytorch
import numpy as np
from DPGPIpoptModel import DPGPIpoptModel,IpoptModel
from DPGPScipyModel import DPGPScipyModel, ScipyModel
from scipy.stats import qmc
import cyipopt
import copy
from typing import List, Tuple
import logging
from math import ceil
from scipy import optimize

from Utils import NonConvergedError
import os

if os.getenv("OMPI_COMM_WORLD_SIZE"):
    import torch.distributed as dist

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

from numpy.linalg import det
from scipy.stats import dirichlet
from scipy.spatial import ConvexHull, Delaunay


def sample_on_convex_hull(points, n):
    dims = points.shape[-1]
    hull = points[ConvexHull(points).vertices]
    deln = hull[Delaunay(hull).simplices]

    vols = np.abs(det(deln[:, :dims, :] - deln[:, dims:, :])) / np.math.factorial(dims)    
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

reg_val = 0.000001
def scale_state_func(unscaled_state,cfg):
    beta = cfg["model"]["params"]["beta"]
    n_types = cfg["model"]["params"]["n_types"]
    sigma = cfg["model"]["params"]["sigma"]
    reg_c = cfg["model"]["params"]["reg_c"]
    upper_w = cfg["model"]["params"]["upper_w"]
    lower_w = cfg["model"]["params"]["lower_w"]
    state_scaled = unscaled_state.clone()
    # state_scaled[...,:n_types] = inv_utility_ind(unscaled_state[...,:n_types] * (1-beta),reg_c,sigma) 
    # state_scaled[...,:] = torch.log(unscaled_state[...,:] * (1-beta) + reg_val)
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
    # unscaled_state[...,:n_types] = utility_ind(state_scaled[...,:n_types],reg_c,sigma) / (1-beta)
    # unscaled_state[...,:] = (torch.exp(state_scaled[...,:]) - reg_val) / (1-beta)
    unscaled_state[...,:] = state_scaled[...,:] / (1-beta)
    return unscaled_state

# ======================================================================


# V infinity
def V_INFINITY(model, scaled_state):
    state = model.unscale_state(scaled_state)
    n_types=model.cfg["model"]["params"]["n_types"]
    trans_mat = model.cfg["model"]["params"]["trans_mat"]
    shock_vals = model.cfg["model"]["params"]["shock_vec"]
    beta=model.cfg["model"]["params"]["beta"]
    reg_c=model.cfg["model"]["params"]["reg_c"]
    sigma=model.cfg["model"]["params"]["sigma"]
    lower_w = (model.cfg["model"]["params"]["lower_w"])
    upper_w = (model.cfg["model"]["params"]["upper_w"])
    lower_V = model.cfg["model"]["params"]["lower_V"]
    disc_state = state[-1].type(torch.IntTensor)

    # max_val = (0.5 * (upper_w[disc_state] - lower_w[disc_state]))**2
    # adjust_fac = (0.5 * shock_vals[disc_state] - (lower_V - max_pen)) / max_val
    # sum_tmp = torch.tensor(0.5 * shock_vals[disc_state])
    # for indxa in range(n_types):
    #     sum_tmp -=  trans_mat[disc_state,indxa]*(
    #          (0.5*(lower_w[disc_state]+upper_w[disc_state]) - state[model.S[f"w_{indxa+1}"]])**2) * adjust_fac



    sum_tmp = torch.tensor(0.)
    for indxa in range(n_types):
        # init guess for policy is beta**t * w; i.e. decaying promise
        # for u(c) = u**(1-sigma) we get u_inv = u**(1/(1-sigma))
        # sum_t beta**t * u_inv(beta**t * w) = u_inv(w) / (1 - beta**((2 - sigma)/(1 - sigma)))
        sum_tmp += trans_mat[disc_state,indxa]*(
            shock_vals[indxa] - inv_utility_ind((1 - beta) * state[model.S[f"w_{indxa+1}"]],reg_c,sigma))


    gp_offset = model.cfg["model"]["params"]["GP_offset"]
    mult = 1.
    if model.epoch == 0:
        mult = 0.
    
    v_infinity = -gp_offset + sum_tmp * mult
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
    future_util_diag_mask = model.future_util_diag_mask
    future_util_mask = model.future_util_mask
    submatrix_mask = model.submatrix_mask

    # M = 2*n_types + n_types - 1 + n_types  # number of constraints
    M = 2*n_types + n_types*(n_types-1) // 2 #+ n_types # number of constraints
    disc_state = state[-1].type(torch.IntTensor)

    counter = 0

    # G = torch.empty(M)

    # #equality constraints
    # for indxs in range(n_types):
    #     G[counter] = utility_ind(control[P[f"c_{indxs+1}"]], reg_c, sigma) - control[P[f"u_{indxs+1}"]]
    #     counter += 1
    
    # # promise keeping
    # for indx in range(n_types):
    #     G[counter] = state[S[f"w_{indx+1}"]] + control[P[f"pen_{indx+1}"]] - control[P[f"pen_u_{indx+1}"]] - \
    #             sum(
    #                 [
    #                     (control[P[f"u_{indxs+1}"]] + beta*(control[P[f"fut_util_{indxs+1}_{indxs+1}"]]))*trans_mat[indx,indxs]  
    #                     for indxs in range(n_types) ])

    #     counter += 1

    # # inequality constraints
    # # incentive constraints
    # for indx_true in range(1,n_types): # true state in
    #     for indx_false in range(indx_true): # false state in
    #         G[counter] = control[P[f"u_{indx_true+1}"]] + beta*(state[S[f"w_{indx_true+1}"]] + control[P[f"fut_util_{indx_true+1}_{indx_true+1}"]]) - \
    #             (utility_ind(shock_vec[indx_true] + control[P[f"c_{indx_false+1}"]] - shock_vec[indx_false], reg_c, sigma) + beta*(state[S[f"w_{indx_false+1}"]] + control[P[f"fut_util_{indx_false+1}_{indx_true+1}"]]))
    #         G[counter] = control[P[f"u_{indx_true+1}"]] + beta*(control[P[f"fut_util_{indx_true+1}_{indx_true+1}"]]) - \
    #             (utility_ind(shock_vec[indx_true] + control[P[f"c_{indx_false+1}"]] - shock_vec[indx_false], reg_c, sigma) + beta*(control[P[f"fut_util_{indx_false+1}_{indx_true+1}"]]))
    #         assert shock_vec[indx_true] - shock_vec[indx_false] >= 0, "shock difference needs to be non-negative"
    #         counter += 1

    # return G

    G_ = torch.empty(M)

    #equality constraints    
    out_tmp = utility_ind(control[P[f"c_{1}"] : P[f"c_{1}"] + n_types], reg_c, sigma) - control[P[f"u_{1}"] : P[f"u_{1}"] + n_types]
    G_[:n_types] = out_tmp
    counter = n_types

    G_[counter : counter + n_types] = state[S[f"w_{1}"] : S[f"w_{1}"] + n_types] + control[P[f"pen_{1}"] : P[f"pen_{1}"] + n_types] - control[P[f"pen_u_{1}"] : P[f"pen_u_{1}"] + n_types] - \
        torch.sum(torch.unsqueeze(control[P[f"u_{1}"]:P[f"u_{1}"]+n_types] + beta*(control[future_util_diag_mask]),0)*trans_mat[:,:],-1)

    # G_[counter : counter + n_types] = state[S[f"w_{1}"] : S[f"w_{1}"] + n_types] + control[P[f"pen_{1}"] : P[f"pen_{1}"] + n_types] - control[P[f"pen_u_{1}"] : P[f"pen_u_{1}"] + n_types] - \
    #     torch.sum(torch.unsqueeze(control[P[f"u_{1}"]:P[f"u_{1}"]+n_types] + beta*(state[S[f"w_{1}"] : S[f"w_{1}"] + n_types] + control[future_util_diag_mask]),0)*trans_mat[:,:],-1)


    counter += n_types


    fut_util_mat = torch.reshape(control[future_util_mask],(n_types,n_types))
    mat_tmp = torch.unsqueeze(control[P[f"u_{1}"]:P[f"u_{1}"]+n_types] + beta*(control[future_util_diag_mask]),0) - \
        (utility_ind(torch.abs(torch.unsqueeze(shock_vec,0) + torch.unsqueeze(control[P[f"c_{1}"]:P[f"c_{1}"]+n_types],-1) - torch.unsqueeze(shock_vec,-1)), reg_c, sigma) + beta*(fut_util_mat))

    # fut_util_mat = torch.unsqueeze(state[S[f"w_{1}"] : S[f"w_{1}"] + n_types],0) + torch.reshape(control[future_util_mask],(n_types,n_types))
    # mat_tmp = torch.unsqueeze(control[P[f"u_{1}"]:P[f"u_{1}"]+n_types] + beta*(state[S[f"w_{1}"] : S[f"w_{1}"] + n_types] + control[future_util_diag_mask]),0) - \
    #     (utility_ind(torch.abs(torch.unsqueeze(shock_vec,0) + torch.unsqueeze(control[P[f"c_{1}"]:P[f"c_{1}"]+n_types],-1) - torch.unsqueeze(shock_vec,-1)), reg_c, sigma) + beta*(fut_util_mat))

    G_[counter : ] = mat_tmp[submatrix_mask]

    # assert torch.abs(torch.sum(G-G_)).item() < 1e-6, "Equality constraints do not match"

    # G = G_

    return G_


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
            policy_dim=4 * cfg["model"]["params"]["n_types"] + cfg["model"]["params"]["n_types"]**2,
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

        self.submatrix_mask = torch.zeros((n_types,n_types),dtype=torch.bool)
        for indx_true in range(1,n_types): # true state in
            for indx_false in range(indx_true): # false state in
                self.submatrix_mask[indx_false,indx_true] = True

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

    def get_solver_config(self,d,p):
        if p > 0:
            training_iter = self.cfg["torch_optim"].get("iter_per_cycle_pol",100)
            lr = self.cfg["torch_optim"].get("lr_pol",1e-3)
        else:
            training_iter = self.cfg["torch_optim"].get("iter_per_cycle_vf",100)
            lr = self.cfg["torch_optim"].get("lr_vf",1e-3)


        return training_iter, lr

    def what_pol_to_fit(self): #per default only fit the VF in each iteration; customize if we need some policies
        n_types = self.cfg["model"]["params"]["n_types"]
        fit_lst = []
        for indxs in range(n_types):
            for indxa1 in range(n_types):
                for indxa2 in range(n_types):
                    fit_lst.append((indxs,self.P[f"fut_util_{indxa1+1}_{indxa2+1}"]))
        return fit_lst

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

        if self.epoch == 0:
            #find n_types + 1 points in stationary subset of feas set and then uniformly sample on its convex hull
            manual_pts = scale_state_func(torch.tensor([[3.6789,4.55191,5.29423,5.86329,6.33018,6.7339,7.09424,7.4197,7.70555,7.90964],
                [17.8707,17.9799,18.1235,18.2738,18.4239,18.5721,18.7176,18.8589,18.9901,19.0876],
                [11.0892,15.1522,16.7708,17.2927,17.5364,17.7081,17.8595,18.0023,18.1339,18.2315],
                [8.72748,10.7201,14.7194,16.2411,16.7366,16.9717,17.1391,17.286,17.4186,17.5163],
                [7.682,8.82175,10.6903,14.3609,15.768,16.2328,16.458,16.6194,16.7557,16.8543],
                [4.86405,5.65632,6.36586,6.92807,7.40599,7.86719,8.46194,9.72167,12.5775,13.5992],
                [3.73645,4.58045,5.3111,5.87671,6.34245,6.74568,7.10575,7.43103,7.71674,7.92075],
                [4.6935,5.56674,6.31001,6.88283,7.36468,7.82781,8.42401,9.68594,12.5464,13.5696],
                [4.9417,5.70207,6.39721,6.95426,7.42996,7.88992,8.48363,9.7417,12.5942,13.6147],
                [3.80971,4.61967,5.33607,5.89716,6.36118,6.7636,7.12318,7.44812,7.73358,7.93743],
                [4.7651,5.64009,6.39076,6.99308,7.59208,8.52035,10.3204,11.0865,11.4074,11.5719],
                [3.79195,4.60991,5.32969,5.89189,6.35639,6.75904,7.11874,7.44377,7.72927,7.93316],
                [6.96523,7.27926,7.6603,8.02747,8.36828,8.68446,8.97919,9.25321,9.49875,9.67638],
                [6.945,7.32655,7.77295,8.19219,8.58296,8.98154,9.51505,10.6674,13.2933,14.2374],
                [7.77358,8.8664,10.7149,14.377,15.7816,16.2457,16.4708,16.6321,16.7684,16.867],
                [6.96675,7.90389,8.91695,10.565,13.9958,15.3195,15.7611,15.977,16.1269,16.2288],
                [4.73607,5.39361,6.03626,6.56904,7.02287,7.43116,7.83969,8.37581,9.51199,12.0021],
                [3.88512,4.66301,5.36531,5.92155,6.3836,6.785,7.14393,7.46841,7.75352,7.95714],
                [4.17887,5.05194,5.79448,6.3644,6.83476,7.25225,7.66723,8.20965,9.3569,11.8701],
                [5.77501,6.65179,7.41002,8.04245,8.76105,10.1642,13.2722,14.4785,14.877,15.0377],
                [6.35034,7.23886,8.04666,8.87665,10.3798,13.6312,14.8911,15.313,15.5146,15.6287],
                [4.24322,5.08388,5.81339,6.37948,6.84854,7.26545,7.68007,8.22219,9.36892,11.881],
                [9.89532,10.1208,10.4068,10.6958,10.9814,11.2859,11.7048,12.6143,14.6824,15.4378],
                [3.69348,4.55893,5.29825,5.86644,6.33306,6.73668,7.09696,7.42239,7.7082,7.91228],
                [6.55072,6.97248,7.45406,7.89778,8.30595,8.71831,9.26636,10.4457,13.1305,14.0945],
                [7.9005,8.36284,8.92692,9.62768,11.0097,14.0623,15.252,15.6563,15.8535,15.9666],
                [5.22462,6.09855,6.84475,7.42925,7.9575,8.60482,9.93232,12.9182,14.0694,14.4084],
                [5.11095,5.59313,6.12227,6.5936,7.00839,7.37959,7.71709,8.02535,8.29806,8.49366],
                [7.06299,7.37151,7.74695,8.10967,8.44706,8.7605,9.05301,9.32521,9.56927,9.74592],
                [4.75451,5.59685,6.32771,6.89686,7.37748,7.84004,8.43585,9.69721,12.5565,13.5794],
                [8.76702,9.23558,9.95559,11.3839,14.5086,15.7255,16.1402,16.3493,16.4975,16.599],
                [3.79899,4.61375,5.33221,5.89395,6.35825,6.7608,7.12046,7.44546,7.73096,7.93484],
                [3.75451,4.5898,5.3169,5.88142,6.34676,6.74981,7.10977,7.43499,7.72065,7.92462],
                [6.12012,6.49233,6.9295,7.33928,7.712,8.05276,8.36705,8.65699,8.9153,9.10144],
                [3.72347,4.5738,5.30705,5.87346,6.33947,6.74283,7.10297,7.4283,7.71405,7.91808],
                [3.68471,4.55468,5.29581,5.86453,6.33131,6.735,7.09531,7.42076,7.70659,7.91068],
                [8.79238,9.05796,9.38876,9.71713,10.0367,10.3736,10.8352,11.8414,14.1369,14.9694],
                [4.14795,4.83276,5.48987,6.0283,6.48203,6.87869,7.23445,7.55661,7.83998,8.04249],
                [8.84208,9.15482,9.54488,9.96252,10.5153,11.6819,14.3206,15.3596,15.7161,15.8669],
                [14.5572,14.6925,14.8694,15.0531,15.2352,15.4137,15.5877,15.7556,15.9108,16.0255],
                [3.75243,4.58874,5.31625,5.88093,6.34631,6.74936,7.10932,7.43455,7.72022,7.9242],
                [3.6874,4.55597,5.29655,5.86511,6.33184,6.7355,7.09581,7.42125,7.70708,7.91117],
                [5.83632,6.68187,7.4275,8.05616,8.7734,10.1756,13.2821,14.4879,14.8862,15.0469],
                [9.05957,12.3074,13.6468,14.1249,14.3825,14.5812,14.7615,14.9322,15.089,15.2048],
                [3.84868,4.64172,5.35076,5.90937,6.37242,6.77435,7.13359,7.4583,7.74358,7.94733],
                [4.73283,5.27715,5.85067,6.34786,6.77871,7.16079,7.50627,7.82067,8.09811,8.2968],
                [4.80574,5.62389,6.34461,6.91055,7.38997,7.85195,8.44731,9.70799,12.5659,13.5883],
                [6.59634,7.37088,8.12983,8.94305,10.4368,13.6762,14.9319,15.3528,15.5542,15.6682],
                [3.78672,4.66204,5.41413,6.02211,6.59504,7.0241,7.37929,7.69446,7.9708,8.16843],
                [5.42388,5.93225,6.48094,6.96467,7.39012,7.77967,8.17379,8.69505,9.80425,12.2377],
                [8.18188,8.54381,8.98256,9.43994,10.0319,11.2651,14.0443,15.1336,15.5027,15.6564],
                [4.80396,5.3353,5.89998,6.3922,6.82003,7.20013,7.54417,7.85747,8.13407,8.3322],
                [5.33195,5.85599,6.41578,6.90596,7.3354,7.7277,8.12398,8.6475,9.76089,12.2032],
                [7.21336,8.0352,8.99818,10.6264,14.0409,15.3592,15.7994,16.0149,16.1648,16.2666],
                [4.80537,5.74417,6.48824,7.0466,7.51587,7.96919,8.55589,9.80195,12.629,13.641],
                [5.14303,6.52335,7.32167,7.81088,8.19754,8.54643,8.90308,9.38123,10.4063,12.658],
                [5.97292,6.35774,6.80639,7.22441,7.60305,7.94824,8.26598,8.55869,8.81919,9.00679],
                [5.17899,5.73148,6.31064,6.81165,7.24764,7.64437,8.04412,8.57126,9.69127,12.1475],
                [5.75544,6.41868,7.06761,7.61606,8.12789,8.7643,10.0773,13.0349,14.176,14.5126],
                [9.89707,10.112,10.3854,10.6615,10.9294,11.1926,11.4727,11.8529,12.6605,14.4189],
                [5.71659,6.12567,6.59569,7.02861,7.41777,7.77071,8.09445,8.39192,8.65619,8.84628],
                [4.24181,4.89896,5.54121,6.07303,6.52338,6.918,7.27237,7.59349,7.87606,8.07806],
                [4.04884,4.76579,5.43927,5.98454,6.44163,6.84027,7.19738,7.52052,7.80463,8.00762],
                [3.76681,4.68669,5.42932,5.98791,6.44593,6.8431,7.19856,7.52031,7.80332,8.00559],
                [4.52268,5.10975,5.711,6.22297,6.66243,7.05019,7.39974,7.71723,7.99704,8.19726],
                [9.5956,11.2312,14.9609,16.3974,16.8714,17.1012,17.2672,17.4137,17.5462,17.644],
                [5.35627,5.87608,6.43289,6.92136,7.34974,7.74131,8.13703,8.65997,9.77226,12.2122],
                [8.49593,8.7421,9.05095,9.35812,9.65062,9.92737,10.1893,10.4358,10.6587,10.8209],
                [5.51241,6.43893,7.65657,9.59543,10.4326,10.8129,11.0714,11.2912,11.4857,11.6272],
                [8.40032,8.66494,8.99424,9.31951,9.6292,9.92903,10.245,10.6745,11.5966,13.6188],
                [9.64038,9.85294,10.1235,10.3965,10.6599,10.9117,11.1522,11.38,11.5873,11.7388],
                [4.50175,5.23449,5.91591,6.46515,6.92714,7.34028,7.75237,8.29203,9.43447,11.9377],
                [5.45345,6.22168,6.92304,7.49295,8.01546,8.65944,9.98283,12.9609,14.1093,14.4476],
                [4.18804,5.05633,5.79698,6.36637,6.83655,7.25397,7.66891,8.2113,9.35849,11.8715],
                [3.80471,4.61691,5.33425,5.89563,6.35978,6.76227,7.12188,7.44686,7.73233,7.9362],
                [8.68786,8.98003,9.34101,9.70232,10.0797,10.5878,11.6812,14.172,15.1467,15.4439],
                [8.17393,9.10416,10.8652,14.4667,15.8516,16.3107,16.5345,16.6956,16.8318,16.9304],
                [4.2262,4.88776,5.53245,6.06541,6.51637,6.91135,7.26595,7.58722,7.86993,8.07201],
                [3.69367,4.55902,5.2983,5.86649,6.3331,6.73672,7.097,7.42242,7.70824,7.91232],
                [7.76959,8.1223,8.54398,8.9536,9.37108,9.92244,11.0976,13.7683,14.8073,15.1197],
                [4.01773,4.74546,5.42424,5.97164,6.42973,6.82896,7.18644,7.50986,7.79419,7.99731],
                [3.68631,4.55546,5.29625,5.86487,6.33163,6.7353,7.0956,7.42105,7.70688,7.91097],
                [8.07762,8.33908,8.66483,8.98653,9.29108,9.57789,9.84836,4.86446,6.87322,7.15439],
                [10.484,10.6774,10.9253,11.1775,11.4225,11.6583,11.8845,5.70337,8.07501,8.20778],
                [12.4149,12.5754,12.7836,12.9981,13.2088,13.4138,13.6123,6.43371,9.11528,9.14047],
                [14.0791,14.2193,14.4024,14.5922,14.78,14.9638,15.1428,7.08915,10.0468,9.98639],
                [15.5653,15.6914,15.8566,16.0287,16.1998,16.3678,16.5321,7.6889,10.8983,10.766],
                [16.9212,17.0367,17.1885,17.3471,17.5052,17.661,17.8137,8.24513,11.6875,11.4929],
                [18.1763,18.2835,18.4247,18.5725,18.7202,18.8661,19.0094,8.76612,12.4265,12.1764]]),self.cfg)
            state_sample_ch = torch.from_numpy(sample_on_convex_hull(manual_pts.detach().numpy(), self.cfg["no_feas_samples"]))
            state_sample_01 = torch.from_numpy(sampler.random(n=int(no_samples_)))
            # state_sample_all = torch.unsqueeze(UB_state - LB_state,dim=0)*state_sample_01 + torch.unsqueeze(LB_state,dim=0)
            epsilon = 0.05/(1-beta)
            ball_sample = torch.from_numpy(sample_from_nd_ball(n_types, epsilon, num_samples=no_samples_))
            state_sample_all = ball_sample + torch.from_numpy(sample_on_convex_hull(manual_pts.detach().numpy(), no_samples_))
            # state_sample_all = ((epsilon)*state_sample_01 - 0.5 * epsilon) + torch.from_numpy(sample_on_convex_hull(manual_pts.detach().numpy(), no_samples_))
            state_sample_all = torch.minimum(
                UB_state,
                torch.maximum(
                    LB_state,
                    state_sample_all
                ),
            )
            state_sample_ = torch.cat((state_sample_all,state_sample_ch),0)
        
        else:
            #sample on [0,1] and later scale to right shape
            state_sample_01 = torch.from_numpy(sampler.random(n=int(no_samples_)))

            # manual_pts_bound_low = torch.ones([n_types,n_types]) * 0.5 * (upper_w + lower_w) + torch.eye(n_types) * (0.5 * lower_w - 0.5 * upper_w)
            # center_w = 0.5 * (upper_w + lower_w)
            # facet_pts = torch.eye(n_types) * lower_w + (torch.ones([n_types,n_types]) - torch.eye(n_types)) * center_w
            state_sample_ = torch.unsqueeze(UB_state - LB_state,dim=0)*state_sample_01 + torch.unsqueeze(LB_state,dim=0)

        no_samples = state_sample_.shape[0]

        #copy sample points for each state and add the corresponding discrete state in the last entry of each row
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

        # val_lst = torch.zeros(n_pts)
        for indxp in range(n_pts):
            control = torch.zeros(policy_sample.shape[-1])
            control[:] = policy_sample[indxp,:]
            if indxp == 0:  #first batch assume we stay at current util
                for indxt1 in range(n_types):
                    for indxt2 in range(n_types):
                        # control[P[f"fut_util_{indxt1+1}_{indxt2+1}"]] = 0.
                        control[P[f"fut_util_{indxt1+1}_{indxt2+1}"]] = state[S[f"w_{indxt2+1}"]]

            control[P[f"pen_{1}"]:P[f"pen_{1}"]+n_types] = 0.
            control[P[f"pen_u_{1}"]:P[f"pen_u_{1}"]+n_types] = 0.

            control[self.P[f"c_{1}"]:self.P[f"c_{1}"]+n_types] = inv_utility_ind(control[P[f"u_{1}"]:P[f"u_{1}"]+n_types] ,reg_c,sigma)

            policy_sample[indxp,:] = control[:]
        
        policy_sample_out = policy_sample#[indx_lst[:n_restarts],:]
        if not (self.epoch == 1): 
            policy_sample_out[-1,:] = policy

        if not self.cfg.get("DISABLE_POLICY_FIT"):
            pol_inter = torch.zeros(policy_sample.shape[-1])
            eval_pt = torch.unsqueeze(state[:-1],dim=0)
            for indxp in range(policy_sample.shape[-1]):
                try:
                    pol_inter[indxp] = torch.minimum(
                        UB[indxp],
                        torch.maximum(
                            LB[indxp],
                            self.M[disc_state][1 + indxp](eval_pt).mean
                        ),
                    )
                    policy_sample_out[-2,indxp] = pol_inter[indxp]
                except:
                    logger.info(f"state {disc_state} policy {indxp} evaluation failed in sample_stps.")

        return policy_sample_out
        
    @torch.no_grad()
    def get_params(self,scaled_state,policy):
        params = {}
        disc_state = scaled_state[-1].type(torch.IntTensor)
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        lower_V = self.cfg["model"]["params"]["lower_V"]

        p_i = torch.unsqueeze(scaled_state[:-1], 0)
        obj_val = (self.M[int(scaled_state[-1].item())][0](p_i).mean[0] + gp_offset)
        params["V_prior"] = obj_val
        if obj_val >= lower_V:
            params["is_feas"] = 1.0
        else:
            params["is_feas"] = 0.0

        return params

    def is_feasible(self, scaled_state,value,control):
        lower_V = self.cfg["model"]["params"]["lower_V"]  
        disc_state = scaled_state[-1].type(torch.IntTensor)
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        if value <= 0.:
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
        pen_cfg = self.cfg["model"]["params"]["pen_vf"]

        upper_V = self.cfg["model"]["params"]["upper_V"]
        lower_V = self.cfg["model"]["params"]["lower_V"]

        total_pen = sum([ control[self.P[f"pen_{indxt}"]] for indxt in range(1,n_types+1)]) + \
            sum([ control[self.P[f"pen_u_{indxt}"]] for indxt in range(1,n_types+1)])
        
        if self.epoch == 0:
            pen_vf = 100000.
        else:
            pen_vf = pen_cfg

        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        out_val = value + total_pen * pen_opt_vf
        if total_pen * pen_opt_vf >= error_tol:

            out_val -= total_pen * pen_vf
                
        else:
            control[self.P[f"pen_{1}"]:self.P[f"pen_{1}"]+n_types] = 0.
            control[self.P[f"pen_u_{1}"]:self.P[f"pen_u_{1}"]+n_types] = 0.

        out_val_adj = torch.minimum(upper_V - gp_offset, torch.maximum(torch.tensor(0.), out_val - gp_offset))

        return control,  out_val_adj


    def u(self, scaled_state, params, control):
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state_in = (state[-1]).type(torch.IntTensor)
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        shock_vals = self.cfg["model"]["params"]["shock_vec"]
        beta = self.cfg["model"]["params"]["beta"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]
        
        total = torch.tensor(0.)
        # for indxa in range(n_types):
        #     total += (1 - beta) * (trans_mat[disc_state_in,indxa]*(shock_vals[indxa] - control[self.P[f"c_{indxa+1}"]])) - control[self.P[f"pen_{indxa+1}"]]*pen_opt_vf - control[self.P[f"pen_u_{indxa+1}"]]*pen_opt_vf

        total = torch.sum((1 - beta) * (
            trans_mat[disc_state_in,:]*(
                shock_vals[:] - control[self.P[f"c_{1}"]:self.P[f"c_{1}"]+n_types]))
            - control[self.P[f"pen_{1}"]:self.P[f"pen_{1}"]+n_types]*pen_opt_vf - control[self.P[f"pen_u_{1}"]:self.P[f"pen_u_{1}"]+n_types]*pen_opt_vf)

        return total  #lowest value will be greater than zero, useful for GP approximation of VF

    def E_V(self, scaled_state, params, control):
        state = self.unscale_state(scaled_state)
        disc_state = state[-1].type(torch.IntTensor)
        """Caclulate the expectation of V"""

        e_v_next = 0
        lower_V = self.cfg["model"]["params"]["lower_V"]
        pen_opt_vf = self.cfg["model"]["params"]["pen_opt_vf"]

        weights, points = self.state_iterate_exp(scaled_state, params, control)
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        for i in range(len(weights)):
            p_i = torch.unsqueeze(points[i, :-1], 0)
            with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(200):
                obj_val = (self.M[int(points[i, -1].item())][0](p_i).mean + gp_offset)
            e_v_next += (obj_val)  * weights[i]
            # e_v_next -= 1000 * pen_opt_vf * (torch.nn.functional.relu(lower_V - obj_val))**2

        return e_v_next


    def state_next(self, scaled_state, params, control, zpy, opt=False):
        """Return next periods states, given the controls of today and the random discrete realization"""
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        S = self.S

        s = state.clone()
        
        # update discrete state
        s[-1] = 1.*zpy
        
        # s[S[f"w_{1}"]:S[f"w_{1}"]+n_types] = state[self.S[f"w_{1}"]:self.S[f"w_{1}"]+n_types] + control[self.P[f"fut_util_{int(zpy)+1}_{1}"]:self.P[f"fut_util_{int(zpy)+1}_{1}"] + n_types]
        s[S[f"w_{1}"]:S[f"w_{1}"]+n_types] = control[self.P[f"fut_util_{int(zpy)+1}_{1}"]:self.P[f"fut_util_{int(zpy)+1}_{1}"] + n_types]

        # for indx in range(n_types):
            # s[S[f"w_{indx+1}"]] = state[self.S[f"w_{int(zpy)+1}"]] + control[self.P[f"fut_util_{int(zpy)+1}_{indx+1}"]]
            # s[S[f"w_{indx+1}"]] = control[self.P[f"fut_util_{int(zpy)+1}_{indx+1}"]]

        return self.scale_state(s)
 
    def state_next_batched(self, scaled_state, params, control):
        """Return next periods states, given the controls of today and the random discrete realization"""
        state = self.unscale_state(scaled_state)
        n_types = self.cfg["model"]["params"]["n_types"]
        S = self.S

        s = torch.zeros([n_types,state.shape[-1]])
        
        # update discrete state
        s[:,-1] = 1.* torch.arange(n_types)
        
        tmp_mat = torch.reshape(control[self.P[f"fut_util_{1}_{1}"]:self.P[f"fut_util_{1}_{1}"] + n_types**2],[n_types,n_types])
        # s[:,S[f"w_{1}"]:S[f"w_{1}"]+n_types] = torch.unsqueeze(state[self.S[f"w_{1}"]:self.S[f"w_{1}"]+n_types],0) + tmp_mat
        s[:,S[f"w_{1}"]:S[f"w_{1}"]+n_types] = tmp_mat

        return self.scale_state(s)

    def state_iterate_exp(self, scaled_state, params, control):
        """How are future states generated from today state and control"""
        state = self.unscale_state(scaled_state)
        
        n_types = self.cfg["model"]["params"]["n_types"]
        disc_state = state[-1].type(torch.IntTensor)
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        weights = torch.tensor([trans_mat[disc_state,indx]  for indx in range(n_types)])
        
        # points = torch.cat(
        #     tuple(
        #         torch.unsqueeze(self.state_next(scaled_state, params, control, z), dim=0)
        #         for z in range(self.discrete_state_dim)
        #     ),
        #     dim=0,
        # )

        points = self.state_next_batched(scaled_state, params, control)
        mask_pos = weights > 0.
  

        return weights[mask_pos], points[mask_pos,:]


    def lb(self, scaled_state, params):
        state = self.unscale_state(scaled_state)
        S = self.S
        P = self.P
        disc_state = int(state[-1].item())
        n_types = self.cfg["model"]["params"]["n_types"]
        beta = self.cfg["model"]["params"]["beta"]
        lower_w = self.cfg["model"]["params"]["lower_w"]
        X_L = np.zeros(self.control_dim)

        # X_L[P[f"fut_util_{1}_{1}"]:P[f"fut_util_{1}_{1}"]+n_types**2] = (lower_w[0])
        for indxa in range(n_types):
            for indxa2 in range(n_types):
                # X_L[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (lower_w[indxa2] - state[self.S[f"w_{indxa2 + 1}"]])
                X_L[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (lower_w[indxa2])
        return X_L

    def ub(self, scaled_state, params):
        state = self.unscale_state(scaled_state)
        S = self.S
        P = self.P
        disc_state = int(state[-1].item())
        n_types = self.cfg["model"]["params"]["n_types"]
        upper_transfer = self.cfg["model"]["params"]["upper_trans"]
        reg_c = self.cfg["model"]["params"]["reg_c"]
        sigma = self.cfg["model"]["params"]["sigma"]
        upper_w = self.cfg["model"]["params"]["upper_w"]

        beta = self.cfg["model"]["params"]["beta"]
        X_U = np.empty(self.control_dim)
        X_U[P[f"u_{1}"]:P[f"u_{1}"]+n_types] = utility_ind(1 * upper_transfer, reg_c, sigma)
        X_U[P[f"pen_{1}"]:P[f"pen_{1}"]+n_types] = 1000.
        X_U[P[f"pen_u_{1}"]:P[f"pen_u_{1}"]+n_types] = 1000.
        X_U[P[f"c_{1}"]:P[f"c_{1}"]+n_types] = 1 * upper_transfer
        # X_U[P[f"fut_util_{1}_{1}"]:P[f"fut_util_{1}_{1}"]+n_types**2] = (upper_w[0])

        for indxa in range(n_types):
            for indxa2 in range(n_types):
                # X_U[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (upper_w[indxa2] - state[self.S[f"w_{indxa2 + 1}"]])
                X_U[self.P[f"fut_util_{indxa+1}_{indxa2+1}"]] = (upper_w[indxa2])

        return X_U

    def cl(self, scaled_state, params):
        n_types = self.cfg["model"]["params"]["n_types"]
        # M = 2*n_types + n_types - 1 + n_types
        M = 2*n_types + n_types*(n_types-1) // 2 #+ n_types
        G_L = np.empty(M)
        G_L[:] =  0.0
        return G_L

    def cu(self, scaled_state, params):
        n_types = self.cfg["model"]["params"]["n_types"]
        # M = 2*n_types + n_types - 1 + n_types
        M = 2*n_types + n_types*(n_types-1) // 2 #+ n_types
        # number of constraints
        G_U = np.empty(M)
        # Set bounds for the constraints
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

    def bayesian_opt_criterion(self, eval_pt,discrete_state, target_p, rho, beta, pen_val=torch.tensor([-1.0e10])):

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

        if self.cfg["model"]["GP_MODEL"]["name"] == "ASGPModel" or self.cfg["model"]["GP_MODEL"]["name"] == "DeepKernel":
            eval_pt = self.M[discrete_state][target_p].feature_extractor(eval_pt)

        #compute the mean squared error MSE according to Dario Azzimonti Gaussian processes and sequential design of experiments (lecture)
        if eval_pt.shape[0] > 1:             
            eval_pt = torch.unsqueeze(eval_pt,1) #doing a batch of points needs to be done as batches of single points

        train_inputs = self.M[discrete_state][target_p].train_inputs[0]
        if self.cfg["model"]["GP_MODEL"]["name"] == "ASGPModel" or self.cfg["model"]["GP_MODEL"]["name"] == "DeepKernel":
            train_inputs = self.M[discrete_state][target_p].feature_extractor(train_inputs)

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

        out_vec = self.mean_squared_error_GP(eval_pt,discrete_state,target_p)
        # out_vec = self.bayesian_opt_criterion(eval_pt,discrete_state,target_p)

        # #compute bayesian optimization criteria
        # pred = self.M[discrete_state][target_p](
        #                 eval_pt
        #             )

        # var_v = pred.variance
        # mean_v = pred.mean

        # #Deisenroth criterion
        # is_v_ok = (var_v > 0.0000001)
        # out_vec = is_v_ok*(rho * (mean_v) + beta / 2.0 * torch.log(var_v + 1e-15)) + torch.logical_not(is_v_ok)*pen_val

        # #upper bound criterion
        # out_vec = (mean_v) + 2 * var_v

        # # #expected improvement criterion
        # tn = torch.max(
        #     self.combined_sample_all[
        #     self.get_d_rows(discrete_state,drop_non_converged=True),
        #     0])
        # un = mean_v - tn
        # mask_var = var_v < 1e-6
        # out_vec_1 = torch.maximum(
        #     torch.tensor(0.),
        #     un)   
        
        # m = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))   
        # var_v_aux = mask_var*torch.ones_like(var_v) + torch.logical_not(mask_var)*var_v
        # out_vec_2 = (
        #     torch.maximum(
        #         torch.tensor(0.),
        #         un) - 
        #     torch.abs(un)*m.cdf(un/var_v_aux) + 
        #     var_v_aux * m.log_prob(un/var_v_aux).exp()) 

        # out_vec = mask_var*out_vec_1 + torch.logical_not(mask_var)*out_vec_2


        return out_vec

    def BAL(self):
        
        n_types = self.cfg["model"]["params"]["n_types"]
        lower_V = self.cfg["model"]["params"]["lower_V"]
        gp_offset = self.cfg["model"]["params"]["GP_offset"]
        pen_val = -1.0e10
        target = self.cfg["BAL"]["targets"][0]
        dim_state = self.state_sample_all.shape[1]


        self.warm_start = False
        self.fit_GP(self.cfg["torch_optim"]["iter_per_cycle"], self.only_fit_trans_pol(), self.warm_start) 
        n_sim_steps = 1000
        P = self.P
        trans_mat = self.cfg["model"]["params"]["trans_mat"]
        beta = self.cfg["model"]["params"]["beta"]

        #compute the state bounds
        lower_w = self.cfg["model"]["params"]["lower_w"]
        upper_w = self.cfg["model"]["params"]["upper_w"]
        LB_state = torch.zeros(dim_state - 1)
        UB_state = torch.zeros(dim_state - 1)

        for indxt in range(n_types):
            LB_state[self.S[f"w_{indxt+1}"]] = lower_w[indxt]
            UB_state[self.S[f"w_{indxt+1}"]] = upper_w[indxt]

        LB_state = scale_state_func(LB_state,self.cfg)
        UB_state = scale_state_func(UB_state,self.cfg)

        #new bal points are stored here with their bal util
        out_tmp = torch.zeros([n_types,1 + dim_state])
        out_tmp[:,-1] = pen_val


        #pick sample point with maximal vf value to start sim from
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

        if self.cfg.get("distributed"):
            if int(dist.get_world_size()) >= 2*n_types:
                tasks_per_worker = 1 # run as many simulations as MPI threads
                n_threads = int(tasks_per_worker*dist.get_world_size()) 
            else:
                tasks_per_worker = int(ceil(dist.get_world_size() / (2*n_types))) # run at least two sims per type
                n_threads = 2*n_types

            torch.manual_seed(1211 + dist.get_rank()*123 + self.epoch*3654)

            # allocate fitting across workers
            worker_slice = [
                A
                for A in range(n_threads)
                if int(A / tasks_per_worker) == dist.get_rank()
            ]
        else:
            tasks_per_worker = 2*n_types
            torch.manual_seed(1054211+ self.epoch*3654)
            worker_slice = list(range(tasks_per_worker))

        for indx_ in worker_slice: #start a simulation for each types's maximum starting point
            indx_type = (indx_%int(n_types))
            current_state = torch.unsqueeze(start_pt[indx_type,:],dim=0)

            for indxt in range(n_sim_steps):
                if indx_ < n_types and indxt < n_sim_steps/2:
                    current_state[0,-1] = 1.*indx_type

                current_disc_state = int(current_state[0,-1].item())

                #compute state transition
                pol_out = torch.zeros(self.control_dim)
                params = self.get_params(current_state[0,:],None)
                try:
                    for p1 in range(1,n_types+1):

                        for p2 in range(1,n_types+1):
                            with torch.no_grad():
                                next_pol_tmp = self.M[current_disc_state][1+P[f"fut_util_{p1}_{p2}"]](current_state[:,:-1]).mean

                            pol_out[P[f"fut_util_{p1}_{p2}"]] = next_pol_tmp[0]
                except:
                    logger.info(f"Evaluation failed in state {current_disc_state}; aborting simulation.")
                    break

                #compare bal utility
                
                with torch.no_grad():
                    bal_util = self.bal_utility_func(current_state,0, pen_val=torch.tensor([pen_val]))
                    v = self.M[current_disc_state][0](current_state[:,:-1]).mean

                if bal_util > out_tmp[current_disc_state,-1]:
                    out_tmp[current_disc_state,:-1] = current_state[0,:]
                    out_tmp[current_disc_state,-1] = bal_util[0]

                #check of we need to abort sim because we walked somewhere nonsensical
                if (v + gp_offset <= lower_V and indxt > 0 and self.epoch > n_rand_its):
                    print(f"Break simulation in iteration {indxt} with value {v + gp_offset} at point {current_state[0,:]}")
                    break
                if v <= 0.0 and indxt > 0: # and self.epoch <= 300:
                    print(f"Break simulation in iteration {indxt} with value {v + gp_offset} at point {current_state[0,:]}")
                    break

                cat_dist = torch.distributions.categorical.Categorical(trans_mat[current_disc_state,:])
                next_disc_state = int((cat_dist.sample()).item())
                current_state = torch.unsqueeze(self.state_next(current_state[0,:], params, pol_out, next_disc_state),0)
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

            logger.info(f"BAL added points {out} with final bal util {final_bal_util}")
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


    def iterate(self, training_iter=100):

        # fit pattern for GPs
        if self.epoch == 0:
            fit_lst = self.only_fit_VF()

            # fit pol and vf in first iteration
            self.fit_GP(training_iter, fit_lst)
            self.save()
            self.warm_start = False

        else:
            fit_lst = self.only_fit_VF()
            
        # perform Howard improvment step
        vf_lst = self.only_fit_VF()

        self.epoch += 1
        logger.info(f"Starting EPOCH #{self.epoch} - current sample size is {self.state_sample_all.shape[0]}")

        # generate new sample
        self.sample_all()

        # update V_sample for next-step by solving the VF iteration
        self.solve_all()

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

        self.fit_GP(training_iter, fit_lst, self.warm_start) 

        self.warm_start = self.cfg.get("warm_start", False)


        metrics_how = {
            "l2": self.convergence_error(fit_lst = vf_lst,ord=2),
            "l_inf": self.convergence_error(fit_lst = vf_lst),
        }
        n_disc_states = self.cfg["model"]["params"]["n_types"]
        indx_it = 0
        error_old = 1
        self.prev_combined_sample = self.combined_sample_all.clone()
        while indx_it < self.cfg["model"]["params"]["n_Howard_steps"] and error_old > 1e-8 and self.epoch > max(100,self.cfg["BAL"]["epoch_freq"]):

            # self.fit_GP(training_iter, vf_lst, self.warm_start)
                
            self.sample_all_Howard()
            self.Howard_step(vf_lst)
            self.allgather()
            
            metrics_how = {
                "l2": self.convergence_error(fit_lst = vf_lst,ord=2),
                "l_inf": self.convergence_error(fit_lst = vf_lst),
            }
            error_new = torch.sum(metrics_how['l2'])/n_disc_states
            logger.info(
                f"Howard iteration {indx_it} error: {error_new} (L2)"
            )            
            
            if error_new > error_old:
                self.combined_sample_all = self.prev_combined_sample.clone()
                logger.info("No improvement in Howard it detected aborting iteration")
                break

            indx_it += 1
            self.prev_combined_sample = self.combined_sample_all.clone()
            error_old = error_new

            self.fit_GP(0, vf_lst, True) #sync GPs without training using previous parameters

        if indx_it > 0:
            self.fit_GP(training_iter, vf_lst, self.warm_start)  #fit GPs after Howard steps


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
            policies.append(torch.zeros([0,self.combined_sample_all.shape[-1] - 1]))

        self.policy_sample = torch.cat(policies, dim=0)
        self.combined_sample = torch.cat(
            (torch.unsqueeze(self.V_sample, dim=1), self.policy_sample), dim=1
        )

    @torch.no_grad()
    def sample_all_Howard(self):

        self.prev_state_sample = self.state_sample_all.clone()
        self.prev_combined_sample = self.combined_sample_all.clone()
        self.prev_feasible = self.feasible_all.clone()

        self.feasible = self.prev_feasible.to(self.device)
        self.state_sample = self.prev_state_sample.to(self.device)
        self.combined_sample = self.prev_combined_sample.to(self.device)


        # the sampling is always for the complete population
        self.state_sample_all = self.state_sample.to(self.device)

        self.combined_sample_all = self.combined_sample.to(self.device)

        self.feasible_all = self.feasible.to(self.device)

        # non-convered points
        self.non_converged_all = torch.zeros(self.state_sample.shape[0]).to(self.device)

        # distribute the samples
        self.scatter_sample()     


    def create_model(self, d, p, train_x, train_y, warm_start=False):
        if self.cfg.get('use_fixed_noise',True):
            if p == 0:
                noise_vec = torch.zeros(train_y.shape[0])
                noise_vec[:] = 1e-5

            else:
                noise_vec = 1e-5

            self.likelihood[d][p] = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                        noise_vec,
                        learn_additional_noise=False
                    ).to(self.device)

        else:
            self.likelihood[d][p] = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(1e-7)
                    ).to(self.device)



        from GPModels.ExactGPModel import GPModel_pol
        if p > 0:
            model = GPModel_pol(
                        d,
                        p,
                        train_x,
                        train_y,
                        self.likelihood[d][p],
                        self.cfg
                    ).to(self.device)
        else:
            model = self.Model(
                        d,
                        p,
                        train_x,
                        train_y,
                        self.likelihood[d][p],
                        self.cfg
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
                scale_bound = scale_state_func(upper_w,self.cfg)
                length_scale_init = 1*scale_bound * torch.sqrt(torch.tensor(1.*n_types))
                # model.covar_module.kernels[1].raw_constant.data.fill_((torch.tensor(1.0)))
            else:
                upper_w = self.cfg["model"]["params"]["upper_w"]
                scale_bound = scale_state_func(upper_w,self.cfg)
                length_scale_init = 0.01*scale_bound * torch.sqrt(torch.tensor(1.*n_types))

            # model.covar_module.base_kernel.alpha = 0.01

            # model.covar_module.base_kernel.lengthscale = (length_scale_init)
            # if not self.cfg.get('use_fixed_noise',True):
            #     model.likelihood.noise_covar.noise = torch.tensor(0.64)


        return model

    def solve_all(self):
        policies = []
        self.V_sample = torch.zeros(self.state_sample.shape[0])
        for s in range(self.state_sample.shape[0]):
            try:
                if self.feasible[s] == 1.0:
                    v, p = self.solve(self.state_sample[s, :],self.combined_sample[s,1:])
                    policies.append(torch.unsqueeze(p, 0))
                    self.feasible[s] = self.is_feasible(self.state_sample[s, :],v,p)
                else:
                    params = self.get_params(self.state_sample[s, :],self.combined_sample[s,1:])
                    control = self.combined_sample[s,1:]

                    value = self.eval_f(self.state_sample[s, :], params, control)
                    pol_new,v = self.post_process_optimization(self.state_sample[s, :], params, control, value)                    
                    pol = self.combined_sample[s,1:]
                    policies.append(torch.unsqueeze(pol, 0))
                    self.feasible[s] =  self.is_feasible(self.state_sample[s, :],v,pol)
                    # v = 0.
                    # self.feasible[s] = 0.

                self.V_sample[s] = v


                if s % (self.state_sample.shape[0] / 10) == 0:
                    logger.debug(f"Finished solving Ipopt Problem #{s}")
            except NonConvergedError as e:
                logger.debug(
                    f"Optimization did not converge for: {str(self.state_sample[s, :])}"
                )
                # let's interpolate then the VF
                with torch.no_grad():
                    # get discrete state
                    d = int(self.state_sample[s, -1].item())
                    sample = self.state_sample[s, :-1]
                    sample = sample[self.get_d_cols(d)]
                    self.V_sample[s] = (
                        (self.M[d][0](torch.unsqueeze(sample, 0)) )
                        .mean
                    )

                    self.non_converged[s] = 1
                    self.feasible[s] = 1.0
                    pol = self.combined_sample[s,1:] #torch.zeros(self.policy_dim)

                policies.append(torch.unsqueeze(pol, 0))


        # get the policy dimension
        if self.state_sample.shape[0] == 0:
            policies.append(torch.zeros([0,self.combined_sample_all.shape[-1] - 1]))

        self.policy_sample = torch.cat(policies, dim=0)
        self.combined_sample = torch.cat(
            (torch.unsqueeze(self.V_sample, dim=1), self.policy_sample), dim=1
        )


    def create_optimizer(self, d, p):
        train_iter,lr = self.get_solver_config(d,p)
        self.cfg["torch_optim"]["config"]["lr"] = lr
        return  getattr(torch.optim, self.cfg["torch_optim"]["name"])(
                    self.M[d][p].parameters(), **self.cfg["torch_optim"]["config"]
                )
    
    def fit_GP(self, training_iter=100, dp: List[Tuple] = [()], warm_start=False):
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
            worker_slice = list(range(len(dp)))

        for w, W in enumerate(dp):
            d, p = W
            # update the training data - last column is the discrete state
            train_sample_rows = self.state_sample_all[
                self.get_d_rows(
                    d, drop_non_converged=self.cfg.get("drop_non_converged")
                ),
                :-1,
            ]
            train_sample = train_sample_rows[:,self.get_d_cols(d)].clone()
            train_v = self.combined_sample_all[
                self.get_d_rows(
                    d, drop_non_converged=self.cfg.get("drop_non_converged")
                ),
                p,
            ].clone()

            train_v = train_v.contiguous()
            train_sample = train_sample.contiguous()
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
                # mask_feas = train_vf >= 0.001
                train_v = (train_v[mask_feas]).contiguous()
                train_sample = (train_sample[mask_feas,:]).contiguous()
            if p == 0:
                no_init_pts = self.cfg["no_feas_samples"] + self.cfg["no_samples"]
                mask_feas_init = train_v[:no_init_pts] >= 0.0001
                mask_infeas_init = torch.logical_not(mask_feas_init)
                init_feas_samples = train_sample[:no_init_pts,...][mask_feas_init,...]
                init_feas_train_v = train_v[:no_init_pts][mask_feas_init]
                init_infeas_samples = train_sample[:no_init_pts,...][mask_infeas_init,...]
                init_infeas_train_v = train_v[:no_init_pts][mask_infeas_init]

                n_types = self.cfg["model"]["params"]["n_types"]
                train_sample = torch.cat((
                    init_feas_samples,
                    init_infeas_samples[:n_types+1,...],
                    train_sample[no_init_pts:,:]
                ),dim=0).contiguous()
                train_v = torch.cat((
                    init_feas_train_v,
                    init_infeas_train_v[:n_types+1],
                    train_v[no_init_pts:]
                ),dim=0).contiguous()



            self.M[d][p] = self.create_model(d, p, train_sample, train_v, warm_start)

            self.optimizer[d][p] = self.create_optimizer(d,p)

            if not self.cfg["model"]["GP_MODEL"]["name"] == "VariationalGPModel":
                self.M[d][p].set_train_data(
                    train_sample,
                    train_v,
                    strict=False,
                )

            # fit first the GP
            self.M[d][p].train()
            self.likelihood[d][p].train()

            if training_iter > 0:
                training_iter,lr = self.get_solver_config(d, p)

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