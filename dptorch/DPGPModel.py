import torch
import gpytorch
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple
import copy
from Utils import NonConvergedError
import os

if os.getenv("OMPI_COMM_WORLD_SIZE"):
    import torch.distributed as dist

# added these lines to include PyTorch-LBFGS
import sys
from hydra.utils import to_absolute_path

torch.set_default_dtype(torch.float64)

rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))
if rank != -1:
    main_logger = logging.getLogger(f"{__name__}.Rank{rank}")
    training_logger = logging.getLogger(f"{__name__}.Rank{rank}.Train")
else:
    main_logger = logging.getLogger(f"{__name__}")
    training_logger = logging.getLogger(f"{__name__}.Train")


class DPGPModel(ABC):
    def __init__(
        self,
        V_guess=None,
        cfg={},
        state_sample=torch.tensor([]),
        V_sample=torch.tensor([]),
        policy_sample=torch.tensor([]),
        policy_dim: int = None,
        discrete_state_dim: int = 1,
        policy_names=[],
        state_names=[],
    ):
        self.current_ll = 0
        self.current_p_ll = 0
        self.epoch = 0
        self.discrete_state_dim = discrete_state_dim
        self.state_sample = state_sample

        if cfg.get("WORKER_MAIN_LOG_LEVEL", False) and rank > 0:
            main_logger.setLevel(cfg["WORKER_MAIN_LOG_LEVEL"])

        if cfg.get("WORKER_TRAIN_LOG_LEVEL", False) and rank > 0:
            training_logger.setLevel(cfg["WORKER_TRAIN_LOG_LEVEL"])

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not cfg.get("force_cpu", False)
            else "cpu"
        )

        main_logger.info("Received config: ")
        main_logger.info(cfg)
        self.cfg = cfg

        # utility to index with policy / state names
        if policy_names:
            self.P = {key: val for val, key in enumerate(policy_names)}

        if state_names:
            self.S = {key: val for val, key in enumerate(state_names)}

        if policy_sample.numel() > 0:
            self.policy_dim = policy_sample.shape[1]
        else:
            assert (
                policy_dim is not None
            ), "Policy dimension needs to be provided in policy_dim constructor argument!"
            self.policy_dim = policy_dim

        if not state_sample.numel() > 0:
            # initialize first iteration from steady-state
            assert (
                V_guess is not None
            ), "No state guess is given, so will be generated from the model. An initial V_guess func needs to be provided however."
            self.sample_all()
            self.V_sample = torch.zeros(self.state_sample.shape[0], device=self.device)
            for i in range(self.state_sample.shape[0]):
                self.V_sample[i] = V_guess(self.state_sample[i, :])
        else:
            self.sample_all(state_sample)
            if not V_sample.numel() > 0:
                assert (
                    V_guess is not None
                ), "Either a V_guess func or a corresponding V_sample needs to be provided for state_sample."
                self.V_sample = torch.zeros(
                    self.state_sample.shape[0], device=self.device
                )
                for i in range(self.state_sample.shape[0]):
                    self.V_sample[i] = V_guess(self.state_sample[i, :])
            else:
                # 'scatter' V_sample (since V_sample is available on all ranks, just subset)
                if self.cfg.get("distributed"):
                    self.V_sample = V_sample[
                        min(self.worker_slice) : (max(self.worker_slice) + 1)
                    ]
                else:
                    self.V_sample = V_sample.to(self.device)

        if policy_sample.numel() > 0:
            # 'scatter' V_sample (since V_sample is available on all ranks, just subset)
            if self.cfg.get("distributed"):
                self.policy_sample = policy_sample[
                    min(self.worker_slice) : (max(self.worker_slice) + 1), :
                ]
            else:
                self.policy_sample = policy_sample.to(self.device)

            self.policy_dim = self.policy_sample.shape[1]
        else:
            assert (
                policy_dim is not None
            ), "Policy dimension needs to be provided in policy_dim constructor argument!"
            self.policy_dim = policy_dim
            self.policy_sample = torch.zeros(
                self.state_sample.shape[0], self.policy_dim
            )

            for i in range(self.state_sample.shape[0]):
                self.policy_sample[i, :] = self.policy_guess(self.state_sample[i, :])

        if policy_names:
            self._policy_names = policy_names
        else:
            self._policy_names = list(range(self.policy_dim))

        # zero-override
        if self.cfg.get("init_with_zeros"):
            self.V_sample = torch.zeros_like(self.V_sample, device=self.device)

        self.combined_sample = torch.cat(
            (torch.unsqueeze(self.V_sample, dim=1), self.policy_sample), dim=1
        )

        # gather the initial VF guesses across all processes
        self.allgather()

        # "Loss" for GPs - the marginal log likelihood
        self.mll = [
            [
                -1
                for m in range(1 + self.policy_dim)
            ]
            for d in range(self.discrete_state_dim)
        ]

        # "Loss" for GPs - the marginal log likelihood
        self.likelihood = [
            [
                -1
                for m in range(1 + self.policy_dim)
            ]
            for d in range(self.discrete_state_dim)
        ]

        self.M = [
            [
                self.create_model(
                    d,
                    p,
                    self.sample_pts(d),
                    self.combined_sample_all[self.get_d_rows(d), p]
                )
                for p in range(1 + self.policy_dim)
            ]
            for d in range(self.discrete_state_dim)
        ]


        # Use the adam optimizer
        self.optimizer = [
            [
                self.create_optimizer(d, p)
                for p in range(1 + self.policy_dim)
            ]
            for d in range(self.discrete_state_dim)
        ]

        for d in range(self.discrete_state_dim):
            for p in range(1 + self.policy_dim):
                self.sync_GP(d, p)


        # if no explicit distinction, then number of policies = number of controls
        if not hasattr(self, "control_dim"):
            self.control_dim = self.policy_dim

    def create_model(self, d, p, train_x, train_y, warm_start=False):
        if self.cfg.get('use_fixed_noise',True):

            noise_vec = torch.ones(train_y.shape[0])*self.cfg["gpytorch"].get("likelihood_noise", 1e-6)

            self.likelihood[d][p] = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                        noise_vec,
                        learn_additional_noise=False
                    ).to(self.device)

        else:
            self.likelihood[d][p] = gpytorch.likelihoods.GaussianLikelihood(
                        noise_constraint=gpytorch.constraints.GreaterThan(
                            self.cfg["gpytorch"].get("likelihood_noise_lb", 1e-3)
                        )
                    ).to(self.device)

        model = self.Model(
                    d,
                    p,
                    train_x,
                    train_y,
                    self.likelihood[d][p],
                    self.cfg,
                    self.epoch
                ).to(self.device)

        self.mll[d][p] = gpytorch.mlls.ExactMarginalLogLikelihood(
                    self.likelihood[d][p], model
                )


        length_scale_init = None
        if not self.cfg.get('use_fixed_noise',True):
            noise_init = None
        if warm_start:
            try:
                state_dict = copy.deepcopy(self.M[d][p].state_dict())
                model.load_state_dict(state_dict)
            except:
                if length_scale_init is not None:
                    model.covar_module.base_kernel.lengthscale = length_scale_init   
                if not self.cfg.get('use_fixed_noise',True) and noise_init is not None:
                    model.likelihood.noise_covar.noise = noise_init
        else:
            if length_scale_init is not None:
                model.covar_module.base_kernel.lengthscale = (length_scale_init)
            if not self.cfg.get('use_fixed_noise',True) and noise_init is not None:
                model.likelihood.noise_covar.noise = noise_init

        return model

    def create_optimizer(self, d, p):
        return  getattr(torch.optim, self.cfg["torch_optim"]["name"])(
                    self.M[d][p].parameters(), **self.cfg["torch_optim"]["config"]
                )

    
    def get_fit_precision(self,d,p):
        rel_ll_change_tol = self.cfg["torch_optim"].get("relative_ll_change_tol",1e-4)
        relative_ll_grad_tol = self.cfg["torch_optim"].get("relative_ll_grad_change_tol",1e-2)
        relative_error_tol = self.cfg["torch_optim"].get("relative_error_tol", 0)
        parameter_change_tol= self.cfg["torch_optim"].get("parameter_change_tol", 0)
        return rel_ll_change_tol, relative_ll_grad_tol, relative_error_tol, parameter_change_tol

    def get_scipy_precision(self,d):
        tol = self.cfg["scipyopt"].get("tol", 1e-6)
        return tol
    

    def sample_pts(self, d):
        tmp_pts = self.state_sample_all[self.get_d_rows(d), :-1]
        return tmp_pts[:,self.get_d_cols(d)]

    def get_d_rows(self, d, drop_non_converged=True):
        d_selected = self.state_sample_all[:, -1] == d
        if drop_non_converged:
            return torch.logical_and(d_selected, self.non_converged_all < 1)
        else:
            return d_selected

    def get_d_cols(self, d):  #per default all GPs have the same inputs
        lst = []
        for i in range(self.state_sample_all.shape[1]-1):
            lst.append(True)

        return torch.tensor(lst)

    def sync_GP(self, d: int = 0, p: int = 0, source_rank: int = 0):
        if self.cfg.get("distributed"):
            b_objects = [self.M[d][p].state_dict(), self.optimizer[d][p].state_dict()]
            # Broadcast parameters from rank 0 to all other processes.
            dist.broadcast_object_list(b_objects, src=source_rank)
            if dist.get_rank() != source_rank:
                self.M[d][p].load_state_dict(b_objects[0])
                self.optimizer[d][p].load_state_dict(b_objects[1])

    def E_V(self, state, params, control):
        """Caclulate the expectation of V"""
        # if not VFI, then return a differentiable zero
        if self.cfg["model"].get("ONLY_POLICY_ITER"):
            return torch.sum(control) * torch.zeros(1)

        e_v_next = 0

        weights, points = self.state_iterate_exp(state, control)
        for i in range(len(weights)):
            d = int(points[i, -1].item())
            p_i_ = torch.unsqueeze(points[i, :-1], 0)
            p_i = (p_i_[:,self.get_d_cols(d)]) 
            e_v_next += (self.M[d][0](p_i).mean) * weights[i]

        return e_v_next

    def grad_E_V(self, state, params, control):
        """Caclulate the gradient wrt. control of the expectation of V"""
        c = control.clone().detach().requires_grad_(True)
        e_v = self.E_V(state, params, c)
        e_v.backward()
        return c.grad

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

            # fit first the GP
            self.M[d][p].train()
            self.likelihood[d][p].train()

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

        main_logger.info(
            f"Interpolation error: {metrics_int['l_inf']} (L_inf) {metrics_int['l2']} (L2) for state policy pairs {dp}"
        )

    def optimize_ll(self,training_iter,d,p,train_sample,train_v):
        if p == 0:
            training_logger.info(
                f"Fitting Value Function for discrete state: {d}"
            )
        else:
            training_logger.info(
                f"Fitting Policy Function #{p} for discrete state: {d}"
            )

        try:
            try:
                current_noise = torch.max(self.M[d][p].likelihood.noise).item()
            except:
                current_noise = 0.
            with torch.no_grad():
                training_logger.info(
                        "Before optimization state %.0f policy %.0f: Loss: %.7f  noise: %.7f"
                        % (d,
                        p,
                        -self.mll[d][p](self.M[d][p](train_sample), train_v),
                        current_noise)
                )
        except:
            training_logger.info(
                    "Evaluation of GP failed reset parameters"
            )
            self.M[d][p] = self.create_model(d, p, train_sample, train_v)
            self.optimizer[d][p] = self.create_optimizer(d,p)


        if self.cfg.get("init_with_zeros") and self.epoch == 1:
            training_iter = 0

        def closure():
            batch_size = self.cfg["torch_optim"].get("minibatch_size")
            if batch_size:
                batch_ix = torch.randperm(train_sample.shape[0])[
                    :batch_size
                ]
                self.M[d][p].set_train_data(
                    train_sample[batch_ix, :],
                    train_v[batch_ix],
                    strict=False,
                )
            else:
                batch_ix = range(train_sample.shape[0])

            # Zero gradients from previous iteration
            self.optimizer[d][p].zero_grad()
            # Output from model - only for the current sample
            output = self.M[d][p](train_sample[batch_ix, :])
            # Calc loss and backprop gradients

            self.current_ll = -self.mll[d][p](output, train_v[batch_ix])

            self.current_ll.backward()

            return self.current_ll

        rel_ll_change_tol,relative_ll_grad_tol,relative_error_tol,parameter_change_tol = self.get_fit_precision(d,p)
        rel_err = 1e10
        prior_grad_norm = torch.tensor(1.)
        old_ll = torch.tensor(0.)
        current_ll = closure()
        rel_ll_change = 1.0
        rel_ll_grad = 1.0
        max_param_change = 1.0
        prev_params = [param.clone() for param in self.M[d][p].parameters()]
        for i in range(1, 1 + training_iter):
            ll_grad_norm = torch.tensor(0.)
            n_paras = 0.
            for group in self.optimizer[d][p].param_groups:
                for para in group["params"]:
                    ll_grad_norm += torch.sum(para.grad**2)
                    try:
                        n_paras += para.grad.shape[-1]
                    except:
                        n_paras +=1


            ll_grad_norm = torch.sqrt(ll_grad_norm)/n_paras
            rel_ll_grad = torch.abs(ll_grad_norm)/(1 + torch.abs(current_ll))


            if rel_ll_grad < relative_ll_grad_tol or rel_ll_change < rel_ll_change_tol or max_param_change < parameter_change_tol:
                try:
                    current_noise = torch.max(self.M[d][p].likelihood.noise).item()
                except:
                    current_noise = 0.
                training_logger.info(
                        "Converged after optimization in state %.0f policy %.0f step %.0f: Loss: %.7f noise: %.7f relerr: %.7f relllerr: %.7f ll_grad: %.7f param_ch: %.7f"
                        % (
                            d,
                            p,
                            i-1,
                            self.current_ll,
                            current_noise,
                            rel_err,
                            rel_ll_change,
                            rel_ll_grad,
                            max_param_change
                        )
                )
                break


            state_dict_old = copy.deepcopy(self.M[d][p].state_dict())
            state_dict_opt_old = copy.deepcopy(self.optimizer[d][p].state_dict())

            self.M[d][p].eval()
            self.likelihood[d][p].eval()          
            try:
                with torch.no_grad():
                    rel_err = torch.linalg.norm((((self.M[d][p](train_sample)).mean - train_v)/(1 + torch.abs(train_v))),ord=2)/train_v.shape[0]
                if rel_err < relative_error_tol:
                    try:
                        current_noise = torch.max(self.M[d][p].likelihood.noise).item()
                    except:
                        current_noise = 0.
                    training_logger.info(
                            "Converged after optimization in state %.0f policy %.0f step %.0f: Loss: %.7f noise: %.7f relerr: %.7f relllerr: %.7f ll_grad: %.7f param_ch: %.7f"
                            % (
                                d,
                                p,
                                i-1,
                                self.current_ll,
                                current_noise,
                                rel_err,
                                rel_ll_change,
                                rel_ll_grad,
                                max_param_change
                            )
                    )
                    break                
            except:
                training_logger.info(f"Training step failed in step {i} for state {d} and policy {p} reset to previous step model and optimizer state and stop.")
                self.M[d][p].load_state_dict(state_dict_old)
                self.optimizer[d][p].load_state_dict(state_dict_opt_old)
                ll_grad_norm = prior_grad_norm
                break                       

            self.M[d][p].train()
            self.likelihood[d][p].train()
            try:
                self.optimizer[d][p].step(closure)
                
                if torch.isnan(self.current_ll):
                    raise ValueError(f"found nan in iteration {i} while training state {d} and policy {p} training inputs {train_sample} training targets {train_v}")                        
            except:
                self.M[d][p].load_state_dict(state_dict_old)
                self.optimizer[d][p].load_state_dict(state_dict_opt_old)
                ll_grad_norm = prior_grad_norm
                self.M[d][p].eval()
                self.likelihood[d][p].eval()    
                with torch.no_grad():
                    current_ll = self.mll[d][p](self.M[d][p](train_sample), train_v)
                    rel_err = torch.linalg.norm((((self.M[d][p](train_sample)).mean - train_v)/(1 + torch.abs(train_v))),ord=2)/train_v.shape[0]

                rel_ll_change = abs(current_ll - old_ll)/(1+abs(old_ll))
                training_logger.info(f"Training step failed in step {i} for state {d} and policy {p} reset to previous step model and optimizer state and stop with interpol error {rel_err} ll error {rel_ll_change} and grad {ll_grad_norm}.")

                break

            max_param_change = max(torch.max((torch.abs(prev_param - param)/(1e-6 + torch.abs(prev_param)))).item() for prev_param, param in zip(prev_params, self.M[d][p].parameters()))
            prev_params = [param.clone() for param in self.M[d][p].parameters()]

            current_ll = self.current_ll
            rel_ll_change = abs(current_ll - old_ll)/(1+abs(old_ll))
            if (i + 1) % (training_iter / 2) == 0:
                try:
                    current_noise = torch.max(self.M[d][p].likelihood.noise).item()
                except:
                    current_noise = 0.
                training_logger.info(
                        "After optimization in state %.0f policy %.0f step %.0f: Loss: %.7f noise: %.7f relerr: %.7f relllerr: %.7f ll_grad: %.7f param_ch: %.7f"
                        % (
                            d,
                            p,
                            i,
                            self.current_ll,
                            current_noise,
                            rel_err,
                            rel_ll_change,
                            rel_ll_grad,
                            max_param_change
                        )
                )
            
            old_ll = current_ll
            prior_grad_norm = ll_grad_norm

        return rel_ll_change,rel_ll_grad



    def gather_tensors(self, input_array):
        world_size = dist.get_world_size()
        ## gather shapes first
        myshape = input_array.shape
        mycount = int(torch.tensor(myshape).prod())
        shape_tensor = torch.tensor(myshape)
        all_shape = [torch.zeros_like(shape_tensor) for _ in range(world_size)]
        dist.all_gather(all_shape, shape_tensor)
        ## compute largest shapes
        all_count = [int(x.prod()) for x in all_shape]
        all_shape = [list(map(int, x)) for x in all_shape]
        max_count = max(all_count)
        ## padding tensors and gather them
        padded_input_array = torch.zeros(max_count)
        output_tensors = [
            torch.zeros_like(padded_input_array) for _ in range(world_size)
        ]
        padded_input_array[:mycount] = input_array.reshape(-1)
        dist.all_gather(output_tensors, padded_input_array)
        ## unpadding gathered tensors
        output = [
            x[: all_count[i]].reshape(all_shape[i])
            for i, x in enumerate(output_tensors)
        ]
        return output


    def allgather(self):
        # gather all the samples from all processes
        if self.cfg.get("distributed"):
            # do gathering with padding
            self.combined_sample_all = (
                torch.cat(self.gather_tensors(self.combined_sample))
                .clone()
                .detach()
                .to(self.device)
            )
            self.non_converged_all = (
                torch.cat(self.gather_tensors(self.non_converged))
                .clone()
                .detach()
                .to(self.device)
            )
            self.feasible_all = (
                torch.cat(self.gather_tensors(self.feasible))
                .clone()
                .detach()
                .to(self.device)
            )
        else:
            self.combined_sample_all = self.combined_sample.to(self.device)
            self.non_converged_all = self.non_converged.to(self.device)
            self.feasible_all = self.feasible.to(self.device)

        V_sample_all,policy_sample_all = self.process_training_data(self.state_sample_all,self.combined_sample_all)
        self.V_sample_all = V_sample_all
        self.policy_sample_all = policy_sample_all

        if torch.sum(self.non_converged_all) / self.combined_sample_all.shape[
            0
        ] >= self.cfg.get("non_converged_accepted_ratio", 0.2):
            raise NonConvergedError(
                "Ratio of optimizations that did not converge exceeded threshold."
            )

    def process_training_data(self,state_sample_all,combined_sample_all):
        V_sample_all = combined_sample_all[:, 0]
        policy_sample_all = combined_sample_all[:, 1:]
        return V_sample_all,policy_sample_all

    def policy_fit(self, training_iter=100):
        if self.epoch > 0:
            self.fit_GP(
                training_iter,
                self.what_pol_to_fit(),
                self.warm_start
            )
        else:
            self.fit_GP(
                training_iter,
                self.what_pol_to_fit(),
            )

    def what_to_fit(self): #per default only fit the VF in each iteration; customize if we need some policies
        return [(d, 0) for d in range(self.discrete_state_dim)]

    def what_pol_to_fit(self): #per default only fit the VF in each iteration; customize if we need some policies
        return [
                  (d, p)
                  for d in range(self.discrete_state_dim)
                  for p in range(1, self.policy_dim + 1)
               ]

    def iterate(self, training_iter=100):

        # fit GP to current sample
        if not self.cfg["model"].get("ONLY_POLICY_ITER"):
            if self.epoch > 0:
                self.fit_GP(training_iter, self.what_to_fit(), self.warm_start)
            else:
                self.fit_GP(training_iter, self.what_to_fit())
        else:
            self.policy_fit(training_iter)

        if self.epoch == 0:
            self.save()

        fit_lst = self.what_to_fit()

        self.epoch += 1
        main_logger.info(f"Starting EPOCH #{self.epoch} - current sample size is {self.state_sample_all.shape[0]}")

        # generate new sample
        self.sample_all()

        # update V_sample for next-step by solving the VF iteration
        self.solve_all()

        # gather estimated VF to all processes
        self.allgather()

        self.metrics[str(self.epoch)] = {
            "l2": self.convergence_error(ord=2),
            "l_inf": self.convergence_error(),
        }

        main_logger.info(
            f"Difference between previous interpolated values & next iterated values: {self.metrics[str(self.epoch)]['l_inf']} (L_inf) {self.metrics[str(self.epoch)]['l2']} (L2) for state policy pairs {fit_lst}"
        )

        if self.epoch % self.cfg.get("CHECKPOINT_INTERVAL", 10) == 0:
            if not self.cfg.get("DISABLE_POLICY_FIT") and not self.cfg["model"].get(
                "ONLY_POLICY_ITER"
            ):
                self.policy_fit(training_iter)
            self.save()

    def eval_f(self, state, params, control):
        return self.u(state, params, control) + self.beta * self.E_V(state, params, control)

    def grad_u(self, state, params, control):
        c = control.clone().detach().requires_grad_(True)
        u = self.u(state, params, c)
        u.backward()
        return c.grad

    def eval_grad_f(self, state, params, control):
        """Gradient wrt. control"""
        return self.grad_u(state, params, control) + self.beta * self.grad_E_V(state, params, control)

    def is_feasible(self,state,V,pol):
        return 1.0

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

                self.V_sample[s] = v


                if s % (self.state_sample.shape[0] / 10) == 0:
                    main_logger.debug(f"Finished solving Ipopt Problem #{s}")
            except NonConvergedError as e:
                main_logger.debug(
                    f"Optimization did not converge for: {str(self.state_sample[s, :])}"
                )
                # let's interpolate then the VF
                with torch.no_grad():
                    # get discrete state
                    d = int(self.state_sample[s, -1].item())
                    sample = self.state_sample[s, :-1]
                    sample = sample[self.get_d_cols(d)]
                    if not self.cfg["model"].get("ONLY_POLICY_ITER"):
                        self.V_sample[s] = (
                            (self.M[d][0](torch.unsqueeze(sample, 0)) )
                            .mean
                        )
                    else:
                        self.V_sample[s] = 0.0

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

    def state_iterate(self, state, params, control):
        """Default implementation, overridden in subclasses"""
        return state

    def sample_start_pts(self, state, params, n_restarts):
        """Default implementation, override in subclasses"""
        policy_sample = torch.zeros([n_restarts, self.control_dim])
        return policy_sample

    def scaling_vector(self, state):
        """Default implementation, override in subclasses"""
        return torch.ones(self.control_dim)

    def post_process_optimization(self, state, params, control, value, policy_old):
        return torch.from_numpy(control),torch.from_numpy(value)

    def pi(self, state):
        """Default implementation of state-transition matrix. Convention: rows sum to 1"""
        return torch.full(
            (self.discrete_state_dim, self.discrete_state_dim),
            1 / self.discrete_state_dim,
        )

    def state_iterate_exp(self, state, params, control):
        """How are future states generated from today state and control"""
        next_state = self.state_iterate(state, params, control)

        weights, points = self.expect.integration_rule(next_state)
        n_continous_samples = points.shape[0]

        # outer-product to calculate weights and points for discrete states
        pi = self.pi(state)
        pi_sliced = (
            torch.index_select(pi, 0, points[:, -1].to(torch.int64))
            .transpose(0, 1)
            .reshape(self.discrete_state_dim * n_continous_samples)
        )

        weights = torch.repeat_interleave(weights, self.discrete_state_dim) * pi_sliced
        points = torch.repeat_interleave(points, self.discrete_state_dim, 0)
        points[:, -1] = torch.arange(self.discrete_state_dim).repeat(
            n_continous_samples
        )

        return weights, points

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
                tmp_vec = (self.M[d][p](
                    (eval_pts[:,self.get_d_cols(d)])
                            ).mean)
                tmp_vec_target = self.combined_sample_all[mask,p]
                err[indxd] = torch.linalg.norm(
                    (
                        tmp_vec - tmp_vec_target
                    )/(1+torch.abs(tmp_vec_target)),ord=ord)

                scale = (
                    1.0
                    if ord == float("inf")
                    else 1 / (self.state_sample_all[mask,:].shape[0])
                )
                err[indxd] = scale * err[indxd]
        return err

    def save(self):
        import os
        import re

        save_path = (
            self.cfg.get("cwd", os.getcwd()) + "/Iter_" + str(self.epoch) + ".pth"
        )
        """ Saves the current state of the model """

        if not self.cfg.get("distributed") or dist.get_rank() == 0:
            torch.save(
                {
                    "beta": self.beta,
                    "epoch": self.epoch,
                    "discrete_state_dim": self.discrete_state_dim,
                    "policy_dim": self.policy_dim,
                    "model_state_dict": {
                        f"{d},{p}": self.M[d][p].state_dict()
                        for d in range(self.discrete_state_dim)
                        for p in range(1 + self.policy_dim)
                    },
                    "optimizer_state_dict": {
                        f"{d},{p}": self.optimizer[d][p].state_dict()
                        for d in range(self.discrete_state_dim)
                        for p in range(1 + self.policy_dim)
                    },
                    "loss": self.current_ll,
                    "cfg": self.cfg,
                    "state_sample_all": self.state_sample_all,
                    "non_converged_all": self.non_converged_all,
                    "feasible_all": self.feasible_all,
                    "combined_sample_all": self.combined_sample_all,
                    "rng_state": torch.get_rng_state(),
                    "metrics": self.metrics,
                },
                save_path,
            )

            m = re.match(
                r".*/runs/(?P<model>.*?)/(?P<run_day>.*?)/(?P<run_time>.*?)/(?P<checkpoint_name>.*)",
                save_path,
            )

            main_logger.info(f"Saving current state to {save_path}")
            main_logger.info(
                f"Post-process command:\n python post_process.py RUN_DIR={save_path}"
            )

    def bal_utility_func(self,eval_pt,discrete_state,target_p,rho,beta):
        eval_pt = (eval_pt[:,self.get_d_cols(discrete_state)])
        pred = self.M[discrete_state][target_p](
                    eval_pt
               )

        return rho * pred.mean  + beta / 2.0 * torch.log(pred.variance + 1e-10)

    def BAL(self):
        if self.cfg.get("distributed") and dist.get_rank() != 0:
            main_logger.debug(
                "Skipping Bayesian-Active learning sampling..."
            )
        else:
            main_logger.info(
                "Starting Bayesian-Active learning sampling..."
            )      
            new_sample = torch.empty((0,self.state_sample_all.shape[1]))
            for target in self.cfg["BAL"]["targets"]:
                # calculate BAL utility for each of the new sample
                bal_utility = -1.0e10*torch.ones(self.state_sample.shape[0])
                non_empty_vec = True

                target_d = target.get("discrete_state", -1)
                d_range = [target_d] if target_d != -1 else range(self.discrete_state_dim)
                target_p = target.get("policy")

                for discrete_state in d_range:
                    self.M[discrete_state][target_p].eval()

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    for s_ind in range(self.state_sample.shape[0]):
                        discrete_state = int(self.state_sample[s_ind, -1].item())
                        if discrete_state in d_range:
                            non_empty_vec = True
                            eval_pt = self.state_sample.to(self.device)[s_ind : (s_ind + 1), :-1]
                            bal_utility[s_ind] = self.bal_utility_func(eval_pt,discrete_state,target_p,target.get("rho"),target.get("beta"))

                if non_empty_vec:
                    new_sample = torch.cat(
                        (
                            new_sample,
                            torch.index_select(
                                self.state_sample,
                                0,
                                torch.argsort(bal_utility, descending=True)[
                                    : self.cfg["BAL"].get("points_per_iter")
                                ],
                            ),
                        ),
                        dim=0,
                    )

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

    @classmethod
    def load(cls, path, cfg_override, **kwargs):
        checkpoint = torch.load(path)
        cf = checkpoint["cfg"]
        cf.update(cfg_override)
        dpgp = cls(
            beta=checkpoint["beta"],
            cfg=cf,
            state_sample=checkpoint["state_sample_all"],
            V_sample=checkpoint["combined_sample_all"][:, 0],
            metrics=checkpoint.get("metrics", {}),
            policy_sample=checkpoint["combined_sample_all"][:, 1:],
            **kwargs
        )
        dpgp.combined_sample_all=checkpoint["combined_sample_all"]
        dpgp.state_sample_all = checkpoint["state_sample_all"]
        dpgp.non_converged_all = checkpoint["non_converged_all"]
        dpgp.feasible_all = checkpoint["feasible_all"]
        # dpgp.feasible_all[:] = 1.0
        dpgp.current_ll = checkpoint["loss"]
        dpgp.epoch = checkpoint["epoch"]
        def get_d_rows(d, drop_non_converged=True):
            d_selected = dpgp.state_sample_all[:, -1] == d
            if drop_non_converged:
                return torch.logical_and(d_selected, dpgp.state_sample_all < 1)
            else:
                return d_selected
            
        for d in range(dpgp.discrete_state_dim):
            for p in range(1 + dpgp.policy_dim):
                dpgp.M[d][p] = dpgp.create_model(
                    d, 
                    p, 
                    dpgp.state_sample_all[get_d_rows(d, drop_non_converged=dpgp.cfg.get("drop_non_converged")),:-1,][:,dpgp.get_d_cols(d)], 
                    dpgp.combined_sample_all[get_d_rows(d, drop_non_converged=dpgp.cfg.get("drop_non_converged")), p,])
                dpgp.M[d][p].load_state_dict(checkpoint["model_state_dict"][f"{d},{p}"])
                dpgp.M[d][p].eval()
                try:
                    dpgp.optimizer[d][p].load_state_dict(
                       checkpoint["optimizer_state_dict"][f"{d},{p}"]
                    )
                except:
                    main_logger.info("Unable to overwrite optimizer from file; proceed without it.")

                if cfg_override.get("override_checkpoint_configs_optimizer", False):
                    for param_group in dpgp.optimizer[d][p].param_groups:
                        old_lr = param_group.get('lr', None)
                        if old_lr:
                            new_lr = cfg_override.get('torch_optim', {}).get('config', {}).get('lr', None)
                            if new_lr and abs(old_lr - new_lr) > 1e-6:
                                param_group['lr'] = new_lr
                                main_logger.info(f'd{d}, p{p}: updated learning rate from {old_lr} to {new_lr}')

        torch.set_rng_state(checkpoint["rng_state"])

        return dpgp

    def scatter_sample(self):
        if self.cfg.get("distributed"):
            # use the samples from the root rank
            if dist.get_rank() == 0:
                self.state_sample_all_shape = torch.tensor([self.state_sample_all.shape[0], self.state_sample_all.shape[1]])
            else:
                self.state_sample_all_shape = torch.tensor([0, 0])
            dist.broadcast(self.state_sample_all_shape, src=0)
            # Resize the state_sample_all tensor on nonzero rank processes,
            # as it can change shape if BAL is enabled, which only runs on rank 0,
            # and broadcast requires the same shape
            if dist.get_rank() != 0:
                self.state_sample_all = torch.zeros(self.state_sample_all_shape[0], self.state_sample_all_shape[1])
                self.combined_sample_all = torch.zeros(self.state_sample_all_shape[0], 1 + self.policy_dim)
                self.feasible_all = torch.zeros(self.state_sample_all_shape[0])
                self.non_converged_all = torch.zeros(self.state_sample_all_shape[0])

            dist.broadcast(self.state_sample_all, src=0)
            dist.broadcast(self.combined_sample_all, src=0)
            dist.broadcast(self.feasible_all, src=0)
            dist.broadcast(self.non_converged_all, src=0)

            points_per_worker = self.state_sample_all.shape[0] / dist.get_world_size()
            self.worker_slice = [
                A
                for A in range(self.state_sample_all.shape[0])
                if int(A / points_per_worker) == dist.get_rank()
            ]
            # subset state to given rank
            self.state_sample = self.state_sample_all[
                min(self.worker_slice) : (max(self.worker_slice) + 1), :
            ]
            self.combined_sample = self.combined_sample_all[
                min(self.worker_slice) : (max(self.worker_slice) + 1), :
            ]
            self.feasible = self.feasible_all[
                min(self.worker_slice) : (max(self.worker_slice) + 1)
            ]
            self.non_converged = self.non_converged_all[
                min(self.worker_slice) : (max(self.worker_slice) + 1)
            ]
        else:
            self.non_converged = self.non_converged_all
            self.feasible = self.feasible_all
            self.combined_sample = self.combined_sample_all
            self.state_sample = self.state_sample_all

    def sample_all(self, init_sample=None):
        if init_sample is None:
            if self.epoch == 0:
                self.state_sample, self.feasible = self.sample()
                self.combined_sample = torch.zeros([self.state_sample.shape[0],self.policy_dim+1])
            else:
                self.prev_state_sample = self.state_sample_all.clone()
                self.prev_combined_sample = self.combined_sample_all.clone()
                self.prev_feasible = self.feasible_all.clone()
                if self.cfg.get("resample_method") == "random":
                    self.sample(self.cfg.get("resample_num_new",1000))
                elif self.cfg.get("resample_method") == "dynamic":
                    # Pull a new state accoriding to weights randomly
                    # Use method: https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/14
                    # sample next state
                    new_state_sample = torch.tensor([])
                    for s_ind in range(self.state_sample_all.shape[0]):
                        if self.prev_feasible[s_ind] == 1:
                            params = self.get_params(
                                    self.state_sample_all[s_ind],
                                    self.policy_sample_all[s_ind][: self.policy_dim])
                            weights, points = self.state_iterate_exp(
                                self.state_sample_all[s_ind],
                                params,
                                self.policy_sample_all[s_ind][: self.policy_dim],
                            )
                            if self.cfg.get("resample_num_new") < 0:
                                new_state_sample = torch.cat(
                                    (new_state_sample, points), dim=0
                                )
                            elif (
                                self.cfg.get("resample_num_new") <= weights.shape[0]
                            ):
                                sample_idx = weights.multinomial(
                                    num_samples=self.cfg.get("resample_num_new"),
                                    replacement=False,
                                )
                                new_state_sample = torch.cat(
                                    (new_state_sample, points[sample_idx]), dim=0
                                )
                            else:
                                sys.stderr.write(
                                    f"Config value of resample_num_new must not exceed the number of possible future states ({int(weights.shape[0])})"
                                )
                                sys.exit(1)

                    self.state_sample = new_state_sample
                elif self.cfg.get("resample_method") == "disabled":
                    self.state_sample = self.state_sample_all
                    self.feasible = self.feasible_all
                    self.combined_sample = self.combined_sample_all
                else:
                    sys.stderr.write(
                        f'Unrecognised config value for resample_method: {self.cfg.get("resample_method")}'
                    )
                    sys.exit(1)
                # do Bayesian Active Learning
                if self.cfg["BAL"].get("enabled"):
                    if (
                        self.epoch % self.cfg["BAL"].get("epoch_freq", 5) == 0
                        and (
                            self.cfg["BAL"]["max_points"] < 0
                            or self.cfg["BAL"]["max_points"]
                            > self.prev_state_sample.shape[0]
                        )
                    ) and self.epoch > 1:
                        if self.cfg.get("distributed") and dist.get_rank() != 0:
                            main_logger.debug(
                                "Skipping Bayesian-Active learning sampling..."
                            )
                        else:
                            main_logger.info(
                                "Starting Bayesian-Active learning sampling..."
                            )

                        self.BAL()
                    else:
                        # if not doing BAL accumulation, then retain sample
                        self.state_sample = self.prev_state_sample
                        self.combined_sample = self.prev_combined_sample
                        self.feasible = self.prev_feasible
                else:
                    self.feasible = self.prev_feasible.to(self.device)
                    self.state_sample = self.prev_state_sample.to(self.device)
                    self.combined_sample = self.prev_combined_sample.to(self.device)

        else:
            self.state_sample = init_sample.to(self.device)
            self.feasible = torch.ones(init_sample.shape[0]).to(self.device)
            self.combined_sample = torch.empty([self.state_sample.shape[0],1 + self.policy_dim])

        # the sampling is always for the complete population
        self.state_sample_all = self.state_sample.to(self.device)

        self.combined_sample_all = self.combined_sample.to(self.device)

        self.feasible_all = self.feasible.to(self.device)

        # non-convered points
        self.non_converged_all = torch.zeros(self.state_sample.shape[0]).to(self.device)

        # distribute the samples
        self.scatter_sample()

    def policy_guess(self, state):
        return torch.zeros(self.policy_dim)

    def u(self, state, control):
        return torch.sum(control) * torch.zeros(1)

    def predict_policies(self, state):
        return torch.tensor(
            [
                (
                    self.M[int(state[-1].item())][1 + p](
                        (torch.unsqueeze(state[:-1], dim=0))
                    )
                ).mean.item()
                for p in range(self.policy_dim[int(state[-1].item())])
            ]
        )

    # Expected to be defined in the subclass
    @abstractmethod
    def sample(self):
        return None,None,None

    @abstractmethod
    def solve(self, state):
        """Do a VFI iteration at state"""
        return 0
