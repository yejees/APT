"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from torchvision import utils
from copy import deepcopy
from torch.autograd import grad
from sklearn.linear_model import OrthogonalMatchingPursuitCV as OMP
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from scipy.signal import butter, filtfilt
import os
import torch.fft as fft
from bezier_curve import nonlinear_transformation


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    
    def p_sample_loop_grad(
        self,
        model,
        c_model,
        adapter,
        shape,
        data,
        mask,
        start_idx,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
 
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_grad(
            model,
            c_model,
            adapter,
            shape,
            data,
            mask,
            start_idx,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    def p_sample_loop_progressive_grad(
        self,
        model,
        c_model,
        adapter,
        shape,
        condition,
        mask,
        start_idx,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
            
        learning_rate = th.linspace(1, 1, steps = len(indices))
        
        selected_channels = th.where(mask[:, :, 0, 0] == 1)[1]
        selected_channels_zero = th.where(mask[:, :, 0, 0] == 0)[1]
        
        
        if len(selected_channels_zero) == 2:
            if selected_channels_zero[0] == 2 and selected_channels_zero[1] == 3:
                c = condition[:,1:2,:,:].clone().detach()
                c = c*0.5 + 0.5
                c1 = c * 0.5
                c1 = c1*2 - 1
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c1, th.tensor([start_idx]*shape[0],device=device))
                
                selected_c = condition[0,1:2,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2 * 1.3
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
        
            elif selected_channels_zero[0] == 1 and selected_channels_zero[1] == 3:
                c = condition[:,2:3,:,:].clone().detach()
                c = c*0.5 + 0.5
                c1 = c * 1.7
                c1 = c1*2 - 1
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c1, th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[1],:,:] = self.q_sample(condition[:,0:1,:,:], th.tensor([start_idx]*en_shape[0],device=device))
            
            elif selected_channels_zero[0] == 1 and selected_channels_zero[1] == 2:
                
                selected_c = condition[0,3:4,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2*0.8
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
                
                c = condition[:,0:1,:,:].clone().detach()
                c = c*0.5 + 0.5
                c22 = c * 0.8
                c22 = c22*2 - 1
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c22, th.tensor([start_idx]*shape[0],device=device))
            
            elif selected_channels_zero[0] == 0 and selected_channels_zero[1] == 3:
                c = condition[:,2:3,:,:].clone().detach()
                c = c*0.5 + 0.5
                c1 = c * 1.3
                c1 = c1*2 - 1
                
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c1, th.tensor([start_idx]*shape[0],device=device))
                
                selected_c = condition[0,1:2,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2 * 1.2
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))

            elif selected_channels_zero[0] == 0 and selected_channels_zero[1] == 2:
                img[:,selected_channels_zero[0],:,:] = self.q_sample(condition[:,3:4,:,:], th.tensor([start_idx]*shape[0],device=device))
                
                c = condition[:,1:2,:,:].clone().detach()
                c = c*0.5 + 0.5
                c22 = c * 0.5
                c22 = c22*2 - 1
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c22, th.tensor([start_idx]*shape[0],device=device))

            elif selected_channels_zero[0] == 0 and selected_channels_zero[1] == 1:
                img[:,selected_channels_zero[0],:,:] = self.q_sample(condition[:,3:4,:,:], th.tensor([start_idx]*shape[0],device=device))
                
                selected_c = condition[0,3:4,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2 * 0.85
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,selected_channels_zero[1],:,:] = self.q_sample(condition[:,2:3,:,:], th.tensor([start_idx]*shape[0],device=device))
        
        if len(selected_channels_zero) == 3:        
            if selected_channels_zero[0] == 1 and selected_channels_zero[1] == 2 and selected_channels_zero[2] == 3:
                c = condition[:,0:1,:,:].clone().detach()
                c = c*0.5 + 0.5
                
                c1 = c * 1.3
                c1 = c1*2 - 1
                
                c2 = c * 0.75
                c2 = c2*2 - 1
                
                c3 = c.clone().detach()
                c3 = c3*2 - 1
                
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c1, th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[2],:,:] = self.q_sample(c3, th.tensor([start_idx]*shape[0],device=device))
            
            elif selected_channels_zero[0] == 0 and selected_channels_zero[1] == 2 and selected_channels_zero[2] == 3:
                c = condition[:,1:2,:,:].clone().detach()
                c = c*0.5 + 0.5
                
                c1 = c * 0.8
                c1 = c1*2 - 1
                
                c2 = c * 0.5
                c2 = c2*2 - 1
                
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c1, th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
                
                selected_c = condition[0,1:2,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2 * 1.3
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,selected_channels_zero[2],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
            
            elif selected_channels_zero[0] == 0 and selected_channels_zero[1] == 1 and selected_channels_zero[2] == 3:
                c = condition[:,2:3,:,:].clone().detach()
                c = c*0.5 + 0.5    

                c1 = c * 1.3
                c1 = c1*2 - 1
                
                c2 = c * 1.7
                c2 = c2*2 - 1
                
                c3 = c * 1.2
                c3 = c3*2 - 1
                
                img[:,selected_channels_zero[0],:,:] = self.q_sample(c1, th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[2],:,:] = self.q_sample(c3, th.tensor([start_idx]*shape[0],device=device))
            
            elif selected_channels_zero[0] == 0 and selected_channels_zero[1] == 1 and selected_channels_zero[2] == 2:
                c = condition[:,3:4,:,:].clone().detach()
                c = c*0.5 + 0.5
                
                c3 = c * 0.8
                c3 = c3*2 - 1
                img[:,selected_channels_zero[0],:,:] = self.q_sample(condition[:,3:4,:,:], th.tensor([start_idx]*shape[0],device=device))
                img[:,selected_channels_zero[2],:,:] = self.q_sample(c3, th.tensor([start_idx]*shape[0],device=device))

                selected_c = condition[0,3:4,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,selected_channels_zero[1],:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
                

        if len(selected_channels_zero) == 1:
            if selected_channels_zero[0] == 0:
                c2 = condition[:,3:4,:,:].clone().detach()
                img[:,0:1,:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
            elif selected_channels_zero[0] == 1:
                selected_c = condition[0,3:4,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = c2*0.8
                c2 = th.from_numpy(c2).to(device)
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,1:2,:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
            elif selected_channels_zero[0] == 2:
                selected_c = condition[:,1:2,:,:].clone().detach()
                c2 = selected_c*0.5 + 0.5
                c2 = c2*0.5
                c2 = c2*2 - 1
                img[:,2:3,:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))

            else:
                selected_c = condition[0,1:2,:,:].clone().detach()
                c_shift = selected_c*0.5 + 0.5
                c_shift = c_shift.cpu().numpy()
                c_shift = c_shift.astype(np.float32)
                c1, c2 = nonlinear_transformation(c_shift)
                c2 = (c2 - np.min(c2)) 
                mask_c2 = (c1>0.04).astype(np.float32)
                c2 = c2*mask_c2
                c2 = c2*1.2
                c2 = th.from_numpy(c2).to(device)
                c2 = c2*2 - 1
                c2 = c2.unsqueeze(0).unsqueeze(0)
                img[:,3:4,:,:] = self.q_sample(c2, th.tensor([start_idx]*shape[0],device=device))
        
        for i in indices:
            if i > start_idx:
                continue
            
            t = th.tensor([i] * shape[0], device=device)
            t_1 = th.tensor([i-1] * shape[0], device=device)

                
            img1 = img.clone().detach()
            img1 = img1.requires_grad_()
            
            total_grad = th.zeros_like(img)  

                
            selected_channels = th.where(mask[:, :, 0, 0] == 1)[1]
            selected_channels_zero = th.where(mask[:, :, 0, 0] == 0)[1]
            real_cond = condition[:, selected_channels, :, :]

            batch, _, h, w = img1.shape
            mul_cond = real_cond.repeat(batch,1,1,1)
            _, num, _, _ = real_cond.shape
            cond_flatten  = mul_cond.view(batch*num,1, h,w)
            cond_flatten_r = cond_flatten.repeat(1,3,1,1)
            upsampled_size = (224, 224)

            with th.no_grad():
                cond_flatten_r_resize = F.interpolate(cond_flatten_r, size=upsampled_size, mode='bilinear', align_corners=False)
                cond_features = c_model.encode_image(cond_flatten_r_resize)
                cond_features_reshape =  cond_features.view(batch,num,-1)
                cond_features_reshape = adapter(cond_features_reshape, cond_features_reshape, train=False) #(batch, 512)
                cond_features_reshape = cond_features_reshape.unsqueeze(1)
                model_kwargs = {}
                model_kwargs['cond'] = cond_features_reshape
                        
            
            out = self.p_sample(
                model,
                img1,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
                
            
            det_loss = 20 * th.norm((out["pred_xstart"]*mask - condition* mask)) 
            total_grad = grad(outputs=det_loss, inputs=img1)[0]

            img_out = out["pred_xstart"] - learning_rate[i]*total_grad
            
            out["pred_xstart"] = img_out*(1-mask) + condition*mask 

            if i-1 <0:
                sample = out["pred_xstart"]
            else:
                sample = self.q_sample(out["pred_xstart"],t_1)
                
            out["sample"] = sample.clone().detach()
            img = out["sample"]
        yield out  
        
        
    def p_sample_loop_last(
        self,
        first_xstart,
        model,
        c_model,
        adapter,
        shape,
        data,
        mask,
        start_idx,
        total_loop_num,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
 
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_last(
            first_xstart,
            model,
            c_model,
            adapter,
            shape,
            data,
            mask,
            start_idx,
            total_loop_num,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]
    
    def p_sample_loop_progressive_last(
        self,
        first_xstart,
        model,
        c_model,
        adapter,
        shape,
        condition,
        mask,
        start_idx,
        total_loop_num,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
            
        last_loop_num = total_loop_num - start_idx
        
        indices = list(range(last_loop_num))
        indices_loop = list(range(self.num_timesteps))[::-1]
        
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
            
        learning_rate = th.linspace(1, 1, steps = len(indices))
        weight_avg = th.linspace(0.5, 0.5, steps = len(indices))
        
        
        img = first_xstart.clone().detach()
        pred_xstart_0 = first_xstart.clone().detach() 

        t = th.tensor([start_idx] * shape[0], device=device)
        t_1 = th.tensor([start_idx-1] * shape[0], device=device)
        img = img*(1-mask) + self.q_sample(condition,t)*mask
        
        for i in indices:

            img1 = img.clone().detach()
            img1 = img1.requires_grad_()
            
            total_grad = th.zeros_like(img)  
                
            selected_channels = th.where(mask[:, :, 0, 0] == 1)[1]
            real_cond = condition[:, selected_channels, :, :]

            batch, _, h, w = img1.shape
            mul_cond = real_cond.repeat(batch,1,1,1)
            _, num, _, _ = real_cond.shape
            cond_flatten  = mul_cond.view(batch*num,1, h,w)
            cond_flatten_r = cond_flatten.repeat(1,3,1,1)
            upsampled_size = (224, 224)

            with th.no_grad():
                cond_flatten_r_resize = F.interpolate(cond_flatten_r, size=upsampled_size, mode='bilinear', align_corners=False)
                cond_features = c_model.encode_image(cond_flatten_r_resize)
                cond_features_reshape =  cond_features.view(batch,num,-1)
                cond_features_reshape = adapter(cond_features_reshape, cond_features_reshape, train=False) #(batch, 512)
                cond_features_reshape = cond_features_reshape.unsqueeze(1)
                model_kwargs = {}
                model_kwargs['cond'] = cond_features_reshape
            
        
            out = self.p_sample(
                model,
                img1,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
                
            img_out = out["pred_xstart"].clone().detach()

            det_loss = 20 * th.norm((out["pred_xstart"]*mask - condition* mask))
            total_grad = grad(outputs=det_loss, inputs=img1)[0]
            img_out = out["pred_xstart"] - learning_rate[i]*total_grad

            img_out = pred_xstart_0*(1-weight_avg[i])+img_out*weight_avg[i]

            out["pred_xstart"] = img_out.clone().detach()
            pred_xstart_0 = out["pred_xstart"].clone().detach() 

            
            if i == len(indices) -1:
                sample = self.q_sample(out["pred_xstart"],t_1)
            else:
                sample = self.q_sample(out["pred_xstart"],t)
            out["sample"] = sample.clone().detach()
            
            img = out["sample"]
        
        for i in indices_loop:
            if i > start_idx-1:
                continue
            img1 = img.clone().detach()
            img1 = img1.requires_grad_()
            t = th.tensor([i] * shape[0], device=device)
            t_1 = th.tensor([i-1] * shape[0], device=device)
            
            out = self.p_sample(
                model,
                img1,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
            )
            det_loss = 20 * th.norm((out["pred_xstart"]*mask - condition* mask))
            total_grad = grad(outputs=det_loss, inputs=img1)[0]
            

            img_out = out["pred_xstart"] - total_grad
            out["pred_xstart"] = img_out.clone().detach()

            if i > 0:
                out["sample"] = self.q_sample(out["pred_xstart"],t_1)
            else:
                out["sample"] = out["pred_xstart"].clone().detach()
            img = out["sample"].clone().detach()   
        yield out

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        data,
        mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            data,
            mask,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        data,
        mask,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            img = img * (1-mask) + self.q_sample(data,t) * mask
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                img = out["sample"]
                img_out = out["pred_xstart"]
                utils.save_image(img_out[:,3:4], str(f'./sample_results/t2_1599_samples_ddim/t2_{i}.png'), nrow = 4, normalize=True)


    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, c_model, adapter, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
            
        x_t = self.q_sample(x_start, t, noise=noise)

        ###########global feature extraction##############
        batch, ch, he, wi = x_start.shape
        x_start_flatten = x_start.view(batch*ch,1,he,wi)
        x_start_flatten_r = x_start_flatten.repeat(1,3,1,1)
        upsampled_size = (224, 224)
        x_start_itp = F.interpolate(x_start_flatten_r, size=upsampled_size, mode='bilinear', align_corners=False)
        
        with th.no_grad():
            image_features = c_model.module.encode_image(x_start_itp)
            img_features = image_features.view(batch,ch,-1)
            img_features = adapter(img_features, img_features, train=False) #(batch, 512)
            img_features = img_features.unsqueeze(1)
        model_kwargs['cond'] = img_features
       
        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape
            
            terms["mse"] = mean_flat((target - model_output) ** 2)
            
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
