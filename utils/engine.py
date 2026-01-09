from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# 全局变量
V_PRED = False

def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("signal_rate", torch.sqrt(alpha_t_bar)) # $ \sqrt{\bar{\alpha_t}} $  the coefficient of $x_0$
        self.register_buffer("noise_rate", torch.sqrt(1.0 - alpha_t_bar)) # $ \sqrt{1 - \bar{\alpha_t}} $ the coefficient of noise($\epsilon$)

    def forward(self, x_0):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        # generate Gaussian noise $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x_0) 

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (extract(self.signal_rate, t, x_0.shape) * x_0 +
               extract(self.noise_rate, t, x_0.shape) * epsilon)

        if V_PRED:
            # -------- using v-prediction --------
            v = extract(self.signal_rate, t, x_0.shape) * epsilon - extract(self.noise_rate, t, x_0.shape) * x_0            # <<< CHANGED

            v_theta = self.model(x_t, t)                   # <<< CHANGED

            loss = F.mse_loss(v_theta, v, reduction="none")# <<< CHANGED

        else:
            # ----------- using epsilon-prediction -----------
            epsilon_theta = self.model(x_t, t)

            # get the gradient
            loss = F.mse_loss(epsilon_theta, epsilon, reduction="none")
        
        # --------- same ---------
        loss = torch.sum(loss)
        return loss


class DDPMSampler(nn.Module):
    def __init__(self, model: nn.Module, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        self.register_buffer("beta_t", torch.linspace(*beta, T, dtype=torch.float32))

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)
        alpha_t_bar_prev = F.pad(alpha_t_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("alpha_t_bar", alpha_t_bar)

        self.register_buffer("coeff_1", torch.sqrt(1.0 / alpha_t))
        self.register_buffer("coeff_2", self.coeff_1 * (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_t_bar))
        self.register_buffer("posterior_variance", self.beta_t * (1.0 - alpha_t_bar_prev) / (1.0 - alpha_t_bar))

    @torch.no_grad()
    def cal_mean_variance(self, x_t, t):
        """
        Calculate the mean and variance for $q(x_{t-1} | x_t, x_0)$
        """
        # epsilon_theta = self.model(x_t, t) # original

        # predict noise using model
        if V_PRED:
            # ------------- using v-prediction -------------
            alpha_t_bar = extract(self.alpha_t_bar, t, x_t.shape)
            v_theta = self.model(x_t, t)
            epsilon_theta = torch.sqrt(alpha_t_bar) * v_theta + torch.sqrt(1-alpha_t_bar) * x_t
        else:
            # ------------- using epsilon-prediction -------------
            epsilon_theta = self.model(x_t, t)

        mean = extract(self.coeff_1, t, x_t.shape) * x_t - extract(self.coeff_2, t, x_t.shape) * epsilon_theta

        # var is a constant
        var = extract(self.posterior_variance, t, x_t.shape)

        return mean, var

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int):
        """
        Calculate $x_{t-1}$ according to $x_t$
        """
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        mean, var = self.cal_mean_variance(x_t, t)

        z = torch.randn_like(x_t) if time_step > 0 else 0
        x_t_minus_one = mean + torch.sqrt(var) * z

        if torch.isnan(x_t_minus_one).int().sum() != 0:
            raise ValueError("nan in tensor!")

        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, only_return_x_0: bool = True, interval: int = 1, **kwargs):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.
            kwargs: no meaning, just for compatibility.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        x = [x_t]
        with tqdm(reversed(range(self.T)), colour="#6565b5", total=self.T) as sampling_steps:
            for time_step in sampling_steps:
                x_t = self.sample_one_step(x_t, time_step)

                if not only_return_x_0 and ((self.T - time_step) % interval == 0 or time_step == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": time_step + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]


class DDIMSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        # generate T steps of beta
        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - beta_t
        self.register_buffer("alpha_t_bar", torch.cumprod(alpha_t, dim=0))

    @torch.no_grad()
    def sample_one_step(self, x_t, time_step: int, prev_time_step: int, eta: float):
        t = torch.full((x_t.shape[0],), time_step, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_time_step, device=x_t.device, dtype=torch.long)

        # get current and previous alpha_cumprod
        alpha_t_bar = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_bar_prev = extract(self.alpha_t_bar, prev_t, x_t.shape)

        # predict noise using model
        if V_PRED:
            # ------------- using v-prediction -------------
            v_theta = self.model(x_t, t)
            epsilon_theta_t = torch.sqrt(alpha_t_bar) * v_theta + torch.sqrt(1-alpha_t_bar) * x_t
        else:
            # ------------- using epsilon-prediction -------------
            epsilon_theta_t = self.model(x_t, t)

        # calculate x_{t-1}
        sigma_t = eta * torch.sqrt((1 - alpha_t_bar_prev) / (1 - alpha_t_bar) * (1 - alpha_t_bar / alpha_t_bar_prev))
        epsilon_t = torch.randn_like(x_t)
        x_t_minus_one = (
                torch.sqrt(alpha_t_bar_prev / alpha_t_bar) * x_t +
                (torch.sqrt(1 - alpha_t_bar_prev - sigma_t ** 2) - torch.sqrt(
                    (alpha_t_bar_prev * (1 - alpha_t_bar)) / alpha_t_bar)) * epsilon_theta_t +
                sigma_t * epsilon_t
        )
        return x_t_minus_one

    @torch.no_grad()
    def forward(self, x_t, steps: int = 1, method="linear", eta=0.0,
                only_return_x_0: bool = True, interval: int = 1,**kwargs):
        """
        Parameters:
            x_t: Standard Gaussian noise. A tensor with shape (batch_size, channels, height, width).
            steps: Sampling steps.
            method: Sampling method, can be "linear" or "quadratic".
            eta: Coefficients of sigma parameters in the paper. The value 0 indicates DDIM, 1 indicates DDPM.
            only_return_x_0: Determines whether the image is saved during the sampling process. if True,
                intermediate pictures are not saved, and only return the final result $x_0$.
            interval: This parameter is valid only when `only_return_x_0 = False`. Decide the interval at which
                to save the intermediate process pictures, according to `step`.
                $x_t$ and $x_0$ will be included, no matter what the value of `interval` is.

        Returns:
            if `only_return_x_0 = True`, will return a tensor with shape (batch_size, channels, height, width),
            otherwise, return a tensor with shape (batch_size, sample, channels, height, width),
            include intermediate pictures.
        """
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(int)
        else:
            raise NotImplementedError(f"sampling method {method} is not implemented!")

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        time_steps = time_steps + 1
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        with tqdm(reversed(range(0, steps)), colour="#6565b5", total=steps) as sampling_steps:
            for i in sampling_steps:
                x_t = self.sample_one_step(x_t, time_steps[i], time_steps_prev[i], eta)

                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                sampling_steps.set_postfix(ordered_dict={"step": i + 1, "sample": len(x)})

        if only_return_x_0:
            return x_t  # [batch_size, channels, height, width]
        return torch.stack(x, dim=1)  # [batch_size, sample, channels, height, width]

class DPMSolverSampler(nn.Module):
    def __init__(self, model, beta: Tuple[int, int], T: int):
        super().__init__()
        self.model = model
        self.T = T

        beta_t = torch.linspace(*beta, T, dtype=torch.float32)
        alpha_t = 1.0 - beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        self.register_buffer("alpha_t_bar", alpha_t_bar)
        self.register_buffer(
            "lambda_t",
            0.5 * (torch.log(alpha_t_bar) - torch.log1p(-alpha_t_bar))
        ) 
        ''' 
        $\lambda_t = \log{ \sqrt{ \frac{\bar{\alpha_t}}{1-\bar{\alpha_t}} } } 
                   = \frac{1}{2} (\log{ \bar{\alpha_t} } - \log{ 1 - \bar{\alpha_t} }) $
        '''

    @torch.no_grad()
    def dpm_solver_1_step(self, x_t, eps_t, t, t_prev):
        """
        DPM-Solver-1 (first-order)
        """
        alpha_t_bar = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_bar_prev = extract(self.alpha_t_bar, t_prev, x_t.shape)

        lambda_t = extract(self.lambda_t, t, x_t.shape)
        lambda_prev = extract(self.lambda_t, t_prev, x_t.shape)

        h = lambda_prev - lambda_t

        x_prev = (
            torch.sqrt(alpha_t_bar_prev / alpha_t_bar) * x_t
            - torch.sqrt(1 - alpha_t_bar_prev) * (torch.exp(h) - 1.0) * eps_t
        )
        return x_prev
    
    @torch.no_grad()
    def dpm_solver_2_step(self, x_t, eps_t, eps_prev, t, t_prev):
        """
        2nd order step
        """
        alpha_t_bar = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_bar_prev = extract(self.alpha_t_bar, t_prev, x_t.shape)
        
        lambda_t = extract(self.lambda_t, t, x_t.shape)
        lambda_prev = extract(self.lambda_t, t_prev, x_t.shape)

        h = lambda_prev - lambda_t

        r = (torch.exp(h) - 1.0)/h
        eps_hat = eps_t + r * (eps_t - eps_prev)

        x_prev = (
            torch.sqrt(alpha_t_bar_prev/alpha_t_bar) * x_t
            - torch.sqrt(1 - alpha_t_bar_prev) * (torch.exp(h) - 1.0) * eps_hat
        )
        return x_prev

    @torch.no_grad()
    def dpm_solver_3_step(self, x_t, eps_t, eps_prev, eps_prev2, t, t_prev):
        """
        3rd order step
        """
        alpha_t_bar = extract(self.alpha_t_bar, t, x_t.shape)
        alpha_t_bar_prev = extract(self.alpha_t_bar, t_prev, x_t.shape)

        lambda_t = extract(self.lambda_t, t, x_t.shape)
        lambda_prev = extract(self.lambda_t, t_prev, x_t.shape)

        h = lambda_prev - lambda_t

        eps_hat = (
            eps_t
            + 0.5 * (eps_t - eps_prev)
            + (h**2)/6.0 * (eps_t - 2*eps_prev + eps_prev2)
        )

        x_prev = (
            torch.sqrt(alpha_t_bar_prev/alpha_t_bar) * x_t
            - torch.sqrt(1 - alpha_t_bar_prev) * (torch.exp(h) - 1.0) * eps_hat
        )
        return x_prev
    @torch.no_grad()
    def forward(
        self,
        x_t,
        steps: int = 50,
        method="quadratic",
        only_return_x_0=True,
        interval=1,
        solver_order = 1, # 1 or 2 or 3
        **kwargs
    ):
        # timestep schedule
        if method == "linear":
            a = self.T // steps
            time_steps = np.asarray(list(range(0, self.T, a)))
        elif method == "quadratic":
            time_steps = (np.linspace(0, np.sqrt(self.T * 0.8), steps) ** 2).astype(int)
        else:
            raise NotImplementedError

        time_steps = time_steps + 1
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        x = [x_t]
        eps_history = [] # to store epsilons

        print(f"solver_order: {solver_order}")
        with tqdm(reversed(range(steps)), total=steps, colour="#6565b5") as pbar:
            for i in pbar:
                t = torch.full((x_t.shape[0],), time_steps[i], device=x_t.device, dtype=torch.long)
                t_prev = torch.full((x_t.shape[0],), time_steps_prev[i], device=x_t.device, dtype=torch.long)
                
                # model predicts epsilon
                # eps_t = self.model(x_t, t) # original
                
                if V_PRED:
                    alpha_t_bar = extract(self.alpha_t_bar, t, x_t.shape)

                    # ------------- using v-prediction -------------
                    v_theta = self.model(x_t, t)
                    eps_t = torch.sqrt(alpha_t_bar) * v_theta + torch.sqrt(1-alpha_t_bar) * x_t
                else:
                    # ------------- using epsilon-prediction -------------
                    eps_t = self.model(x_t, t)

                # get x_t
                if solver_order == 1:
                    x_t = self.dpm_solver_1_step(x_t, eps_t, t, t_prev)
                    # no need to store epsilons

                elif solver_order == 2:
                    if  len(eps_history) == 0:
                        x_t = self.dpm_solver_1_step(x_t, eps_t, t, t_prev)
                    else:
                        x_t = self.dpm_solver_2_step(x_t, eps_t, eps_history[-1], t, t_prev)
                    
                    eps_history.append(eps_t)
                    if len(eps_history) > 1:
                        eps_history.pop(0) # keep only last one epsilon
                
                elif solver_order == 3:
                    if  len(eps_history) == 0:
                        x_t = self.dpm_solver_1_step(x_t, eps_t, t, t_prev)
                    elif len(eps_history) == 1:
                        x_t = self.dpm_solver_2_step(x_t, eps_t, eps_history[-1], t, t_prev)
                    else:
                        x_t = self.dpm_solver_3_step(x_t, eps_t, eps_history[-1], eps_history[-2], t, t_prev)
                    
                    eps_history.append(eps_t)
                    if len(eps_history) > 2:
                        eps_history.pop(0) # keep only last two epsilons
                
                else:
                    NotImplementedError
                
                if not only_return_x_0 and ((steps - i) % interval == 0 or i == 0):
                    x.append(torch.clip(x_t, -1.0, 1.0))

                pbar.set_postfix(step=i + 1, sample=len(x))


        if only_return_x_0:
            return x_t
        return torch.stack(x, dim=1)
