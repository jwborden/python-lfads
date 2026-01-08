from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Poisson


def stable_kl(input_dist: Tensor, target_dist: Tensor, sm_dim: int = 1) -> Tensor:
    """
    Numerically stable KL divergence
    assumes distributions range from -infinity to infinity and don't start in log space
    the default softmax dim (sm_dim) of 1 assumes shape (B, D) or (B, D, T)
    """
    assert input_dist.shape == target_dist.shape, "distribution shapes must batch for kl div"
    input_dist = F.softmax(input_dist, dim=sm_dim).log()
    target_dist = F.softmax(target_dist, dim=sm_dim).log()
    divergence = F.kl_div(input_dist, target_dist, reduction="batchmean", log_target=True)
    return divergence


class ClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bidirectional: bool = False,
        batch_first: bool = False,
        clip_min: float = -200.0,
        clip_max: float = 200.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.clip_min = clip_min
        self.clip_max = clip_max

        # forward cell
        self.fw = nn.GRUCell(input_size, hidden_size)

        # backward cell (if needed)
        if bidirectional:
            self.bw = nn.GRUCell(input_size, hidden_size)

    def forward(self, x, h0=None):
        """
        :param x: inputs
            (seq_len, batch, input)  if batch_first=False
            (batch, seq_len, input)  if batch_first=True
        :param h0: initial state
            (num_directions, batch, hidden_size) or None

        :returns:
            output: output features from each step
                (seq_len, batch, num_directions*hidden_size)  if batch_first=False
                (batch, seq_len, num_directions*hidden_size)  if batch_first=True
            h_n: output features from final step
                (num_directions, batch, hidden_size)
        """
        # setup
        if self.batch_first:
            x = x.transpose(0, 1)  # (B, T, F) -> (T, B, F)
        seq_len, batch, _ = x.size()
        num_dir = 2 if self.bidirectional else 1

        if h0 is None:
            h0 = torch.zeros(num_dir, batch, self.hidden_size, device=x.device, dtype=x.dtype)

        # forward sweep
        h_fw = h0[0]
        fw_out = []
        for t in range(seq_len):
            h_fw = self.fw(x[t], h_fw)
            h_fw = torch.clamp(h_fw, self.clip_min, self.clip_max)
            fw_out.append(h_fw.unsqueeze(0))

        fw_out = torch.cat(fw_out, dim=0)

        # backward sweep
        if self.bidirectional:
            h_bw = h0[1]
            bw_out = []

            for t in reversed(range(seq_len)):
                h_bw = self.bw(x[t], h_bw)
                h_bw = torch.clamp(h_bw, self.clip_min, self.clip_max)
                bw_out.append(h_bw.unsqueeze(0))

            bw_out.reverse()
            bw_out = torch.cat(bw_out, dim=0)

            outputs = torch.cat([fw_out, bw_out], dim=-1)
            h_n = torch.stack([h_fw, h_bw], dim=0)
        else:
            outputs = fw_out
            h_n = h_fw.unsqueeze(0)

        # transpose if needed and return
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, h_n


class LFADS(nn.Module):
    def __init__(
        self,
        neurons: int = 128,
        a_dim: int = 16,  # dimension of a, a vector with observed kinematics, etc.
        H: int = 32,  # hidden dimension
        kappa: float = 0.1,  # std of sampling g0, from paper
        factors: int = 64,  # number of factors from the generator
        tau: float = 1e-3,  # AR(1) process autocorrelation near zero, from paper
        process_var: float = 0.1,  # AR(1) process variance near 0.1, from paper
        p_drop: float = 0.1,  # dropout probability
    ) -> None:
        super().__init__()
        # hyperparameters
        self.neurons = neurons
        self.a_dim = a_dim
        self.H = H
        self.kappa = kappa
        self.factors = factors

        # learnable parameters
        self.tau = nn.Parameter(torch.tensor(tau))
        self.process_var = nn.Parameter(torch.tensor(process_var))
        self.c0 = nn.Parameter(torch.zeros(H).unsqueeze(0).unsqueeze(0))  # (1, 1, H)
        self.f0 = nn.Parameter(torch.zeros(factors).unsqueeze(0).unsqueeze(0))  # (1, 1, F)

        # dropouts
        self.drop_in = nn.Dropout(p_drop)  # applied on inputs
        self.drop_gen = nn.Dropout(p_drop)  # applied before affine for g0
        self.drop_con = nn.Dropout(p_drop)  # applied on control
        self.drop_fac = nn.Dropout(p_drop)  # applied before affine for factors

        # encoder
        self.encoder = ClippedGRU(
            input_size=neurons + a_dim,
            hidden_size=H,
            bidirectional=True,
            batch_first=False,
        )
        self.W_mu_g0 = nn.Linear(2 * H, H)
        self.W_sig_g0 = nn.Linear(2 * H, H)

        # generator
        self.generator = ClippedGRU(
            input_size=H, hidden_size=H, bidirectional=False, batch_first=False
        )
        self.W_fac = nn.Linear(H, factors)
        self.W_rate = nn.Linear(factors, neurons)

        # controller
        self.controller_bf = ClippedGRU(  # bidirectional
            input_size=neurons + a_dim,
            hidden_size=H,
            bidirectional=True,
            batch_first=False,
        )
        self.controller_f = ClippedGRU(  # unidirectional
            input_size=(2 * H + factors),
            hidden_size=H,
            bidirectional=False,
            batch_first=False,
        )
        self.W_mu_c = nn.Linear(H, H)
        self.W_sig_c = nn.Linear(H, H)

        # initialize based on a normal distribution
        for p in self.parameters():
            if len(p.shape) == 0:
                continue
            k = p.shape[-1]
            nn.init.normal_(p, mean=0, std=1 / k)

        # count
        self.parameter_count = sum(p.numel() for p in self.parameters())

        return None

    def infer_u(self, B: int, D: int, T: int) -> Tensor:
        alpha = torch.exp(-1 / self.tau)
        eps_var = self.process_var * (1 - (alpha**2))
        eps_std = torch.sqrt(eps_var)

        u = [  # u0 uses eps-process vs. eps-noise
            torch.normal(0, 1, size=(B, D)).to(self.tau.device) * self.process_var
        ]

        for _ in range(T - 1):
            s_t_minus1 = u[-1]
            eps = torch.normal(0, 1, size=(B, D)).to(self.tau.device) * eps_std
            u.append(alpha * s_t_minus1 + eps)

        return torch.stack(u, dim=-1)  # shape (B, D, T)

    def __inner_forward(
        self,
        t_minus1: int,  # prior time step
        factors: Tensor | nn.Parameter,  # (1, B, F) # B=1 for first step
        controls: Tensor | nn.Parameter,  # (1, B, H) # B=1 for first step
        gt: Tensor,  # (1, B, 2*H) # drawn from Q(z)
        E_con: Tensor,  # (1, T, B, 2*(N+O)) # output of bidirectional controller RNN
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        single-step inputs represent the t-1 step
        """
        # prep
        _, B, _ = gt.shape
        if factors.shape[1] < B:
            factors = factors.expand(size=(factors.shape[0], B, factors.shape[2]))
        if controls.shape[1] < B:
            controls = controls.expand(size=(controls.shape[0], B, controls.shape[2]))

        # sample the control
        controls = self.drop_con(controls)
        _, controls = self.controller_f(
            torch.cat((E_con[:, t_minus1, ...], factors), 2),  # (1, B, 2*H+F)
            controls,  # (1, B, H)
        )
        mu_c = self.W_mu_c(controls)  # (1, B, H)
        sig_c = torch.exp(self.W_sig_c(controls) / 2)
        ut = torch.normal(0, 1, size=controls.shape).to(self.tau.device) * sig_c + mu_c

        # generate
        _, gt = self.generator(ut, gt)  # (1, B, H)
        gt = self.drop_fac(gt)
        factors = self.W_fac(gt)  # (1, B, F)
        rates = torch.exp(self.W_rate(factors))  # (1, B, N)

        return rates, factors, gt, controls, ut

    def forward(
        self,
        x: Optional[Tensor] = None,  # spiking data
        a: Optional[Tensor] = None,  # observations (e.g., kinematics)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        encoder-decoder forward

        :param x: neural data, defaults to zeros for generation
        :param a: observations, same shape as x, defaults to zeros
        :return: rates, factors, states, inferred_inputs, loss
        """
        # setup # shape is (batch, neurons, time-steps) # generate 10 time steps if generating
        B, N, T = x.shape if x is not None else (1, self.neurons, 10)
        Ba, O, Ta = a.shape if a is not None else (B, self.a_dim, T)  # noqa: E741
        assert B == Ba, f"x and a have different batch dimensions ({B} vs. {Ba})"
        assert T == Ta, f"x and a have different time-step dimensions ({T} vs. {Ta})"
        assert N == self.neurons, (
            f"Model not designed to handle this number of neurons ({N} vs. {self.neurons})"
        )
        assert O == self.a_dim, (
            f"Model not designed to handle this observation dimension ({O} vs. {self.a_dim})"
        )

        # initialize
        if x is None and a is None:  # check if we're using the encoder or sampling
            g0 = (
                torch.normal(0, 1, size=(1, B, 2 * self.H)).to(self.tau.device) * self.kappa
            )  # (1, B, H)
            xa = torch.zeros(size=(T, B, (N + O))).to(self.tau.device)  # (T, B, N+O)
            xa = xa + 1e-8  # add a small value for numerical stability
        else:
            x = (
                torch.zeros(size=(B, N, T)).to(self.tau.device)
                if x is None
                else x.to(self.tau.device)
            )
            a = (
                torch.zeros(size=(B, O, T)).to(self.tau.device)
                if a is None
                else a.to(self.tau.device)
            )
            xa = torch.cat((x, a), dim=1).permute(2, 0, 1)  # (T, B, N+O)
            xa = xa + 1e-8  # add a small value for numerical stability
            xa = self.drop_in(xa)

            _, E_gen = self.encoder(xa)  # fwd/back (2), B, hidden_size (H)
            E_gen = torch.cat((E_gen[0], E_gen[1]), dim=-1)  # B, 2*H
            E_gen = self.drop_gen(E_gen)

            mu_g0 = self.W_mu_g0(E_gen)  # B, 2*H
            sig_g0 = torch.exp(self.W_sig_g0(E_gen) / 2)
            g0 = torch.normal(0, 1, size=mu_g0.shape).to(self.tau.device) * sig_g0 + mu_g0  # B, 2*H
            g0 = g0.unsqueeze(0)  # (1, B, 2*H)

        E_con, _ = self.controller_bf(xa)  # T, B, 2*(N+O)
        E_con = E_con.unsqueeze(0)  # 1, T, B, 2*(N+O)
        factors: Tensor | nn.Parameter = self.f0
        controls: Tensor | nn.Parameter = self.c0
        gt = g0

        # step through our dynamical system
        states_list: list[Tensor] = []
        factors_list: list[Tensor] = []
        rates_list: list[Tensor] = []
        infer_list: list[Tensor] = []

        for t_minus1 in range(T):  # from 0 to T-1, when __inner_forward gives from 1 to T
            rt, factors, gt, controls, ut = self.__inner_forward(
                t_minus1, factors, controls, gt, E_con
            )  # type: ignore
            states_list.append(gt)  # (1, B, 2*H)
            factors_list.append(factors)  # (1, B, F)
            rates_list.append(rt)  # (1, B, N)
            infer_list.append(ut)  # (1, B, H)
        states: Tensor = torch.cat(states_list, dim=0).permute(1, 2, 0)  # (B, 2*H, T)
        factors = torch.cat(factors_list, dim=0).permute(1, 2, 0)  # (B, F, T)
        rates: Tensor = torch.cat(rates_list, dim=0).permute(1, 2, 0)  # (B, N, T)
        inferred_inputs: Tensor = torch.cat(infer_list, dim=0).permute(1, 2, 0)  # (B, H, T)

        # loss
        loss = None
        if x is not None:
            # reconstruction loss
            dist = Poisson(rates)
            L_x = -(dist.log_prob(x)).sum()  # reconstruction loss

            # g0 kl-div loss
            p_g0 = torch.normal(0, 1, size=g0.shape).to(self.tau.device) * self.kappa  # (1, B, 2*H)
            L_kl_g0 = stable_kl(g0[0], p_g0[0])

            # inferred inputs loos
            inferred_inputs_0 = inferred_inputs[..., 0]  # (B, H)
            inferred_inputs_t = inferred_inputs[..., 1:]  # (B, H, T-1)
            p_u = self.infer_u(*inferred_inputs.shape)  # (B, H, T)
            p_u_0 = p_u[..., 0]  # (B, H)
            p_u_t = p_u[..., 1:]  # (B, H, T-1)
            L_kl_u0 = stable_kl(inferred_inputs_0, p_u_0)
            L_kl_ut = stable_kl(inferred_inputs_t, p_u_t)

            # total loss
            loss = (L_x + L_kl_g0 + L_kl_u0 + L_kl_ut) / B  # average by batch

        return rates, factors, states, inferred_inputs, loss
