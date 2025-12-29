# Copyright 2025 Nur Muhammad Mahi Shafiullah,
# and The HuggingFace Inc. team. All rights reserved.
# Heavy inspiration taken from
# * DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# * DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT
# * DiT Policy by Dasari et. al. : https://github.com/sudeepdasari/dit-policy

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections import deque
import logging
from math import exp
import math
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from lerobot.constants import (
    OBS_ENV_STATE,
    OBS_STATE,
    ACTION,
    OBS_IMAGES,
)
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot_policy_ditflow.configuration_ditflow import DiTFlowConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.policies.normalize import Normalize, Unnormalize

logger = logging.getLogger(__name__)


def _get_activation_fn(activation: str):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class _TimeNetwork(nn.Module):
    def __init__(
        self, frequency_embedding_dim, hidden_dim, learnable_w=False, max_period=1000
    ):
        assert frequency_embedding_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(frequency_embedding_dim // 2)
        super().__init__()

        w = np.log(max_period) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t):
        assert len(t.shape) == 1, "assumes 1d input timestep array"
        t = t[:, None] * self.w[None]
        t = torch.cat((torch.cos(t), torch.sin(t)), dim=1)
        return self.out_net(t)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * (1 + self.scale(c)[None]) + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(
        self, d_model=256, nhead=6, dim_feedforward=2048, dropout=0.0, activation="gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # create mlp
        self.mlp = nn.Sequential(
            self.linear1,
            self.activation,
            self.dropout2,
            self.linear2,
            self.dropout3,
        )

        # create modulation layers
        self.attn_modulate = _ShiftScaleMod(d_model)
        self.attn_gate = _ZeroScaleMod(d_model)
        self.mlp_modulate = _ShiftScaleMod(d_model)
        self.mlp_gate = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond, need_weights=False):
        # process the conditioning vector first
        cond = cond + t

        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=need_weights)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        x3 = self.mlp_modulate(self.norm2(x), cond)
        x3 = self.mlp(x3)
        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, t, cond):
        # process the conditioning vector first
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerDecoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_module) for _ in range(num_layers)]
        )
        self.reset_parameters()

    def forward(self, src, t, cond, **kwargs):
        x = src
        for layer in self.layers:
            x = layer(x, t, cond, **kwargs)
        return x

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        cond_dim,
        time_dim=256,
        hidden_dim=256,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
        clip_sample=False,
        clip_sample_range=1.0,
    ):
        """DiT Noise Prediction Network.

        Args:
            ac_dim: Action dimension.
            ac_chunk: Number of action steps to predict in one forward pass.
            cond_dim: Dimension of the global conditioning vector.
            time_dim: Dimension of the time embedding.
            hidden_dim: Hidden dimension of the transformer.
            num_blocks: Number of transformer blocks.
            dropout: Dropout rate.
            dim_feedforward: Dimension of the feedforward layer in the transformer.
            nhead: Number of attention heads in the transformer.
            activation: Activation function to use in the transformer.
            clip_sample: Whether to clip the output samples.
            clip_sample_range: Range to clip the output samples if `clip_sample` is True.
        """
        super().__init__()
        self.ac_dim, self.ac_chunk = ac_dim, ac_chunk

        # positional encoding blocks
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        # input encoder mlps
        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # decoder blocks
        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        # turns predicted tokens into epsilons
        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

        # clip the output samples
        self.clip_sample = clip_sample
        self.clip_sample_range = clip_sample_range

        logger.info(
            "Number of flow params: {:.2f}M".format(
                sum(p.numel() for p in self.parameters()) / 1e6
            )
        )

    def forward(self, noisy_actions, time, global_cond, need_weights=False):
        c = self.cond_proj(global_cond)
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noisy_actions)  # [B, T, adim] -> [B, T, hidden_dim]
        ac_tokens = ac_tokens.transpose(
            0, 1
        )  # [B, T, hidden_dim] -> [T, B, hidden_dim]

        # Allow variable length action chunks
        dec_in = ac_tokens + self.dec_pos[: ac_tokens.size(0)]  # [T, B, hidden_dim]

        # apply decoder
        dec_out = self.decoder(dec_in, time_enc, c, need_weights=need_weights)

        # apply final epsilon prediction layer
        eps_out = self.eps_out(
            dec_out, time_enc, c
        )  # [T, B, hidden_dim] -> [T, B, adim]
        return eps_out.transpose(0, 1)  # [T, B, adim] -> [B, T, adim]

    def sample(
        self,
        condition: torch.Tensor,
        timesteps: int = 5,
        generator: torch.Generator | None = None,
        reference_trajectory: torch.Tensor | None = None,
        guidance_scale: float = 5.0,
    ) -> torch.Tensor:
        """Sample actions using Euler integration to solve the ODE.

        Args:
            condition: Global conditioning tensor (B, cond_dim).
            timesteps: Number of integration steps.
            generator: Random number generator for reproducibility.
            reference_trajectory: a trajectory used for guidance (B, ac_chunk, ac_dim).
                Could be either a previously generated trajectory or a desired trajectory to follow.
            guidance_scale: Maximum guidance scale for prior trajectory guidance.
            n_action_steps: Number of action steps executed since the prior trajectory was generated.
                Used to shift the prior trajectory appropriately.

        Returns:
            Sampled action trajectory (B, ac_chunk, ac_dim).
        """
        # Use Euler integration to solve the ODE.
        batch_size, device = condition.shape[0], condition.device

        with torch.no_grad():
            x_t = self.sample_noise(batch_size, device, generator)

        if reference_trajectory is not None:
            assert reference_trajectory.shape == x_t.shape, (
                f"reference_trajectory shape {reference_trajectory.shape} does not match expected shape {x_t.shape}"
            )

        dt = 1.0 / timesteps
        with torch.no_grad():
            t_all = (
                torch.arange(timesteps, device=device)
                .float()
                .unsqueeze(0)
                .expand(batch_size, timesteps)
                / timesteps
            )

        for k in range(timesteps):
            t = t_all[:, k]

            if reference_trajectory is not None:
                v = self._apply_action_guidance(
                    original_velocity=lambda x: self.forward(x, t, condition),
                    x_t=x_t,
                    reference_trajectory=reference_trajectory,
                    t=t,
                    guidance_scale=guidance_scale,
                )
            else:
                with torch.no_grad():
                    v = self.forward(x_t, t, condition)
            x_t = x_t + dt * v

            if self.clip_sample:
                x_t = torch.clamp(x_t, -self.clip_sample_range, self.clip_sample_range)

            x_t = x_t.detach()

        return x_t

    def _apply_action_guidance(
        self,
        original_velocity: Callable[[torch.Tensor], torch.Tensor],
        x_t: torch.Tensor,
        reference_trajectory: torch.Tensor,
        t: torch.Tensor,
        guidance_scale: float,
        guidance_method: str = "rtc-guidance",
    ):
        """Apply temporal consistency guidance using the prior trajectory.

        This method modifies the velocity field to guide the current sample toward matching
        the previously predicted trajectory, shifted by n_action_steps to account for
        actions that have already been executed.

        Args:
            original_velocity: Function to compute velocity from state.
            x_t: Current state in the flow trajectory (B, ac_chunk, ac_dim).
            reference_trajectory: Reference trajectory (either previously generated or a trajectory to follow) (B, ac_chunk, ac_dim).
            t: Current timestep in [0, 1] (B,).
            guidance_scale: Maximum guidance scale.

        Returns:
            Modified velocity with guidance applied (B, ac_chunk, ac_dim).
        """
        device = x_t.device

        x_t = x_t.clone().detach()

        guidance_weights = self._get_guidance_weights(
            batch_size=x_t.shape[0],
            seq_len=x_t.shape[1],
            device=device,
            weights_type="exponential",
        )
        # logger.info(f"guidance_weights: {guidance_weights}")
        assert reference_trajectory.shape == x_t.shape, (
            f"Reference trajectory shape {reference_trajectory.shape} does not match x_t shape {x_t.shape}"
        )
        args = (
            original_velocity,
            x_t,
            reference_trajectory,
            t,
            guidance_weights,
            guidance_scale,
            device,
        )
        if guidance_method == "self-guided":
            v = self.__get_self_guided_guidance(*args)
        elif guidance_method == "my-guidance":
            v = self.__get_my_guidance(*args)
        else:
            v = self.__get_rtc_guidance(*args)

        return v

    def __get_self_guided_guidance(
        self,
        original_velocity,
        x_t,
        reference_trajectory,
        t,
        guidance_weights,
        guidance_scale,
        device,
    ):
        v = original_velocity(x_t)
        with torch.enable_grad():
            x_t.requires_grad_(True)
            error = (
                (((reference_trajectory - x_t) ** 2) * guidance_weights)
                .sum(dim=1)
                .mean()
            )
            # logger.info(f"Guidance error: {error}")
            correction = torch.autograd.grad(error, x_t, retain_graph=False)[0]
            # logger.info(f"Guidance correction: {correction}")
        return v - guidance_scale * correction

    def __get_my_guidance(
        self,
        original_velocity,
        x_t,
        reference_trajectory,
        t,
        guidance_weights,
        guidance_scale,
        device,
    ):
        with torch.enable_grad():
            x_t.requires_grad_(True)
            v = original_velocity(x_t)
            # We compute an estimate of the last denoised sample x_1 from x_t and v.
            x_1 = x_t + (1 - t[0]) * v
            error = (reference_trajectory - x_1) ** 2 * guidance_weights

            grad_outputs = error.clone().detach()

            correction = torch.autograd.grad(
                x_1, x_t, grad_outputs=grad_outputs, retain_graph=False
            )[0]

        final_guidance_scale = guidance_scale
        v.clone().detach()

        return v - final_guidance_scale * correction

    def __get_rtc_guidance(
        self,
        original_velocity,
        x_t,
        reference_trajectory,
        t,
        guidance_weights,
        guidance_scale,
        device,
    ):
        with torch.enable_grad():
            x_t.requires_grad_(True)
            v = original_velocity(x_t)
            # We compute an estimate of the last denoised sample x_1 from x_t and v.
            x_1 = x_t + (1 - t[0]) * v
            error = (reference_trajectory - x_1) * guidance_weights
            # logger.info(f"Guidance error: {error}")

            grad_outputs = error.clone().detach()
            correction = torch.autograd.grad(
                x_1, x_t, grad_outputs=grad_outputs, retain_graph=False
            )[0]
            # logger.info(f"Guidance correction: {correction}")

        sigma_d_squared = torch.as_tensor(1.0, device=device)
        max_guidance_scale = torch.as_tensor(guidance_scale, device=device)
        squared_one_minus_t = (1.0 - t) ** 2
        inv_r_squared = (squared_one_minus_t + t**2 * sigma_d_squared) / (
            squared_one_minus_t * sigma_d_squared
        )
        time_dependent_guidance_scale = torch.nan_to_num(
            (1 - t) / t,
            posinf=guidance_scale,
        )
        time_dependent_guidance_scale = torch.nan_to_num(
            time_dependent_guidance_scale * inv_r_squared,
            posinf=guidance_scale,
        )
        final_guidance_scale = torch.minimum(
            time_dependent_guidance_scale, max_guidance_scale
        ).view(-1, 1, 1)

        # final_guidance_scale = guidance_scale
        return v + final_guidance_scale * correction

    def sample_noise(
        self, batch_size: int, device, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return torch.randn(
            batch_size, self.ac_chunk, self.ac_dim, device=device, generator=generator
        )

    def _get_guidance_weights(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        make_ones_len: int = 4,
        make_zero_len: int = 2,
        weights_type: str = "exponential",
    ) -> torch.Tensor:
        if weights_type == "ones":
            return torch.ones(
                (batch_size, seq_len, 1), dtype=torch.float32, device=device
            )
        if weights_type == "uniform":
            return torch.ones(
                (batch_size, seq_len, 1), dtype=torch.float32, device=device
            )
        if weights_type == "exponential":
            # The weights should be:
            # [1, 1, ..., 1, exp(-1/4), exp(-2/4), exp(-3/4), ..., exp(-(N- make_zero_len - make_ones_len)/4)]
            weights = (
                torch.exp(
                    -(
                        torch.arange(seq_len, dtype=torch.float32, device=device)
                        + 1
                        - make_ones_len
                    )
                    / 4.0
                )
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            weights[:, :make_ones_len, :] = 0.75
            weights[:, -make_zero_len:, :] = 0.0
            return weights

        if weights_type == "zeros":
            weights = torch.zeros(
                (batch_size, seq_len, 1), dtype=torch.float32, device=device
            )
            return weights

        raise ValueError(f"Unknown guidance weights type: {weights_type}")


class DiTFlowPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiTFlowConfig
    name = "DiTFlow"

    def __init__(
        self,
        config: DiTFlowConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.dit_flow = DiTFlowModel(config)
        self.previous_action: torch.Tensor | None = None

        self.reset()

    def get_optim_params(self) -> dict:
        return self.dit_flow.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {}
        for input_feature in self.config.input_features.keys():
            self._queues[input_feature] = deque(maxlen=self.config.n_obs_steps)
        for output_feature in self.config.output_features.keys():
            self._queues[output_feature] = deque(maxlen=self.config.n_action_steps)
        self._queues["observation.images"] = deque(
            maxlen=self.config.n_obs_steps
        )  # for stacking image observations

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a chunk of actions given environment observations."""
        batch = self.stack_queues_of_observations_to_batch(batch)
        actions = self.dit_flow.generate_actions(batch)
        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    @torch.no_grad()
    def stack_queues_of_observations_to_batch(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Stack the n latest observations from the queues into the batch dictionary.

        This method assumes that the queues have already been populated with enough observations.
        """
        return {
            k: torch.stack(list(self._queues[k]), dim=1)
            for k in batch
            if k in self._queues
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying flow model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The flow model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)

        # NOTE: for offline evaluation, we have action in the batch, so we need to pop it out
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        action = (
            self._create_consistent_flow_action(batch)
            if self.config.do_consistent_flow
            else self._create_flow_action(batch)
        )
        return action

    def _create_flow_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def _create_consistent_flow_action(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        batch_size = batch[OBS_STATE].shape[0]

        batch = self._expand_batch(batch)

        self._queues = populate_queues(self._queues, batch)

        actions = self.predict_action_chunk(batch)

        actions = einops.rearrange(
            actions,
            "(a b) h d -> b a h d",
            a=self.config.action_batch_size,
            b=batch_size,
        )

        if self.previous_action is None:
            # Store only the first action sequence for future reference
            self.previous_action = actions[:, 0, ...]
            # action = self.unnormalize_outputs({ACTION: self.previous_action[:, 0, :]})[
            #     ACTION
            # ]
            return self.previous_action[:, 0, :]

        action = self._select_action_based_on_previous(actions, batch_size)
        self.previous_action = action

        # action = self.unnormalize_outputs({ACTION: action[:, 0, :]})[ACTION]
        return action[:, 0, :]

    def _expand_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expand batch dimension for action_batch_size.

        Repeats each observation action_batch_size times to generate multiple candidates.

        Args:
            batch: Original batch dictionary

        Returns:
            Expanded batch dictionary
        """
        expanded_batch = {}

        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue

            # Shape: (B, ...) -> (action_batch_size, B, ...) -> (action_batch_size * B, ...)
            expanded_batch[key] = (
                value.unsqueeze(0)
                .expand(
                    self.config.action_batch_size,
                    -1,
                    *(-1 for _ in value.shape[1:]),
                )
                .reshape(-1, *value.shape[1:])
            )

        return expanded_batch

    def _select_action_based_on_previous(
        self,
        action_candidates_full: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """TODO: Docstring."""
        if self.previous_action is None:
            raise ValueError("previous_action is None, cannot select based on it.")

        # TODO: Verify dimensions
        previous_action = self.previous_action[
            :,
            1:,
            :,
        ]

        action_candidates = action_candidates_full[
            :,
            :,
            :-1,
            :,
        ]

        # Compute L2 distances: (B, action_batch_size, remaining_horizon, action_dim)
        # Sum over action dimension, then over time dimension
        distances = torch.norm(
            action_candidates - previous_action.unsqueeze(1), dim=-1
        )  # (B, action_batch_size, remaining_horizon)
        distances = torch.sum(distances, dim=-1)  # (B, action_batch_size)

        # Select indices based on sampling strategy
        if self.config.action_batch_size == 1:
            indices = torch.zeros(
                batch_size, dtype=torch.long, device=action_candidates.device
            )
        elif self.config.sampling_strategy == "deterministic":
            indices = torch.argmin(distances, dim=-1)  # (B,)
        elif self.config.sampling_strategy == "stochastic":
            # Convert distances to similarity scores
            mean_distance = torch.mean(distances, dim=-1, keepdim=True)
            std_distance = torch.std(distances, dim=-1, keepdim=True) + 1e-8

            similarity = torch.exp(
                -(distances - mean_distance)
                / (std_distance * self.config.sampling_temperature)
            )
            probabilities = similarity / similarity.sum(dim=-1, keepdim=True)

            indices = torch.multinomial(probabilities, num_samples=1).squeeze(
                -1
            )  # (B,)
        else:
            raise ValueError(
                f"Unknown batch action sampling strategy: {self.config.sampling_strategy}"
            )

        selected_actions = action_candidates_full[
            torch.arange(batch_size, device=action_candidates.device), indices
        ]

        return selected_actions

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)

        loss = self.dit_flow.compute_loss(batch)
        return loss, None


class DiTFlowModel(nn.Module):
    def __init__(self, config: DiTFlowConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = 0

        if self.config.use_proprioceptive:
            if self.config.use_mlp_for_state_encoding:
                dims = [
                    self.config.robot_state_feature.shape[0]
                ] + self.config.mlp_state_encoding_dims
                layers = []
                for i in range(len(dims) - 1):
                    layers.append(nn.Linear(dims[i], dims[i + 1]))
                    if i < len(dims) - 2:
                        layers.append(_get_activation_fn("gelu"))
                self.mlp_for_state_encoding = nn.Sequential(*layers)
                global_cond_dim += self.config.mlp_state_encoding_dims[-1]
            else:
                global_cond_dim += self.config.robot_state_feature.shape[0]

        self.rgb_encoder: DiffusionRgbEncoder | nn.ModuleList
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(self.config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
            else:
                self.rgb_encoder = DiffusionRgbEncoder(self.config)
            global_cond_dim += self.get_image_conditioning_dim()

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.global_cond_dim = global_cond_dim
        self.velocity_net = _DiTNoiseNet(
            ac_dim=config.action_feature.shape[0],
            ac_chunk=config.horizon,
            cond_dim=self.global_cond_dim * config.n_obs_steps,
            time_dim=config.frequency_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.num_heads,
            activation=config.activation,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
        )

        self.training_noise_sampling = config.training_noise_sampling
        if config.training_noise_sampling == "uniform":
            self.noise_distribution = torch.distributions.Uniform(
                low=0,
                high=1,
            )
        elif config.training_noise_sampling == "beta":
            # From the Pi0 paper, https://www.physicalintelligence.company/download/pi0.pdf Appendix B.
            # There, they say the PDF for the distribution they use is the following:
            # $p(t) = Beta((s-t) / s; 1.5, 1)$
            # So, we first figure out the distribution over $t'$ and then transform it to $t = s - s * t'$.
            s = 0.999  # constant from the paper
            beta_dist = torch.distributions.Beta(
                concentration1=1.5,  # alpha
                concentration0=1.0,  # beta
            )
            affine_transform = torch.distributions.transforms.AffineTransform(
                loc=s, scale=-s
            )
            self.noise_distribution = torch.distributions.TransformedDistribution(
                beta_dist, [affine_transform]
            )
        else:
            raise ValueError(f"Unknown {config.training_noise_sampling=}")

        self.reference_trajectory: torch.Tensor | None = None
        logger.info("DiTFlowModel initialized.")

    # ========= inference  ============
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        reference_trajectory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Expand global conditioning to the batch size.
        if global_cond is not None:
            global_cond = global_cond.expand(batch_size, -1).to(
                device=device, dtype=dtype
            )

        # Sample prior.
        sample = self.velocity_net.sample(
            global_cond,
            timesteps=self.config.num_inference_steps
            if self.config.num_inference_steps < 100
            else 5,
            generator=generator,
            reference_trajectory=reference_trajectory,
            guidance_scale=self.config.guidance_scale,
        )
        return sample

    def _prepare_global_conditioning(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        global_cond_feats = []

        if self.config.use_proprioceptive and self.config.robot_state_feature:
            global_cond_feats.append(
                self.mlp_for_state_encoding(batch[OBS_STATE])
                if self.config.use_mlp_for_state_encoding
                else batch[OBS_STATE]
            )

        img_features = self.encode_image_features(batch)
        if img_features is not None:
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def encode_image_features(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        if not self.config.image_features:
            return None

        batch_size, n_obs_steps = batch[OBS_IMAGES].shape[:2]

        if self.config.use_separate_rgb_encoder_per_camera:
            # Combine batch and sequence dims while rearranging to make the camera index dimension first.
            images_per_camera = einops.rearrange(
                batch[OBS_IMAGES], "b s n ... -> n (b s) ..."
            )
            img_features_list = torch.cat(
                [
                    encoder(images)
                    for encoder, images in zip(
                        self.rgb_encoder, images_per_camera, strict=True
                    )
                ]
            )
            # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features_list,
                "(n b s) ... -> b s (n ...)",
                b=batch_size,
                s=n_obs_steps,
            )
        else:
            # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
            images_merged = einops.rearrange(
                batch[OBS_IMAGES], "b s n ... -> (b s n) ..."
            )
            img_features = self.rgb_encoder(images_merged)
            # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
            # feature dim (effectively concatenating the camera features).
            img_features = einops.rearrange(
                img_features,
                "(b s n) ... -> b s (n ...)",
                b=batch_size,
                s=n_obs_steps,
            )
        return img_features

    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        # Use an available input feature to extract batch size and n_obs_steps.
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        use_action_guidance = True  # TODO: make this configurable

        actions = self.conditional_sample(
            batch_size,
            global_cond=global_cond,
            reference_trajectory=None
            if not use_action_guidance
            else self.reference_trajectory,
        )

        if use_action_guidance:
            self.reference_trajectory = actions.clone().detach()
            trajectory_left_over_after_execution = self.reference_trajectory[
                :, self.config.n_action_steps :, :
            ]
            self.reference_trajectory = torch.zeros_like(actions)
            self.reference_trajectory[
                :, : trajectory_left_over_after_execution.shape[1], :
            ] = trajectory_left_over_after_execution

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        # assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        noise = self.velocity_net.sample_noise(trajectory.shape[0], trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = self.noise_distribution.sample((trajectory.shape[0],)).to(
            trajectory.device
        )
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = (1 - timesteps[:, None, None]) * noise + timesteps[
            :, None, None
        ] * trajectory

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.velocity_net(
            noisy_actions=noisy_trajectory, time=timesteps, global_cond=global_cond
        )
        target = trajectory - noise
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    def get_image_conditioning_dim(self) -> int:
        if self.rgb_encoder is None:
            raise ValueError(
                "No image encoder found in the model, cannot get image conditioning dim."
            )

        num_images = len(self.config.image_features)
        if self.config.use_separate_rgb_encoder_per_camera:
            return num_images * self.rgb_encoder[0].feature_dim
        else:
            return num_images * self.rgb_encoder.feature_dim
