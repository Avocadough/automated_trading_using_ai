# src/models/custom_policy.py
"""
Sequence-Aware Custom Feature Extractor for PPO.

Architecture: 1D-CNN → LayerNorm → LSTM → LayerNorm → Linear

Design Philosophy (Lead DL Architect Perspective):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. WHY NOT PLAIN MLP?
   An MLP treats the (64, 24) observation as a flat 1536-dim vector.
   All temporal structure is destroyed. The network cannot distinguish
   "RSI was 0.8 then dropped to -0.3" from "RSI was -0.3 then rose to 0.8".
   These are OPPOSITE market regimes but MLP sees identical features.

2. WHY CNN + LSTM, NOT TRANSFORMER?
   - Window = 64 steps. Transformers need O(n²) attention = 4096 ops per
     head per layer — overkill for n=64. LSTM is O(n) and more parameter-
     efficient at this scale.
   - Financial time series has strong ORDERING bias. LSTM's recurrent
     inductive bias (process left→right) matches this perfectly.
     Transformers need positional encoding to learn ordering — an extra
     learnable component that can overfit on small datasets.
   - LSTM hidden state naturally acts as a "market regime detector" —
     it learns to maintain internal state about volatility regimes,
     trend persistence, etc.

3. WHY CNN BEFORE LSTM?
   1D-CNN extracts LOCAL patterns: 3-bar candlestick formations,
   sudden vol spikes, momentum reversals. These are then fed as
   "cleaned features" to the LSTM, which focuses on TEMPORAL dependencies.
   This is analogous to how a human trader reads candle patterns first,
   then considers the trend context.

4. REGULARIZATION FOR NOISY FINANCIAL DATA:
   - LayerNorm (not BatchNorm): stable across varying batch sizes,
     doesn't leak statistics across samples.
   - Dropout(0.1) after LSTM: prevents memorizing noise patterns.
   - Orthogonal initialization: proven to help PPO convergence
     (Andrychowicz et al. 2020, "What Matters in On-Policy RL?").
   - Weight decay via AdamW in the optimizer (set in train script).
"""
from __future__ import annotations

import math
from typing import Dict, Type

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CnnLstmFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor: Conv1D → LayerNorm → LSTM → LayerNorm → Linear

    Input:  observation of shape (batch, window_size, n_features)
    Output: flat feature vector of shape (batch, features_dim)

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │  Input: (B, 64, 24)                             │
    │    ↓                                            │
    │  Conv1D(24→64, k=3, pad=1) + GELU + LayerNorm  │
    │    ↓                                            │
    │  Conv1D(64→64, k=3, pad=1) + GELU + LayerNorm  │
    │    ↓                                            │
    │  LSTM(64→128, 2 layers, dropout=0.1)            │
    │    ↓ (last hidden state)                        │
    │  LayerNorm(128) + Dropout(0.1)                  │
    │    ↓                                            │
    │  Linear(128→features_dim) + GELU                │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        cnn_channels: int = 64,
        cnn_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        # Infer dimensions from observation space
        # observation_space.shape = (window_size, n_features)
        obs_shape = observation_space.shape
        if len(obs_shape) != 2:
            raise ValueError(
                f"CnnLstmFeaturesExtractor expects 2D observations (window, features), "
                f"got shape {obs_shape}"
            )
        self.window_size = obs_shape[0]
        self.n_input_features = obs_shape[1]

        # ========================================
        # 1D-CNN: local pattern extraction
        # ========================================
        # Conv1d expects (batch, channels, seq_len)
        # We'll permute (batch, seq_len, features) → (batch, features, seq_len)
        self.cnn = nn.Sequential(
            # Layer 1: captures 3-bar patterns
            nn.Conv1d(
                in_channels=self.n_input_features,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2,  # same padding → preserve sequence length
                bias=False,               # bias unnecessary before LayerNorm
            ),
            nn.GELU(),

            # Layer 2: deeper local patterns
            nn.Conv1d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel,
                padding=cnn_kernel // 2,
                bias=False,
            ),
            nn.GELU(),
        )

        # LayerNorm after CNN (applied per-timestep on the channel dimension)
        # We normalize over the last dimension after permuting back
        self.cnn_norm = nn.LayerNorm(cnn_channels)

        # ========================================
        # LSTM: temporal dependency modeling
        # ========================================
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            # bidirectional=False — causal only (no future peeking!)
        )

        # Post-LSTM normalization + regularization
        self.lstm_norm = nn.LayerNorm(lstm_hidden)
        self.dropout = nn.Dropout(dropout)

        # ========================================
        # Output projection
        # ========================================
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_hidden, features_dim),
            nn.GELU(),
        )

        # ========================================
        # Orthogonal initialization (critical for PPO)
        # ========================================
        self._init_weights()

    def _init_weights(self):
        """
        Orthogonal initialization for all weight matrices.
        This is the gold standard for PPO (Andrychowicz et al. 2020).

        - CNN weights: orthogonal with gain=sqrt(2) (ReLU/GELU family)
        - LSTM weights: orthogonal with gain=1.0 (tanh/sigmoid gates)
        - Linear weights: orthogonal with gain=sqrt(2)
        """
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param, gain=1.0)
                    elif "bias" in name:
                        # Set forget gate bias to 1.0 (Jozefowicz et al. 2015)
                        # This prevents the LSTM from "forgetting everything"
                        # early in training, stabilizing gradient flow
                        n = param.size(0)
                        param.data.fill_(0.0)
                        # Forget gate is the second quarter of the bias
                        param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          Input:  (batch, window_size, n_features)  e.g. (256, 64, 24)
          Output: (batch, features_dim)              e.g. (256, 128)
        """
        batch_size = observations.shape[0]

        # ---- CNN: extract local patterns ----
        # Permute (batch, seq, feat) → (batch, feat, seq) for Conv1d
        x = observations.permute(0, 2, 1)    # (B, 24, 64)
        x = self.cnn(x)                       # (B, 64, 64)

        # Permute back: (batch, channels, seq) → (batch, seq, channels)
        x = x.permute(0, 2, 1)               # (B, 64, 64)
        x = self.cnn_norm(x)                  # LayerNorm per timestep

        # ---- LSTM: capture temporal dependencies ----
        # LSTM input: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the LAST hidden state from the TOP layer as the summary
        # h_n shape: (num_layers, batch, hidden), we want [-1] = top layer
        temporal_summary = h_n[-1]            # (B, 128)

        # ---- Post-processing ----
        temporal_summary = self.lstm_norm(temporal_summary)
        temporal_summary = self.dropout(temporal_summary)

        # ---- Output projection ----
        features = self.output_proj(temporal_summary)  # (B, 128)

        return features


# ============================================================
# Helper: policy_kwargs builder for SB3 PPO
# ============================================================
def get_policy_kwargs(
    observation_space: gym.spaces.Box | None = None,
    features_dim: int = 64,
    pi_layers: list[int] | None = None,
    vf_layers: list[int] | None = None,
) -> Dict:
    """
    Build policy_kwargs dict for SB3 PPO with CnnLstmFeaturesExtractor.

    The feature extractor processes the (window, features) observation
    into a flat vector, which then feeds into:
      - Policy head (pi): [64] → action logits
      - Value head (vf):  [64] → scalar value estimate

    Usage:
        model = PPO("MlpPolicy", env, policy_kwargs=get_policy_kwargs())
    """
    if pi_layers is None:
        pi_layers = [64]
    if vf_layers is None:
        vf_layers = [64]

    return dict(
        features_extractor_class=CnnLstmFeaturesExtractor,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            cnn_channels=32,
            cnn_kernel=3,
            lstm_hidden=32,
            lstm_layers=1,
            lstm_dropout=0.0,
            dropout=0.3,
        ),
        net_arch=dict(pi=pi_layers, vf=vf_layers),
        # SB3 uses ortho init internally for pi/vf heads,
        # but our extractor already handles its own init
    )
