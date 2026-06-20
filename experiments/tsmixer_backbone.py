"""
backbone.py
-----------
PatchTSMixer backbone with custom pretraining heads for ICME detection.

The HuggingFace default forecasting head is DISABLED by using bare
PatchTSMixerModel (not ForPrediction / ForPretraining) as the encoder.

Two custom pretraining heads are attached:
  ┌──────────────────────────────────────────────────────────────────┐
  │  PatchTSMixerModel (backbone)                                    │
  │       ↓   last_hidden_state  (B, C, num_patches, d_model)        │
  │  ┌────┴──────────────────────────────┐                           │
  │  │ PretrainForecastHead              │   MSE vs future_values    │
  │  │   always active during pretrain   │                           │
  │  └───────────────────────────────────┘                           │
  │  ┌────┴──────────────────────────────┐                           │
  │  │ PretrainAnomalyHead (optional)    │   BCE per patch           │
  │  │   binary ICME / non-ICME label    │   (use_anomaly_head=True) │
  │  └───────────────────────────────────┘                           │
  └──────────────────────────────────────────────────────────────────┘

After pretraining call get_latent_representations() for XGBoost / RF
feature extraction.

Patch labelling convention (from experiment spec)
--------------------------------------------------
A patch is labelled ICME=1 if the fraction of its timesteps that fall
within a C&R ICME interval is >= overlap_threshold (default 0.10, i.e.
10%).  This is handled in the dataset, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PatchTSMixerConfig, PatchTSMixerModel


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def num_patches_from_config(config: PatchTSMixerConfig) -> int:
    """
    Mirrors PatchTSMixer's internal formula so our heads match exactly.

    num_patches = (max(context_length, patch_length) - patch_length)
                  // patch_stride + 1
    """
    return (
        max(config.context_length, config.patch_length) - config.patch_length
    ) // config.patch_stride + 1


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining forecast head
# ─────────────────────────────────────────────────────────────────────────────

class PretrainForecastHead(nn.Module):
    """
    Projects patch embeddings to a multi-step forecast (MSE pretraining).

    Uses a per-channel independent linear projection, matching the approach
    in the original PatchTSMixer paper's linear head.

    Input
    -----
    hidden_state : (B, C, num_patches, d_model)

    Output
    ------
    forecast : (B, prediction_length, C)
    """

    def __init__(self, config: PatchTSMixerConfig, head_dropout: float = 0.1):
        super().__init__()
        P = num_patches_from_config(config)
        D = config.d_model
        T = config.prediction_length

        # Pool across patches via learnable projection, channel-independently
        # (B, C, P*D) → (B, C, T) → transpose → (B, T, C)
        self.dropout = nn.Dropout(head_dropout)
        self.projection = nn.Linear(P * D, T)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        B, C, P, D = hidden_state.shape
        # (B, C, P, D) → (B, C, P*D)
        x = hidden_state.reshape(B, C, P * D)
        x = self.dropout(x)
        # (B, C, T) → (B, T, C)
        return self.projection(x).permute(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pretraining anomaly head
# ─────────────────────────────────────────────────────────────────────────────

class PretrainAnomalyHead(nn.Module):
    """
    Binary ICME / non-ICME classifier, applied independently per patch.

    Aggregates across channels (mean) then uses a small MLP to produce
    a scalar logit per patch.

    Input
    -----
    hidden_state : (B, C, num_patches, d_model)

    Output
    ------
    logits : (B, num_patches)   — raw pre-sigmoid scores
    """

    def __init__(self, config: PatchTSMixerConfig, head_dropout: float = 0.1):
        super().__init__()
        D = config.d_model
        hidden = max(D // 2, 8)
        self.dropout = nn.Dropout(head_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(D, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # (B, C, P, D) — mean over C (dim=1) → (B, P, D)
        x = hidden_state.mean(dim=1)
        x = self.dropout(x)
        # (B, P, 1) → (B, P)
        return self.mlp(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ICMEBackboneOutput:
    """Typed container for PatchTSMixerICMEBackbone.forward() results."""

    last_hidden_state: torch.Tensor
    """(B, C, num_patches, d_model) — raw patch embeddings from the backbone."""

    total_loss: Optional[torch.Tensor] = None
    """Weighted sum of forecast_loss and anomaly_loss (if both present)."""

    forecast_loss: Optional[torch.Tensor] = None
    """MSE between predicted and actual future values."""

    anomaly_loss: Optional[torch.Tensor] = None
    """BCE (with logits) across all patches."""

    forecast_pred: Optional[torch.Tensor] = None
    """(B, prediction_length, C) — forecast head output."""

    anomaly_logits: Optional[torch.Tensor] = None
    """(B, num_patches) — anomaly head raw logits."""


# ─────────────────────────────────────────────────────────────────────────────
# Main backbone wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PatchTSMixerICMEBackbone(nn.Module):
    """
    PatchTSMixer backbone with DISABLED default HF head and custom pretraining
    heads for heliophysics ICME detection.

    Pretraining
    -----------
    Provide ``future_values`` and optionally ``patch_labels`` in forward().
    Optimise on ``output.total_loss``.

    Downstream (XGBoost / Random Forest)
    -------------------------------------
    After pretraining, call ``get_latent_representations(past_values)`` to
    get a flat feature vector per window, then feed to sklearn / XGBoost.

    Parameters
    ----------
    config : PatchTSMixerConfig
        Must set: context_length, patch_length, patch_stride,
        num_input_channels, d_model, prediction_length.
    use_anomaly_head : bool
        Attach the patch-level binary ICME head.
    head_dropout : float
        Dropout applied inside both heads.
    forecast_loss_weight : float
        Scalar multiplier for the MSE forecast loss term.
    anomaly_loss_weight : float
        Scalar multiplier for the BCE anomaly loss term.
    pos_weight : float | None
        Passed to BCEWithLogitsLoss to compensate for ICME class imbalance.
        Typical value: (number of non-ICME patches) / (number of ICME patches).
    """

    def __init__(
        self,
        config: PatchTSMixerConfig,
        *,
        use_anomaly_head: bool = True,
        head_dropout: float = 0.1,
        forecast_loss_weight: float = 1.0,
        anomaly_loss_weight: float = 1.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_anomaly_head = use_anomaly_head
        self.forecast_loss_weight = forecast_loss_weight
        self.anomaly_loss_weight = anomaly_loss_weight
        self.num_patches = num_patches_from_config(config)

        # ── Core backbone: pure MLP-Mixer encoder, no HF prediction head ──
        self.backbone = PatchTSMixerModel(config)

        # ── Pretrain forecast head (always active) ──────────────────────
        self.forecast_head = PretrainForecastHead(config, head_dropout)

        # ── Optional pretrain anomaly head ──────────────────────────────
        self.anomaly_head: Optional[PretrainAnomalyHead]
        if use_anomaly_head:
            self.anomaly_head = PretrainAnomalyHead(config, head_dropout)
        else:
            self.anomaly_head = None

        # pos_weight for BCEWithLogitsLoss (class imbalance)
        if pos_weight is not None:
            self.register_buffer(
                "_pos_weight", torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self._pos_weight: Optional[torch.Tensor] = None

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        past_values: torch.Tensor,
        future_values: Optional[torch.Tensor] = None,
        patch_labels: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> ICMEBackboneOutput:
        """
        Parameters
        ----------
        past_values   : FloatTensor (B, context_length, C)
        future_values : FloatTensor (B, prediction_length, C)   — forecast loss
        patch_labels  : FloatTensor (B, num_patches)  in {0,1} — anomaly loss
        observed_mask : FloatTensor (B, context_length, C)      — optional

        Returns
        -------
        ICMEBackboneOutput
        """
        # ── Backbone forward ─────────────────────────────────────────────
        enc = self.backbone(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )
        hidden = enc.last_hidden_state  # (B, C, P, D)

        # ── Forecast head ────────────────────────────────────────────────
        forecast_loss: Optional[torch.Tensor] = None
        forecast_pred: Optional[torch.Tensor] = None
        if future_values is not None:
            forecast_pred = self.forecast_head(hidden)           # (B, T, C)
            forecast_loss = F.mse_loss(forecast_pred, future_values.float())

        # ── Anomaly head ─────────────────────────────────────────────────
        anomaly_loss: Optional[torch.Tensor] = None
        anomaly_logits: Optional[torch.Tensor] = None
        if self.anomaly_head is not None and patch_labels is not None:
            anomaly_logits = self.anomaly_head(hidden)           # (B, P)
            anomaly_loss = F.binary_cross_entropy_with_logits(
                anomaly_logits,
                patch_labels.float(),
                pos_weight=self._pos_weight,
            )

        # ── Combined pretraining loss ────────────────────────────────────
        total_loss: Optional[torch.Tensor] = None
        if forecast_loss is not None or anomaly_loss is not None:
            total_loss = past_values.new_zeros(())  # scalar zero, same device/dtype
            if forecast_loss is not None:
                total_loss = total_loss + self.forecast_loss_weight * forecast_loss
            if anomaly_loss is not None:
                total_loss = total_loss + self.anomaly_loss_weight * anomaly_loss

        return ICMEBackboneOutput(
            last_hidden_state=hidden,
            total_loss=total_loss,
            forecast_loss=forecast_loss,
            anomaly_loss=anomaly_loss,
            forecast_pred=forecast_pred,
            anomaly_logits=anomaly_logits,
        )

    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_latent_representations(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        pool: Literal["mean", "max", "flatten"] = "mean",
    ) -> torch.Tensor:
        """
        Extract a flat feature vector per sample for XGBoost / RF.

        Parameters
        ----------
        past_values   : (B, context_length, C)
        observed_mask : (B, context_length, C) optional
        pool          : aggregation over patches
                          "mean"    → (B, C * d_model)
                          "max"     → (B, C * d_model)
                          "flatten" → (B, num_patches * C * d_model)

        Returns
        -------
        features : (B, feature_dim)
        """
        self.eval()
        enc = self.backbone(
            past_values=past_values,
            observed_mask=observed_mask,
            return_dict=True,
        )
        h = enc.last_hidden_state  # (B, C, P, D)
        B, C, P, D = h.shape

        if pool == "mean":
            return h.mean(dim=2).reshape(B, C * D)
        if pool == "max":
            return h.amax(dim=2).reshape(B, C * D)
        if pool == "flatten":
            return h.reshape(B, C * P * D)
        raise ValueError(f"pool must be 'mean', 'max', or 'flatten', got {pool!r}")

    # ─────────────────────────────────────────────────────────────────────
    def get_patch_latents(
        self,
        past_values: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Return per-patch embeddings, channel-pooled.

        Useful for patch-level XGBoost classification (each patch as a
        separate sample).

        Returns
        -------
        (B, num_patches, d_model)
        """
        self.eval()
        with torch.no_grad():
            enc = self.backbone(
                past_values=past_values,
                observed_mask=observed_mask,
                return_dict=True,
            )
        h = enc.last_hidden_state  # (B, C, P, D)
        return h.mean(dim=1)       # (B, P, D)

    # ─────────────────────────────────────────────────────────────────────
    def freeze_backbone(self) -> None:
        """Freeze backbone weights (e.g. before downstream linear probe)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone weights."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    # ─────────────────────────────────────────────────────────────────────
    def summary(self) -> None:
        """Print a concise architecture summary."""
        P = self.num_patches
        C = self.config.num_input_channels
        D = self.config.d_model
        T = self.config.prediction_length
        print(
            f"PatchTSMixerICMEBackbone\n"
            f"  context_length     : {self.config.context_length}\n"
            f"  patch_length       : {self.config.patch_length}\n"
            f"  patch_stride       : {self.config.patch_stride}\n"
            f"  num_patches        : {P}\n"
            f"  num_input_channels : {C}\n"
            f"  d_model            : {D}\n"
            f"  num_layers         : {self.config.num_layers}\n"
            f"  prediction_length  : {T}\n"
            f"  use_anomaly_head   : {self.use_anomaly_head}\n"
            f"  latent_dim (mean)  : {C * D}\n"
            f"  total_params       : "
            f"{sum(p.numel() for p in self.parameters()):,}"
        )