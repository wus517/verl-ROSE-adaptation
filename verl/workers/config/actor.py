# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig, RolloutCorrectionConfig
from verl.utils.profiler.config import ProfilerConfig
from verl.utils.qat import QATConfig

from .checkpoint import McoreCheckpointConfig
from .engine import (
    FSDPEngineConfig,
    McoreEngineConfig,
    TorchtitanEngineConfig,
    VeOmniEngineConfig,
)
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
    "VeOmniActorConfig",
    "QATConfig",
    "TorchTitanActorConfig",
]


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Registered policy loss name. Options: 'vanilla', 'dppo_tv', 'dppo_kl', 'gspo', 'sapo',
            'gpg', 'clip_cov', 'kl_cov', 'geo_mean', 'dro', 'cispo', 'entropy_control', and 'bypass_mode'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        dro_beta (Optional[float]): Quadratic log-ratio penalty for DRO. Required when loss_mode is 'dro'.
        rollout_correction (RolloutCorrectionConfig): Configuration for rollout correction.
        token_weight_min (float): STEER lower bound; default matches the old actor fallback.
        token_weight_max (float): STEER upper bound; symmetric exponential always uses 1.
        linear (bool): Use the released linear mapping instead of exponential mapping.
        entropy_control_mode (str): STEER symmetric magnitude or asymmetric signed proxy.
        steer_quadrant_logging (bool): Log STEER quadrant statistics.
        q2_high_entropy_ratio (float): Fraction of valid response tokens used to set the Q2 release cutoff.
        q2_high_entropy_boost_strength (float): Maximum extra weight for selected Q2 tokens. Zero uses unit weights.
        steer_recovery_quadrants (list[int]): Quadrants eligible for STEER recovery or entropy weighting.
        steer_recovery_top_ratio (float): Fraction of valid response tokens used to set the recovery entropy cutoff.
        steer_recovery_min_entropy (float): Exclusive lower entropy bound for recovery.
        steer_recovery_rate (float): Fraction of the STEER weight reduction to restore.
        steer_entropy_weighting (bool): Enable V8 low/high entropy weight mapping.
        steer_low_entropy_weight_min (float): Mapping floor shared by enabled low entropy groups.
        steer_high_entropy_weight_min_by_quadrant (list[float]): High entropy mapping floors in Q1–Q4 order.
        steer_recovery_rate_by_quadrant (list[float]): High entropy recovery rates in Q1–Q4 order.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    dro_beta: Optional[float] = None
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    token_weight_min: float = 0.95
    token_weight_max: float = 1.05
    linear: bool = True
    entropy_control_mode: str = "symmetric"
    steer_quadrant_logging: bool = False
    q2_high_entropy_ratio: float = 0.0
    q2_high_entropy_boost_strength: float = 0.0
    steer_recovery_quadrants: list[int] = field(default_factory=list)
    steer_recovery_top_ratio: float = 0.20
    steer_recovery_min_entropy: float = 0.5
    steer_recovery_rate: float = 0.25
    steer_entropy_weighting: bool = False
    steer_low_entropy_weight_min: float = 0.50
    steer_high_entropy_weight_min_by_quadrant: list[float] = field(default_factory=lambda: [0.70] * 4)
    steer_recovery_rate_by_quadrant: list[float] = field(default_factory=lambda: [0.35] * 4)

    def __post_init__(self):
        if not isinstance(self.steer_quadrant_logging, bool):
            raise ValueError("STEER steer_quadrant_logging must be a boolean.")
        ratio = self.q2_high_entropy_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, int | float) or not 0.0 <= ratio <= 1.0:
            raise ValueError("STEER q2_high_entropy_ratio must be a finite number in [0, 1].")
        strength = self.q2_high_entropy_boost_strength
        if (
            isinstance(strength, bool)
            or not isinstance(strength, int | float)
            or not math.isfinite(strength)
            or (strength != 0.0 and strength < 0.001)
        ):
            raise ValueError("STEER q2_high_entropy_boost_strength must be zero or a finite number >= 0.001.")
        if ratio > 0.0 and (
            self.loss_mode != "entropy_control" or self.entropy_control_mode != "symmetric" or self.linear
        ):
            raise ValueError(
                "STEER q2_high_entropy_ratio > 0 requires entropy_control with symmetric exponential weights."
            )
        quadrants = self.steer_recovery_quadrants
        if (
            not isinstance(quadrants, list)
            or any(type(quadrant) is not int or not 1 <= quadrant <= 4 for quadrant in quadrants)
            or len(set(quadrants)) != len(quadrants)
        ):
            raise ValueError("STEER steer_recovery_quadrants must be a list of distinct integers in [1, 4].")
        top_ratio = self.steer_recovery_top_ratio
        if isinstance(top_ratio, bool) or not isinstance(top_ratio, int | float) or not 0.0 < top_ratio <= 1.0:
            raise ValueError("STEER steer_recovery_top_ratio must be a finite number in (0, 1].")
        min_entropy = self.steer_recovery_min_entropy
        if (
            isinstance(min_entropy, bool)
            or not isinstance(min_entropy, int | float)
            or not 0.0 <= min_entropy < math.inf
        ):
            raise ValueError("STEER steer_recovery_min_entropy must be a finite non-negative number.")
        recovery_rate = self.steer_recovery_rate
        if (
            isinstance(recovery_rate, bool)
            or not isinstance(recovery_rate, int | float)
            or not 0.0 <= recovery_rate <= 1.0
        ):
            raise ValueError("STEER steer_recovery_rate must be a finite number in [0, 1].")
        if not isinstance(self.steer_entropy_weighting, bool):
            raise ValueError("STEER steer_entropy_weighting must be a boolean.")
        low_floor = self.steer_low_entropy_weight_min
        if isinstance(low_floor, bool) or not isinstance(low_floor, int | float) or not 0.0 < low_floor <= 1.0:
            raise ValueError("STEER steer_low_entropy_weight_min must be a finite number in (0, 1].")
        high_floors = self.steer_high_entropy_weight_min_by_quadrant
        if (
            not isinstance(high_floors, list)
            or len(high_floors) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value <= 1.0
                for value in high_floors
            )
        ):
            raise ValueError("STEER steer_high_entropy_weight_min_by_quadrant requires four finite values in (0, 1].")
        recovery_rates = self.steer_recovery_rate_by_quadrant
        if (
            not isinstance(recovery_rates, list)
            or len(recovery_rates) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0
                for value in recovery_rates
            )
        ):
            raise ValueError("STEER steer_recovery_rate_by_quadrant requires four finite values in [0, 1].")
        if quadrants:
            if self.loss_mode != "entropy_control" or self.entropy_control_mode != "symmetric" or self.linear:
                raise ValueError("STEER recovery requires entropy_control with symmetric exponential weights.")
            if ratio > 0.0:
                raise ValueError("STEER recovery and q2_high_entropy_ratio > 0 cannot be enabled together.")
        if self.loss_mode != "entropy_control":
            if self.steer_quadrant_logging:
                raise ValueError("STEER quadrant logging requires loss_mode=entropy_control.")
            return
        if not math.isfinite(self.token_weight_min) or not 0 < self.token_weight_min <= 1:
            raise ValueError("STEER token_weight_min must be in (0, 1].")
        if not math.isfinite(self.token_weight_max) or self.token_weight_max < self.token_weight_min:
            raise ValueError("STEER token_weight_max must be finite and >= token_weight_min.")
        if self.entropy_control_mode not in ("symmetric", "asymmetric"):
            raise ValueError("STEER entropy_control_mode must be symmetric or asymmetric.")
        if self.steer_quadrant_logging and (self.entropy_control_mode != "symmetric" or self.linear):
            raise ValueError("STEER quadrant logging requires symmetric exponential weighting (linear=false).")


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode, including 'token-mean', 'token-sum', and sequence modes.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    calculate_sum_pi_squared: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 42
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)
    qat: QATConfig = field(default_factory=QATConfig)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "token-sum",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
        checkpoint (McoreCheckpointConfig): Megatron-specific checkpoint config
            that adds ``mbridge_config`` on top of the base checkpoint fields.
    """

    strategy: str = "megatron"
    entropy_from_logits_with_chunking: bool = False
    entropy_from_logits_chunk_size: int = 2048
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False
    checkpoint: McoreCheckpointConfig = field(default_factory=McoreCheckpointConfig)

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        pad_to_length (bool): Whether to pad every packed micro-batch to a static token count.
            Forwarded to ``fsdp_config.pad_to_length``, which is what the engine reads.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_from_logits_chunk_size: int = 2048
    entropy_checkpointing: bool = False
    pad_to_length: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config
        # Sync strategy to engine config so engine_workers can pick the right FSDP version.
        # EngineConfig.strategy defaults to None, so without this, engine_workers.py always
        # falls back to FSDP1 even when actor.strategy="fsdp2".
        object.__setattr__(self.engine, "strategy", self.strategy)

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)
        if (
            self.ulysses_sequence_parallel_size > 1
            and model_config
            and not model_config.get("use_remove_padding", False)
        ):
            raise ValueError(
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )


@dataclass
class VeOmniActorConfig(ActorConfig):
    """Configuration for VeOmni actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'veomni' for VeOmni parallelism.
        veomni (dict[str, Any]): Configuration for VeOmni settings.
        pad_to_length (bool): Whether to pad every packed micro-batch to a static token count.
            Forwarded to ``veomni.pad_to_length``, which is what the engine reads.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "veomni"
    veomni: VeOmniEngineConfig = field(default_factory=VeOmniEngineConfig)
    pad_to_length: bool = False
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate VeOmni actor configuration parameters."""
        super().__post_init__()
        self.engine = self.veomni
        if self.veomni.router_replay.mode != "disabled" and not self.use_remove_padding:
            raise RuntimeError(
                "router_replay requires use_remove_padding=True. In VeOmni engine, "
                "the non-remove-padding path also disables Ulysses SP slicing and "
                "the fused-kernel log_probs path, and is not a tested production "
                "configuration for MoE routing replay. Set "
                "actor.use_remove_padding=True or router_replay.mode='disabled'."
            )


@dataclass
class TorchTitanActorConfig(ActorConfig):
    """Configuration for TorchTitan actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'torchtitan' for TorchTitan parallelism.
        torchtitan (TorchtitanEngineConfig): Configuration for TorchTitan engine settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
        use_rollout_log_probs (bool): Whether to use log probabilities from rollout engine
    """

    strategy: str = "torchtitan"
    torchtitan: TorchtitanEngineConfig = field(default_factory=TorchtitanEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate TorchTitan actor configuration parameters."""
        super().__post_init__()
        self.engine = self.torchtitan
