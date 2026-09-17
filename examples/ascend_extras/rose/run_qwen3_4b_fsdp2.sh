#!/usr/bin/env bash
# FSDP2 training with vLLM-Ascend rollout.
set -xeuo pipefail

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HYDRA_FULL_ERROR=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
TRAIN_FILE=${TRAIN_FILE:-/data/math/train.parquet}
VAL_FILE=${VAL_FILE:-/data/math/test.parquet}
EMBEDDING_PATH=${EMBEDDING_PATH:?Prepare embeddings with verl.experimental.rose.prepare_embeddings and set EMBEDDING_PATH}
EMBEDDING_META_PATH=${EMBEDDING_META_PATH:-${EMBEDDING_PATH%.*}.json}

NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_N=${ROLLOUT_N:-8}
ROSE_EPSILON=${ROSE_EPSILON:-0.5}
ROSE_ALPHA=${ROSE_ALPHA:-1.0}

python3 -m verl.trainer.main_ppo \
    trainer.device=npu \
    trainer.use_v1=true \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NPUS_PER_NODE}" \
    trainer.project_name=rose-verl-npu \
    trainer.experiment_name=qwen3-4b-rose \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.ref.use_torch_compile=false \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.enable_prefix_caching=true \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ \
    actor_rollout_ref.rollout.rose.enable=true \
    actor_rollout_ref.rollout.rose.num_trajectories="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.rose.epsilon="${ROSE_EPSILON}" \
    actor_rollout_ref.rollout.rose.top_k=20 \
    actor_rollout_ref.rollout.rose.embedding_path="${EMBEDDING_PATH}" \
    actor_rollout_ref.rollout.rose.embedding_meta_path="${EMBEDDING_META_PATH}" \
    actor_rollout_ref.rollout.rose.semantic_device=cpu \
    algorithm.adv_estimator=rose_tree \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_in_reward=false \
    algorithm.advantage_extra_fields='["rose_tree_metadata"]' \
    algorithm.rose.length_calibration_alpha="${ROSE_ALPHA}" \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=reward \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    trainer.v1.sampler.sync_refill_failed_groups=true \
    "$@"
