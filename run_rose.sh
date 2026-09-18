#!/usr/bin/env bash
# ROSE training entrypoint for Ascend NPU.
#
# Edit BASE_PATH and MODEL_PATH in the "user settings" section when moving
# this script to another server.  The dataset and reward paths below are kept
# identical to run_cure.bash; the remaining values are derived or are ROSE
# defaults chosen to mirror it.
set -xeuo pipefail

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

# ----------------------------- user settings -----------------------------
# Keep the dataset/reward assignments in the command below unchanged when
# using the xyi dataset/reward.
BASE_PATH='/home/ma-user/work'
MODEL_PATH='/home/ma-user/work/model/qwen3-0.6B/main'
# --------------------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file does not exist: $1" >&2
        exit 1
    fi
}

if [[ -d "$MODEL_PATH" ]]; then
    require_file "$MODEL_PATH/config.json"
fi
require_file "$BASE_PATH/dataset/xyi/train.json"
require_file "$BASE_PATH/dataset/xyi/val_high_pass.json"
require_file "$BASE_PATH/dataset/xyi/val_low_pass.json"
require_file "$BASE_PATH/rewards/xyi/reward_math_verifier.py"

# Use the repository interpreter when it exists; otherwise use python3.
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/.venv/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

detect_npus() {
    local visible count
    visible="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_VISIBLE_DEVICES:-}}"
    if [[ -n "$visible" ]]; then
        awk -F',' '{n=0; for (i=1; i<=NF; ++i) if ($i != "") ++n; print n}' <<< "$visible"
        return
    fi
    if command -v npu-smi >/dev/null 2>&1; then
        count=$(npu-smi info -l 2>/dev/null | awk '/Total Count/ {print $NF; exit}')
        if [[ "$count" =~ ^[0-9]+$ ]] && (( count > 0 )); then
            echo "$count"
            return
        fi
    fi
    # run_cure.bash targets the four-NPU server; retain that default when the
    # platform utility is unavailable during shell-side setup.
    echo 4
}

NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-$(detect_npus)}
if ! [[ "$NPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPUS_PER_NODE must be a positive integer, got: $NPUS_PER_NODE" >&2
    exit 1
fi
DEFAULT_ROLLOUT_TP=$(( NPUS_PER_NODE < 4 ? NPUS_PER_NODE : 4 ))
ROLLOUT_TP=${ROLLOUT_TP:-$DEFAULT_ROLLOUT_TP}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
ROLLOUT_N=${ROLLOUT_N:-8}
ROSE_EPSILON=${ROSE_EPSILON:-0.5}
ROSE_ALPHA=${ROSE_ALPHA:-1.0}
ROSE_TOP_K=${ROSE_TOP_K:-20}
ROSE_SEMANTIC_CHUNK_SIZE=${ROSE_SEMANTIC_CHUNK_SIZE:-256}
ROSE_PROJECT_NAME=${ROSE_PROJECT_NAME:-verl_rose_xyi_qwen3_06b}
ROSE_EXPERIMENT_NAME=${ROSE_EXPERIMENT_NAME:-ROSE}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-180}
TEST_FREQ=${TEST_FREQ:-8}

MODEL_TAG=$(basename "${MODEL_PATH%/}")
EMBEDDING_DIR=${EMBEDDING_DIR:-"${BASE_PATH}/rose_embeddings/${MODEL_TAG}"}
EMBEDDING_PATH=${EMBEDDING_PATH:-"${EMBEDDING_DIR}/input_embeddings.f16"}
EMBEDDING_META_PATH=${EMBEDDING_META_PATH:-"${EMBEDDING_DIR}/input_embeddings.json"}
mkdir -p "$(dirname "$EMBEDDING_PATH")"

# Prepare the normalized input-embedding mmap once.  The metadata check makes
# changing MODEL_PATH safe: a stale table is regenerated automatically.
prepare_embedding=0
if [[ ! -s "$EMBEDDING_PATH" || ! -s "$EMBEDDING_META_PATH" ]]; then
    prepare_embedding=1
elif ! "$PYTHON_BIN" - "$EMBEDDING_META_PATH" "$MODEL_PATH" <<'PY'
import json
import sys

metadata_path, model_path = sys.argv[1:]
try:
    with open(metadata_path, encoding="utf-8") as stream:
        metadata = json.load(stream)
    valid = metadata.get("normalized") is True and metadata.get("model_path") == model_path
except (OSError, ValueError, TypeError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
    prepare_embedding=1
fi

if (( prepare_embedding )); then
    "$PYTHON_BIN" -m verl.experimental.rose.prepare_embeddings \
        --model-path "$MODEL_PATH" \
        --output-path "$EMBEDDING_PATH"
fi

# The four user-specified data/reward paths are intentionally kept unchanged
# from run_cure.bash below.  ROSE changes only rollout and advantage handling.
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    trainer.device=npu \
    trainer.use_v1=true \
    algorithm.adv_estimator=rose_tree \
    data.train_files="$BASE_PATH/dataset/xyi/train.json" \
    data.val_files="$BASE_PATH/dataset/xyi/val_high_pass.json","$BASE_PATH/dataset/xyi/val_low_pass.json" \
    data.prompt_key=prompt \
    data.reward_key=data_source \
    data.return_raw_chat=true \
    reward.custom_reward_function.path="$BASE_PATH/rewards/xyi/reward_math_verifier.py" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.enable=false \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.32 \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent \
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ \
    actor_rollout_ref.rollout.rose.enable=true \
    actor_rollout_ref.rollout.rose.num_trajectories="$ROLLOUT_N" \
    actor_rollout_ref.rollout.rose.epsilon="$ROSE_EPSILON" \
    actor_rollout_ref.rollout.rose.top_k="$ROSE_TOP_K" \
    actor_rollout_ref.rollout.rose.embedding_path="$EMBEDDING_PATH" \
    actor_rollout_ref.rollout.rose.embedding_meta_path="$EMBEDDING_META_PATH" \
    actor_rollout_ref.rollout.rose.semantic_device=cpu \
    actor_rollout_ref.rollout.rose.semantic_chunk_size="$ROSE_SEMANTIC_CHUNK_SIZE" \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.enforce_eager=true \
    algorithm.norm_adv_by_std_in_grpo=false \
    algorithm.use_kl_in_reward=false \
    algorithm.advantage_extra_fields='["rose_tree_metadata"]' \
    algorithm.rose.length_calibration_alpha="$ROSE_ALPHA" \
    algorithm.filter_groups.enable=true \
    algorithm.filter_groups.metric=reward \
    algorithm.filter_groups.max_inflight_gen_batches=1 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="$ROSE_PROJECT_NAME" \
    trainer.experiment_name="$ROSE_EXPERIMENT_NAME" \
    trainer.n_gpus_per_node="$NPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.nnodes="$NNODES" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.v1.sampler.sync_refill_failed_groups=true \
    "$@"
