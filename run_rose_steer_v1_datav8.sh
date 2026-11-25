#!/usr/bin/env bash
# ROSE tree rollouts with the STEER actor update on datav8 / Ascend 910B.
set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "${VERL_PROJECT_ROOT:-$SCRIPT_DIR}"
PROJECT_ROOT=$PWD
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

BASE_PATH=${BASE_PATH:-${lhx_sh:-}}
SAVE_PATH=${SAVE_PATH:-${lhx_save:-}}
XYI_DIR="$BASE_PATH/xyi/v8_all"
TRAIN_FILE=${TRAIN_FILE:-$XYI_DIR/math.train.parquet}
VAL_EASY=${VAL_EASY:-$XYI_DIR/math.val.easy.parquet}
VAL_ABITHARD=${VAL_ABITHARD:-$XYI_DIR/math.val.a_bit_hard.parquet}
VAL_HARD=${VAL_HARD:-$XYI_DIR/math.val.hard.parquet}
VAL_VERYHARD=${VAL_VERYHARD:-$XYI_DIR/math.val.very_hard.parquet}
TEST_EASY=${TEST_EASY:-$XYI_DIR/math.test.easy.parquet}
TEST_ABITHARD=${TEST_ABITHARD:-$XYI_DIR/math.test.a_bit_hard.parquet}
TEST_HARD=${TEST_HARD:-$XYI_DIR/math.test.hard.parquet}
TEST_VERYHARD=${TEST_VERYHARD:-$XYI_DIR/math.test.very_hard.parquet}
REWARD_FILE=${REWARD_FILE:-$XYI_DIR/verify_choice_answer_new_v4.py}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}

MODEL_PATH=${MODEL_PATH:-$BASE_PATH/model/Qwen3_5_9B}
NNODES=${NNODES:-${MA_NUM_HOSTS:-1}}
NPUS_PER_NODE=${NPUS_PER_NODE:-${MA_NUM_GPUS:-8}}
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-$NPUS_PER_NODE}
PROJECT_NAME=${PROJECT_NAME:-verl_rose_steer_v1_datav8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_NAME:-qwen3_5_9b_rose_steer_v1}}
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$SAVE_PATH/saved_models/rl_ckpt/rose_steer_v1/$EXPERIMENT_NAME}
LOG_PATH=${LOG_PATH:-$SAVE_PATH/qwen35_rl/log/rose_steer_v1/$EXPERIMENT_NAME}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$SAVE_PATH/tensorboard/rose_steer_v1/${EXPERIMENT_NAME}_$TIMESTAMP}
ROLLOUT_HISTORY_DIR=${ROLLOUT_HISTORY_DIR:-$SAVE_PATH/rollout_history/rose_steer_v1/${EXPERIMENT_NAME}_$TIMESTAMP}
TRAIN_ROLLOUT_DIR="$ROLLOUT_HISTORY_DIR/train"
VAL_ROLLOUT_DIR="$ROLLOUT_HISTORY_DIR/val"
SAVELOG_FILE_NAME=${SAVELOG_FILE_NAME:-${EXPERIMENT_NAME}_$TIMESTAMP.log}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-48}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-5120}
ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-6}
VAL_FREQ=${VAL_FREQ:-6}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5120}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-5120}
ROSE_EPSILON=${ROSE_EPSILON:-0.5}
ROSE_TOP_K=${ROSE_TOP_K:-20}
ROSE_SEMANTIC_CHUNK_SIZE=${ROSE_SEMANTIC_CHUNK_SIZE:-256}
MODEL_TAG=$(basename "${MODEL_PATH%/}")
EMBEDDING_DIR=${EMBEDDING_DIR:-$BASE_PATH/rose_embeddings/$MODEL_TAG}
EMBEDDING_PATH=${EMBEDDING_PATH:-$EMBEDDING_DIR/input_embeddings.f16}
EMBEDDING_META_PATH=${EMBEDDING_META_PATH:-${EMBEDDING_PATH%.*}.json}

STEER_TOKEN_WEIGHT_MIN=${STEER_TOKEN_WEIGHT_MIN:-0.7}
STEER_TOKEN_WEIGHT_MAX=${STEER_TOKEN_WEIGHT_MAX:-1.0}
STEER_LINEAR=${STEER_LINEAR:-false}
STEER_ENTROPY_CONTROL_MODE=${STEER_ENTROPY_CONTROL_MODE:-symmetric}

trainer_args=(
    algorithm.adv_estimator=grpo
    algorithm.rollout_correction.rollout_is=null
    algorithm.rollout_correction.rollout_rs=null
    algorithm.rollout_correction.bypass_mode=False
    algorithm.use_kl_in_reward=False
    algorithm.filter_groups.enable=True
    algorithm.filter_groups.metric=reward
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_EASY}', '${VAL_ABITHARD}', '${VAL_HARD}', '${VAL_VERYHARD}', '${TEST_EASY}', '${TEST_ABITHARD}', '${TEST_HARD}', '${TEST_VERYHARD}']"
    data.image_key=images
    data.prompt_key=prompt
    data.reward_fn_key=data_source
    data.return_raw_chat=True
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    data.max_prompt_length="${MAX_PROMPT_LENGTH}"
    data.max_response_length="${MAX_RESPONSE_LENGTH}"
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    ++data.apply_chat_template_kwargs.enable_thinking=False
    ++actor_rollout_ref.rollout.engine_kwargs.chat_template_kwargs.enable_thinking=False
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
    actor_rollout_ref.actor.clip_ratio="${CLIP_RATIO}"
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}"
    actor_rollout_ref.actor.ppo_epochs="${PPO_EPOCHS}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.calculate_entropy=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=False
    actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control
    actor_rollout_ref.actor.policy_loss.token_weight_min="${STEER_TOKEN_WEIGHT_MIN}"
    actor_rollout_ref.actor.policy_loss.token_weight_max="${STEER_TOKEN_WEIGHT_MAX}"
    actor_rollout_ref.actor.policy_loss.linear="${STEER_LINEAR}"
    actor_rollout_ref.actor.policy_loss.entropy_control_mode="${STEER_ENTROPY_CONTROL_MODE}"
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.loss_scale_factor=null
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False
    actor_rollout_ref.actor.fsdp_config.grad_sync_every_micro_batch=True
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']"
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}"
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}"
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    reward.reward_model.enable=false
    actor_rollout_ref.actor.clip_ratio_high=0.20
    actor_rollout_ref.actor.clip_ratio_low=0.20
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.agent.num_workers=4
    actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS"
    actor_rollout_ref.rollout.calculate_log_probs=true
    actor_rollout_ref.rollout.agent.default_agent_loop=single_turn_agent
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl.experimental.rose.rose_agent_loop_tq.RoseAgentLoopManagerTQ
    actor_rollout_ref.rollout.rose.enable=true
    actor_rollout_ref.rollout.rose.num_trajectories="$ROLLOUT_N"
    actor_rollout_ref.rollout.rose.epsilon="$ROSE_EPSILON"
    actor_rollout_ref.rollout.rose.top_k="$ROSE_TOP_K"
    actor_rollout_ref.rollout.rose.embedding_path="$EMBEDDING_PATH"
    actor_rollout_ref.rollout.rose.embedding_meta_path="$EMBEDDING_META_PATH"
    actor_rollout_ref.rollout.rose.semantic_device=cpu
    actor_rollout_ref.rollout.rose.semantic_chunk_size="$ROSE_SEMANTIC_CHUNK_SIZE"
    algorithm.norm_adv_by_std_in_grpo=true
    algorithm.advantage_extra_fields='[]'
    reward.reward_manager.name=naive
    reward.num_workers="${REWARD_NUM_WORKERS}"
    reward.custom_reward_function.path="${REWARD_FILE}"
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.balance_batch=True
    trainer.logger='["console","tensorboard"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="$EXPERIMENT_NAME"
    trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}"
    trainer.validation_data_dir="${VAL_ROLLOUT_DIR}"
    trainer.n_gpus_per_node="${NTRAINER_GPUS_PER_NODE}"
    trainer.nnodes="${TRAINER_NNODES}"
    trainer.device=npu
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}"
    trainer.val_before_train=False
    trainer.v1.sampler.sync_refill_failed_groups=True
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${VAL_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
)

if [[ ${1:-} == --print-config ]]; then
    shift
    exec "$PYTHON_BIN" -m verl.trainer.main_ppo --cfg job "${trainer_args[@]}" "$@"
fi

if [[ -z "$BASE_PATH" || -z "$SAVE_PATH" ]]; then
    echo "Set BASE_PATH and SAVE_PATH (or lhx_sh and lhx_save) before launching training." >&2
    exit 1
fi

# Cluster environment from the ROSE datav8 launcher.
PROXY_ADDR=${PROXY_ADDR-http://10.31.196.119:80}
if [[ -n "$PROXY_ADDR" ]]; then
    export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
    export http_proxy="$PROXY_ADDR" https_proxy="$PROXY_ADDR"
    export HTTP_PROXY="$PROXY_ADDR" HTTPS_PROXY="$PROXY_ADDR"
    export FTP_PROXY="$PROXY_ADDR" ftp_proxy="$PROXY_ADDR"
fi

source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
source /home/ma-user/Ascend/nnal/atb/set_env.sh
export LD_PRELOAD="/home/ma-user/Ascend/cann-8.5.2/lib64/libprofapi.so${LD_PRELOAD:+:$LD_PRELOAD}"
export LD_LIBRARY_PATH="/usr/local/lib:/root/miniconda3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NON_MEGATRON=true MULTI_STREAM_MEMORY_REUSE=2 TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0 ACLNN_CACHE_LIMIT=100000 CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF
export ACL_OP_COMPILER_CACHE_DIR
export HCCL_EXEC_TIMEOUT=7200 HCCL_IF_BASE_PORT=64000 ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200 TORCH_DIST_TIMEOUT=3600 HCCL_OP_EXPANSION_MODE=AIV
export HYDRA_FULL_ERROR=1 ASCEND_GLOBAL_LOG_LEVEL=3
export RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0 RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_gcs_server_rpc_server_thread_num=32 RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000 RAY_worker_register_timeout_seconds=1200
export RAY_HEARTBEAT_TIMEOUT_MS=60000 RAY_gcs_actor_failure_grace_period_ms=60000
export VLLM_RPC_TIMEOUT=600000

NODE_RANK=${VC_TASK_INDEX:-0}
WORKER_HOSTS=${VC_WORKER_HOSTS:-${MA_CURRENT_IP:?Set MA_CURRENT_IP to the local Ray node IP}}
MASTER_ADDR=${MASTER_ADDR:-${WORKER_HOSTS%%,*}}
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
export NODE_RANK MASTER_ADDR MASTER_PORT NPUS_PER_NODE
export WORLD_SIZE=$((NNODES * NPUS_PER_NODE))
export HCCL_SOCKET_IFNAME="$SOCKET_IFNAME" GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" TP_SOCKET_IFNAME="$SOCKET_IFNAME"
export TMPDIR=${TMPDIR_RAY:-/cache/ray_tmp}
mkdir -p "$TMPDIR"

"$PYTHON_BIN" -c 'import torch_npu; print(torch_npu.npu.is_available())'
"$PYTHON_BIN" -m pip install 'accelerate>=1.6.0' aiohttp aiofiles math-verify \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

prepare_embeddings() {
    if "$PYTHON_BIN" - "$EMBEDDING_PATH" "$EMBEDDING_META_PATH" "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

embedding, metadata = map(Path, sys.argv[1:3])
try:
    info = json.loads(metadata.read_text(encoding="utf-8"))
    assert info["normalized"] is True and info["dtype"] == "float16"
    assert info["model_path"] == sys.argv[3]
    assert embedding.stat().st_size == 2 * info["vocab_size"] * info["hidden_size"]
except (OSError, ValueError, TypeError, KeyError, AssertionError):
    raise SystemExit(1) from None
PY
    then
        return
    fi

    mkdir -p "$(dirname "$EMBEDDING_PATH")" "$(dirname "$EMBEDDING_META_PATH")"
    (
        tmp_dir=$(mktemp -d "$(dirname "$EMBEDDING_PATH")/.rose-steer-embeddings.XXXXXX")
        trap 'rm -rf -- "$tmp_dir"' EXIT
        "$PYTHON_BIN" -m verl.experimental.rose.prepare_embeddings \
            --model-path "$MODEL_PATH" --output-path "$tmp_dir/embeddings.f16"
        mv -f -- "$tmp_dir/embeddings.f16" "$EMBEDDING_PATH"
        mv -f -- "$tmp_dir/embeddings.json" "$EMBEDDING_META_PATH"
    )
}

ray_node_args=("--node-ip-address=$MA_CURRENT_IP" "--resources={\"NPU\": $NPUS_PER_NODE}")
if [[ "$NODE_RANK" == 0 ]]; then
    mkdir -p "$LOG_PATH" "$TENSORBOARD_DIR" "$TRAIN_ROLLOUT_DIR" "$VAL_ROLLOUT_DIR"
    ray start --head "${ray_node_args[@]}" --port="$MASTER_PORT" \
        --dashboard-host=0.0.0.0 --dashboard-port="$DASHBOARD_PORT"

    # All Ray nodes read the shared datav8 embedding files. The head prepares them once.
    prepare_embeddings
    while true; do
        ray_status=$(ray status 2>/dev/null || true)
        npu_count=$(awk 'match($0, /\/[0-9]+(\.[0-9]+)?[[:space:]]+NPU/) { print int(substr($0, RSTART + 1)); exit }' \
            <<< "$ray_status")
        if (( ${npu_count:-0} >= NNODES * NPUS_PER_NODE )); then
            break
        fi
        echo "Waiting for Ray NPUs. Current=${npu_count:-0}, expected=$((NNODES * NPUS_PER_NODE))"
        sleep 10
    done

    "$PYTHON_BIN" -m verl.trainer.main_ppo "${trainer_args[@]}" "$@" 2>&1 | tee -a "$LOG_PATH/$SAVELOG_FILE_NAME"
else
    ray start "${ray_node_args[@]}" --address="$MASTER_ADDR:$MASTER_PORT" --block
fi
