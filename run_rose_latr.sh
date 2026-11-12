#!/bin/bash
# ROSE + sampled LATR-style lookahead; baseline run_rose_new.sh is unchanged.
# Same BASE_PATH / SAVE_PATH / MODEL_PATH and NPU scheduler environment as the baseline.
# Actual training entry: python3 -m verl.trainer.main_ppo below.
# Matched baseline control: ROSE_LOOKAHEAD_ENABLE=false bash run_rose_latr.sh
# Read ROSE_LATR_RUN_GUIDE.md before the first NPU smoke test.
set -x

# dependency: NPU vllm==0.18.0, vllm-ascend@<54879467>, transformers@<cc7ab9be>

echo "datetime: $(date +'%Y-%m-%d %H:%M:%S')"
echo $MA_CURRENT_HOST_IP
echo $MA_CURRENT_IP

################## 上海三方电信 代理配置 ##################
PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTPS_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"

echo "***************************************** copy file *****************************************"
WORKDIR=`pwd`
echo "inital dictionary: $WORKDIR"
PROJECT_ROOT=/cache/algorithm

ROOT_PATH="/opt/huawei"
cp -r ${ROOT_PATH}/schedule-train/algorithm ${PROJECT_ROOT}
ls ${PROJECT_ROOT}

echo "***************************************** update cann *****************************************"
ls /home/ma-user/
ls /home/ma-user/Ascend/

# ./Ascend-cann-toolkit_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
# ./Ascend-cann-910b-ops_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
# ./Ascend-cann-nnal_8.5.2_linux-aarch64.run --install --quiet
source /home/ma-user/Ascend/nnal/atb/set_env.sh

echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF

export ACL_OP_COMPILER_CACHE_DIR=$ACL_OP_COMPILER_CACHE_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV
export LD_PRELOAD=/home/ma-user/Ascend/cann-8.5.2/lib64/libprofapi.so${LD_PRELOAD:+:$LD_PRELOAD}

python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

# <<< 修改：使用变量 BASE_PATH 指向 xyi 数据目录
# ================= 数据集 & 奖励函数路径（xyi） =================
# ${lnn_sh} 在环境中已定义；若没有，请手动改为绝对路径
BASE_PATH=${BASE_PATH:-${lhx_sh}}
SAVE_PATH=${SAVE_PATH:-${lhx_save}}
echo "BASE_PATH resolved to: ${BASE_PATH}"

XYI_DIR="${BASE_PATH}/xyi"
TRAIN_FILE=${TRAIN_FILE:-${XYI_DIR}/train_verl.json}
VAL_FILE_HIGH=${VAL_FILE_HIGH:-${XYI_DIR}/high_pass_val_verl.json}
VAL_FILE_LOW=${VAL_FILE_LOW:-${XYI_DIR}/low_pass_val_verl.json}
REWARD_FILE=${REWARD_FILE:-${XYI_DIR}/verify_choice_answer_new_v4.py}

echo "TRAIN_FILE      = ${TRAIN_FILE}"
# echo "VAL_FILE    = ${VAL_FILE}"
echo "VAL_FILE_HIGH   = ${VAL_FILE_HIGH}"
echo "VAL_FILE_LOW    = ${VAL_FILE_LOW}"
echo "REWARD_FILE     = ${REWARD_FILE}"
pip install "accelerate>=1.6.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import accelerate; print(accelerate.__version__)"
# ==========================ROSE配置=================================

ROSE_LOOKAHEAD_ENABLE=${ROSE_LOOKAHEAD_ENABLE:-true}
ROSE_LOOKAHEAD_WINDOWS=${ROSE_LOOKAHEAD_WINDOWS:-'[20,30,50]'}
ROSE_LOOKAHEAD_THRESHOLD=${ROSE_LOOKAHEAD_THRESHOLD:-0.4}
ROSE_LOOKAHEAD_MAX_TRIALS=${ROSE_LOOKAHEAD_MAX_TRIALS:-2}
ROSE_LOOKAHEAD_MAX_PIVOTS=${ROSE_LOOKAHEAD_MAX_PIVOTS:-4}
ROSE_LOOKAHEAD_GROUP_BUDGET=${ROSE_LOOKAHEAD_GROUP_BUDGET:-700}
ROSE_LOOKAHEAD_REQUEST_TIMEOUT=${ROSE_LOOKAHEAD_REQUEST_TIMEOUT:-600}
# Optional bounded per-worker JSONL; token IDs are sensitive and off by default.
ROSE_LOOKAHEAD_TRACE_DIR=${ROSE_LOOKAHEAD_TRACE_DIR:-}
ROSE_LOOKAHEAD_TRACE_TOKEN_IDS=${ROSE_LOOKAHEAD_TRACE_TOKEN_IDS:-false}
LOOKAHEAD_TRACE_ARGS=()
if [[ -n "$ROSE_LOOKAHEAD_TRACE_DIR" ]]; then
    LOOKAHEAD_TRACE_ARGS+=("actor_rollout_ref.rollout.rose.lookahead.trace_dir='${ROSE_LOOKAHEAD_TRACE_DIR}'")
fi
# Optional smoke-test step cap. Unset preserves the baseline total_epochs behavior.
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    LOOKAHEAD_TRACE_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

ROSE_EPSILON=${ROSE_EPSILON:-0.3} #
ROSE_ALPHA=${ROSE_ALPHA:-0} #
ROSE_TOP_K=${ROSE_TOP_K:-20}
ROSE_SEMANTIC_CHUNK_SIZE=${ROSE_SEMANTIC_CHUNK_SIZE:-256}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-4800} #
# ===============================================================
MODEL_PATH=${MODEL_PATH:-${BASE_PATH}/model/Qwen3_5_9B} #
NNODES=${NNODES:-$MA_NUM_HOSTS}
EXP_NAME=${EXP_NAME:-qwen3_5_9b_rose_latr}

# ================= TensorBoard 输出目录（带时间戳） =================
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TENSORBOARD_DIR="${SAVE_PATH}/tensorboard/rose_latr/${EXP_NAME}_${TIMESTAMP}"
mkdir -p "${TENSORBOARD_DIR}"
echo "TENSORBOARD_DIR = ${TENSORBOARD_DIR}"

# ========================== 保存rollout历史 =================================
ROLLOUT_HISTORY_DIR="${SAVE_PATH}/rollout_history/rose_latr/${EXP_NAME}_${TIMESTAMP}"

TRAIN_ROLLOUT_DIR="${ROLLOUT_HISTORY_DIR}/train"
VAL_ROLLOUT_DIR="${ROLLOUT_HISTORY_DIR}/val"

mkdir -p "${TRAIN_ROLLOUT_DIR}" "${VAL_ROLLOUT_DIR}"

echo "TRAIN_ROLLOUT_DIR = ${TRAIN_ROLLOUT_DIR}"
echo "VAL_ROLLOUT_DIR   = ${VAL_ROLLOUT_DIR}"
# ========================================================================
NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}
# ========================================================================

# GRPO 不需要区分 Student/Teacher，整节点资源全给 actor/rollout/ref
NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}
# rule-only：trainer 占用全部 NNODES（不再为 GenRM 预留节点）
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}

# verl_grpo_geo3k
PROJECT_NAME=${PROJECT_NAME:-verl_rose_latr_xyi_qwen3_5_9b}

# TRAIN_BATCH_SIZE 和 PPO_MINI_BATCH_SIZE设置成卡数
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1500}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4800}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}
GRAD_CLIP=${GRAD_CLIP:-1.0}

ROLLOUT_TP=${ROLLOUT_TP:-2} #
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7} #
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${SAVE_PATH}/saved_models/rl_ckpt/rose_latr/${EXP_NAME}}

# ${cq_qwen_datasets}/saved_models/rl_ckpt
LOG_PATH=${LOG_PATH:-${SAVE_PATH}/qwen35_rl/log/rose_latr/${EXP_NAME}/}
echo "LOG_PATH resolved to: ${LOG_PATH}"

TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-4} #
VAL_FREQ=${VAL_FREQ:-6} #

# MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4800}

### 启动分布式网络配置（多节点）
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
MASTER_PORT=${MASTER_PORT:-6167}
DASHBOARD_PORT=${DASHBOARD_PORT:-8191}

export NODE_RANK MASTER_ADDR MASTER_PORT
export TMPDIR=/cache/ray_tmp
mkdir -p /cache/ray_tmp

export NPUS_PER_NODE=8
export WORLD_SIZE=$(( NNODES * NPUS_PER_NODE ))

readonly SOCKET_IFNAME="eth0"

export RAY_DEDUP_LOGS=0
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export ASCEND_GLOBAL_LOG_LEVEL=3
export HYDRA_FULL_ERROR=1
export RAY_gcs_server_rpc_server_thread_num=32
export RAY_gcs_server_request_timeout_seconds=1200
export RAY_timeout_ms=120000
export RAY_worker_register_timeout_seconds=1200
export RAY_USAGE_STATS_ENABLED=0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

export VLLM_RPC_TIMEOUT=600000

pip install aiohttp aiofiles math-verify -i https://pypi.tuna.tsinghua.edu.cn/simple
pip list

export TENSORBOARD_DIR="${TENSORBOARD_DIR}"

export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000

echo $RAY_gcs_rpc_server_reconnect_timeout_s

# ================= 准备embedding =================
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
elif ! python3 - "$EMBEDDING_META_PATH" "$MODEL_PATH" <<'PY'
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
    python3 -m verl.experimental.rose.prepare_embeddings \
        --model-path "$MODEL_PATH" \
        --output-path "$EMBEDDING_PATH"
fi

# ================= 多节点：启动 Ray 集群 =================
if [ "${NODE_RANK}" = "0" ]; then
    # ---- 头节点：启动 Ray Head ----
    echo "Starting Ray HEAD node on ${MA_CURRENT_IP}..."
    ray start --head \
        --node-ip-address=${MA_CURRENT_IP} \
        --port=${MASTER_PORT} \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=${DASHBOARD_PORT} \
        --resources='{"NPU": '${NPUS_PER_NODE}'}'

    while true; do
        ray_status_output=$(ray status)
        npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
        npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
        device_count=$((npu_count_int / NPUS_PER_NODE))

        if [ "$device_count" -eq "$NNODES" ]; then
            echo "Ray cluster ready: ${device_count} nodes"
            ray status

            mkdir -p ${LOG_PATH}

            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=rose_tree \
                algorithm.use_kl_in_reward=False \
                data.train_files="['${TRAIN_FILE}']" \
                data.val_files="['${VAL_FILE_HIGH}', '${VAL_FILE_LOW}']" \
                data.image_key=images \
                data.prompt_key=prompt \
                data.reward_fn_key=data_source \
                data.return_raw_chat=True \
                data.train_batch_size=${TRAIN_BATCH_SIZE} \
                data.max_prompt_length=${MAX_PROMPT_LENGTH} \
                data.max_response_length=${MAX_RESPONSE_LENGTH} \
                data.filter_overlong_prompts=True \
                data.truncation='error' \
                data.shuffle=True \
                ++data.apply_chat_template_kwargs.enable_thinking=False \
                ++actor_rollout_ref.rollout.engine_kwargs.chat_template_kwargs.enable_thinking=False \
                actor_rollout_ref.model.path="${MODEL_PATH}" \
                actor_rollout_ref.model.use_remove_padding=False \
                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                actor_rollout_ref.actor.strategy=fsdp2 \
                actor_rollout_ref.actor.use_torch_compile=False \
                actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
                actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO} \
                actor_rollout_ref.actor.grad_clip=${GRAD_CLIP} \
                actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS} \
                actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
                actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.actor.use_dynamic_bsz=False \
                actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.actor.use_kl_loss=True \
                actor_rollout_ref.actor.kl_loss_coef=0.01 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.calculate_entropy=False \
                actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.fsdp_config.offload_policy=True \
                actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.actor.fsdp_config.grad_sync_every_micro_batch=True \
                actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']" \
                actor_rollout_ref.ref.strategy=fsdp2 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
                actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.name=vllm \
                actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
                actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL} \
                actor_rollout_ref.rollout.n=${ROLLOUT_N} \
                actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
                actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P} \
                actor_rollout_ref.rollout.val_kwargs.n=1 \
                actor_rollout_ref.rollout.val_kwargs.do_sample=False \
                actor_rollout_ref.rollout.val_kwargs.temperature=0 \
                actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
                actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
                actor_rollout_ref.rollout.enforce_eager=True \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.rollout.enable_chunked_prefill=True \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
                reward.reward_model.enable=false \
                actor_rollout_ref.actor.clip_ratio_high=0.20 \
                actor_rollout_ref.rollout.enable_prefix_caching=False \
                actor_rollout_ref.rollout.agent.num_workers=4 \
                actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
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
                actor_rollout_ref.rollout.rose.lookahead.enable="$ROSE_LOOKAHEAD_ENABLE" \
                actor_rollout_ref.rollout.rose.lookahead.windows="$ROSE_LOOKAHEAD_WINDOWS" \
                actor_rollout_ref.rollout.rose.lookahead.edit_distance_threshold="$ROSE_LOOKAHEAD_THRESHOLD" \
                actor_rollout_ref.rollout.rose.lookahead.max_trials_per_pivot="$ROSE_LOOKAHEAD_MAX_TRIALS" \
                actor_rollout_ref.rollout.rose.lookahead.max_pivots_per_slot="$ROSE_LOOKAHEAD_MAX_PIVOTS" \
                actor_rollout_ref.rollout.rose.lookahead.max_probe_tokens_per_group="$ROSE_LOOKAHEAD_GROUP_BUDGET" \
                actor_rollout_ref.rollout.rose.lookahead.request_timeout_seconds="$ROSE_LOOKAHEAD_REQUEST_TIMEOUT" \
                actor_rollout_ref.rollout.rose.lookahead.trace_include_token_ids="$ROSE_LOOKAHEAD_TRACE_TOKEN_IDS" \
                "${LOOKAHEAD_TRACE_ARGS[@]}" \
                algorithm.norm_adv_by_std_in_grpo=false \
                algorithm.advantage_extra_fields='["rose_tree_metadata"]' \
                algorithm.rose.length_calibration_alpha="$ROSE_ALPHA" \
                reward.reward_manager.name=naive \
                reward.num_workers=${REWARD_NUM_WORKERS} \
                reward.custom_reward_function.path="${REWARD_FILE}" \
                trainer.balance_batch=True \
                trainer.logger='["console","tensorboard"]' \
                trainer.project_name=${PROJECT_NAME} \
                trainer.experiment_name=${EXP_NAME} \
                trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
                trainer.validation_data_dir="${VAL_ROLLOUT_DIR}" \
                trainer.n_gpus_per_node=${NTRAINER_GPUS_PER_NODE} \
                trainer.nnodes=${TRAINER_NNODES} \
                trainer.device=npu \
                trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
                trainer.val_before_train=False \
                trainer.v1.sampler.sync_refill_failed_groups=True \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=${VAL_FREQ} \
                trainer.total_epochs=${TOTAL_EPOCHS} 2>&1 | tee -a ${LOG_PATH}/${SAVELOG_FILE_NAME}

            python_exit_code=${PIPESTATUS[0]}
            if [ "${python_exit_code}" -ne 0 ]; then
                echo "Training failed with exit code ${python_exit_code}...."
                sleep 3h
                exit ${python_exit_code}
            fi
            break
        fi

        echo "Waiting for Ray nodes. Current=${device_count}, expected=${NNODES}"
        sleep 10
    done
else
    # # ---- 工作节点：注册到 Ray Head ----
    # while true; do
    #     sleep 5

    #     # echo "Waiting for Ray Head (GCS) to be ready..."
    #     # while ! nc -z ${MASTER_ADDR} ${MASTER_PORT}; do
    #     #     sleep 30
    #     # done


    #     echo "Ray Head is up! Starting worker..."
    #     ray start \
    #         --node-ip-address=${MA_CURRENT_IP} \
    #         --address=${MASTER_ADDR}:${MASTER_PORT} \
    #         --resources='{"NPU": '${NPUS_PER_NODE}'}'

    #     if ray status; then
    #         echo "Worker connected to Ray cluster"
    #         sleep 600
    #         break
    #     fi

    #     echo "Failed to connect Ray head, retrying..."
    #     sleep 10
    # done

    echo "Ray Head is up! Starting worker..."

    ray start \
    --node-ip-address=${MA_CURRENT_IP} \
    --address=${MASTER_ADDR}:${MASTER_PORT} \
    --resources='{"NPU": '${NPUS_PER_NODE}'}'

    sleep infinity
fi
