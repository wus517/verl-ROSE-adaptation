#!/bin/bash
set -x
set -o pipefail

echo "datetime: $(date +'%Y-%m-%d %H:%M:%S')"
echo $MA_CURRENT_HOST_IP
echo $MA_CURRENT_IP

PROXY_ADDR="http://10.31.196.119:80"
export no_proxy="${no_proxy:+$no_proxy,}10.44.***.***"
export http_proxy="$PROXY_ADDR"
export https_proxy="$PROXY_ADDR"
export HTTP_PROXY="$PROXY_ADDR"
export HTTPS_PROXY="$PROXY_ADDR"
export FTP_PROXY="$PROXY_ADDR"
export ftp_proxy="$PROXY_ADDR"

export LD_PRELOAD=/home/ma-user/Ascend/cann-8.5.2/lib64/libprofapi.so${LD_PRELOAD:+:$LD_PRELOAD}
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

source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
source /home/ma-user/Ascend/cann-8.5.2/set_env.sh
source /home/ma-user/Ascend/nnal/atb/set_env.sh

echo "--------------------- adapt MTP end!-----------------------"
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0
export ACLNN_CACHE_LIMIT=100000
export CPU_AFFINITY_CONF=1
unset PYTORCH_NPU_ALLOC_CONF
export HCCL_BUFFSIZE=64
export ACL_OP_COMPILER_CACHE_DIR=$ACL_OP_COMPILER_CACHE_DIR
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV

python -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"

BASE_PATH=${BASE_PATH:-${lhx_sh}}
SAVE_PATH=${SAVE_PATH:-${lhx_save}}
echo "BASE_PATH resolved to: ${BASE_PATH}"

XYI_DIR="${BASE_PATH}/xyi/v8_all"
TRAIN_FILE=${TRAIN_FILE:-${XYI_DIR}/math.train.parquet}
VAL_EASY=${VAL_EASY:-${XYI_DIR}/math.val.easy.parquet}
VAL_ABITHARD=${VAL_ABITHARD:-${XYI_DIR}/math.val.a_bit_hard.parquet}
VAL_HARD=${VAL_HARD:-${XYI_DIR}/math.val.hard.parquet}
VAL_VERYHARD=${VAL_VERYHARD:-${XYI_DIR}/math.val.very_hard.parquet}
TEST_EASY=${TEST_EASY:-${XYI_DIR}/math.test.easy.parquet}
TEST_ABITHARD=${TEST_ABITHARD:-${XYI_DIR}/math.test.a_bit_hard.parquet}
TEST_HARD=${TEST_HARD:-${XYI_DIR}/math.test.hard.parquet}
TEST_VERYHARD=${TEST_VERYHARD:-${XYI_DIR}/math.test.very_hard.parquet}
REWARD_FILE=${REWARD_FILE:-${XYI_DIR}/verify_choice_answer_new_v4.py}

echo "TRAIN_FILE      = ${TRAIN_FILE}"
echo "VAL_EASY   = ${VAL_EASY}"
echo "VAL_ABITHARD    = ${VAL_ABITHARD}"
echo "VAL_HARD   = ${VAL_HARD}"
echo "VAL_VERYHARD     = ${VAL_VERYHARD}"
echo "TEST_EASY   = ${TEST_EASY}"
echo "TEST_ABITHARD    = ${TEST_ABITHARD}"
echo "TEST_HARD   = ${TEST_HARD}"
echo "TEST_VERYHARD     = ${TEST_VERYHARD}"
pip install "accelerate>=1.6.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import accelerate; print(accelerate.__version__)"
MODEL_PATH=${MODEL_PATH:-${BASE_PATH}/model/Qwen3_5_9B}
NNODES=${NNODES:-$MA_NUM_HOSTS}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${EXP_NAME:-qwen3_5_9b_steer_quadrant_v8}}

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
TENSORBOARD_DIR="${SAVE_PATH}/tensorboard/steer_quadrant_v8/${EXPERIMENT_NAME}_${TIMESTAMP}"
mkdir -p "${TENSORBOARD_DIR}"
echo "TENSORBOARD_DIR = ${TENSORBOARD_DIR}"

ROLLOUT_HISTORY_DIR="${SAVE_PATH}/rollout_history/steer_quadrant_v8/${EXPERIMENT_NAME}_${TIMESTAMP}"

TRAIN_ROLLOUT_DIR="${ROLLOUT_HISTORY_DIR}/train"
VAL_ROLLOUT_DIR="${ROLLOUT_HISTORY_DIR}/val"

mkdir -p "${TRAIN_ROLLOUT_DIR}" "${VAL_ROLLOUT_DIR}"

echo "TRAIN_ROLLOUT_DIR = ${TRAIN_ROLLOUT_DIR}"
echo "VAL_ROLLOUT_DIR   = ${VAL_ROLLOUT_DIR}"

NGPUS_PER_NODE=${MA_NUM_GPUS:-8}

REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-16}

NTRAINER_GPUS_PER_NODE=${NTRAINER_GPUS_PER_NODE:-8}
TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}

PROJECT_NAME=${PROJECT_NAME:-verl_steer_quadrant_v8_mathv8}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-48}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-12888}

ACTOR_LR=${ACTOR_LR:-1e-6}
PPO_EPOCHS=${PPO_EPOCHS:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}

# Set STEER_RECOVERY_QUADRANTS='[]' and a distinct EXPERIMENT_NAME for the STEER control.
STEER_TOKEN_WEIGHT_MIN=${STEER_TOKEN_WEIGHT_MIN:-0.7}
STEER_TOKEN_WEIGHT_MAX=${STEER_TOKEN_WEIGHT_MAX:-1.0}
STEER_LINEAR=${STEER_LINEAR:-false}
STEER_ENTROPY_CONTROL_MODE=${STEER_ENTROPY_CONTROL_MODE:-symmetric}
STEER_QUADRANT_LOGGING=${STEER_QUADRANT_LOGGING:-true}
STEER_RECOVERY_QUADRANTS=${STEER_RECOVERY_QUADRANTS:-'[1,2,3,4]'}
STEER_RECOVERY_TOP_RATIO=${STEER_RECOVERY_TOP_RATIO:-0.20}
STEER_RECOVERY_MIN_ENTROPY=${STEER_RECOVERY_MIN_ENTROPY:-0.5}
STEER_RECOVERY_RATE=${STEER_RECOVERY_RATE:-0.25}
STEER_ENTROPY_WEIGHTING=${STEER_ENTROPY_WEIGHTING:-true}
STEER_LOW_ENTROPY_WEIGHT_MIN=${STEER_LOW_ENTROPY_WEIGHT_MIN:-0.50}
STEER_HIGH_ENTROPY_WEIGHT_MIN_BY_QUADRANT=${STEER_HIGH_ENTROPY_WEIGHT_MIN_BY_QUADRANT:-'[0.70,0.70,0.70,0.70]'}
STEER_RECOVERY_RATE_BY_QUADRANT=${STEER_RECOVERY_RATE_BY_QUADRANT:-'[0.35,0.35,0.35,0.35]'}
GRAD_CLIP=${GRAD_CLIP:-1.0}
OVERLONG_ENABLE=${OVERLONG_ENABLE:-True}
OVERLONG_BUFFER_LENGTH=${OVERLONG_BUFFER_LENGTH:-2048}

ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.99}

DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-${SAVE_PATH}/saved_models/rl_ckpt/steer_quadrant_v8/${EXPERIMENT_NAME}}

LOG_PATH=${LOG_PATH:-${SAVE_PATH}/qwen35_rl/log/steer_quadrant_v8/${EXPERIMENT_NAME}/}
echo "LOG_PATH resolved to: ${LOG_PATH}"

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-5}
VAL_FREQ=${VAL_FREQ:-5}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-12888}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-12888}

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

# Put the adapted package on the Python path before Ray starts on each node.
if [ -z "${VERL_CODE_DIR:-}" ]; then
    SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    for candidate in "${SCRIPT_DIR}/verl-0.9.0" "${PROJECT_ROOT}/verl-0.9.0" \
        "${PROJECT_ROOT}/algorithm/verl-0.9.0" "${PROJECT_ROOT}"; do
        if [ -f "${candidate}/verl/trainer/ppo/core_algos.py" ]; then
            VERL_CODE_DIR=${candidate}
            break
        fi
    done
fi
if [ -n "${VERL_CODE_DIR:-}" ]; then
    VERL_CODE_DIR=$(cd "${VERL_CODE_DIR}" && pwd) || exit 1
    export PYTHONPATH="${VERL_CODE_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if ! python3 - <<'PY_CHECK_STEER_V8'
import inspect

from verl.trainer.ppo.core_algos import compute_steer_v8_weights

print("STEER V8 source:", inspect.getfile(inspect.unwrap(compute_steer_v8_weights)))
PY_CHECK_STEER_V8
then
    echo "STEER V8 source check failed. Set VERL_CODE_DIR to the adapted verl-0.9.0 checkout." >&2
    exit 1
fi

export RAY_HEARTBEAT_TIMEOUT_MS=60000
export RAY_gcs_actor_failure_grace_period_ms=60000

echo $RAY_gcs_rpc_server_reconnect_timeout_s

if [ "${NODE_RANK}" = "0" ]; then
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

            mkdir -p "${LOG_PATH}"

            python3 -m verl.trainer.main_ppo \
                algorithm.adv_estimator=grpo \
                algorithm.rollout_correction.rollout_is=null \
                algorithm.rollout_correction.rollout_rs=null \
                algorithm.rollout_correction.bypass_mode=False \
                algorithm.use_kl_in_reward=False \
                data.train_files="['${TRAIN_FILE}']" \
                data.val_files="['${VAL_EASY}', '${VAL_ABITHARD}', '${VAL_HARD}', '${VAL_VERYHARD}', '${TEST_EASY}', '${TEST_ABITHARD}', '${TEST_HARD}', '${TEST_VERYHARD}']" \
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
                actor_rollout_ref.actor.policy_loss.loss_mode=entropy_control \
                actor_rollout_ref.actor.policy_loss.token_weight_min=${STEER_TOKEN_WEIGHT_MIN} \
                actor_rollout_ref.actor.policy_loss.token_weight_max=${STEER_TOKEN_WEIGHT_MAX} \
                actor_rollout_ref.actor.policy_loss.linear=${STEER_LINEAR} \
                actor_rollout_ref.actor.policy_loss.entropy_control_mode=${STEER_ENTROPY_CONTROL_MODE} \
                actor_rollout_ref.actor.policy_loss.steer_quadrant_logging=${STEER_QUADRANT_LOGGING} \
                actor_rollout_ref.actor.policy_loss.q2_high_entropy_ratio=0.0 \
                actor_rollout_ref.actor.policy_loss.q2_high_entropy_boost_strength=0.0 \
                "actor_rollout_ref.actor.policy_loss.steer_recovery_quadrants=${STEER_RECOVERY_QUADRANTS}" \
                "actor_rollout_ref.actor.policy_loss.steer_recovery_top_ratio=${STEER_RECOVERY_TOP_RATIO}" \
                "actor_rollout_ref.actor.policy_loss.steer_recovery_min_entropy=${STEER_RECOVERY_MIN_ENTROPY}" \
                "actor_rollout_ref.actor.policy_loss.steer_recovery_rate=${STEER_RECOVERY_RATE}" \
                "actor_rollout_ref.actor.policy_loss.steer_entropy_weighting=${STEER_ENTROPY_WEIGHTING}" \
                "actor_rollout_ref.actor.policy_loss.steer_low_entropy_weight_min=${STEER_LOW_ENTROPY_WEIGHT_MIN}" \
                "actor_rollout_ref.actor.policy_loss.steer_high_entropy_weight_min_by_quadrant=${STEER_HIGH_ENTROPY_WEIGHT_MIN_BY_QUADRANT}" \
                "actor_rollout_ref.actor.policy_loss.steer_recovery_rate_by_quadrant=${STEER_RECOVERY_RATE_BY_QUADRANT}" \
                actor_rollout_ref.actor.loss_agg_mode=token-mean \
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
                actor_rollout_ref.actor.kl_loss_coef=0 \
                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                actor_rollout_ref.actor.entropy_coeff=0 \
                actor_rollout_ref.actor.calculate_entropy=False \
                actor_rollout_ref.actor.entropy_from_logits_with_chunking=False \
                actor_rollout_ref.actor.fsdp_config.param_offload=True \
                actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
                actor_rollout_ref.actor.fsdp_config.offload_policy=False \
                actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
                actor_rollout_ref.actor.fsdp_config.grad_sync_every_micro_batch=True \
                actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']" \
                actor_rollout_ref.ref.strategy=fsdp2 \
                actor_rollout_ref.ref.fsdp_config.param_offload=True \
                actor_rollout_ref.ref.fsdp_config.forward_prefetch=False \
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
                actor_rollout_ref.rollout.enforce_eager=False \
                actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
                actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
                actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU} \
                actor_rollout_ref.rollout.enable_chunked_prefill=True \
                actor_rollout_ref.rollout.free_cache_engine=True \
                actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
                actor_rollout_ref.rollout.enable_prefix_caching=False \
                actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
                actor_rollout_ref.rollout.calculate_log_probs=true \
                reward.reward_manager.name=dapo \
                ++reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
                ++reward.reward_kwargs.overlong_buffer_cfg.enable=${OVERLONG_ENABLE} \
                ++reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LENGTH} \
                ++reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
                ++reward.reward_kwargs.overlong_buffer_cfg.log=True \
                reward.num_workers=${REWARD_NUM_WORKERS} \
                reward.custom_reward_function.path="${REWARD_FILE}" \
                trainer.use_v1=True \
                trainer.v1.trainer_mode=sync \
                trainer.balance_batch=True \
                trainer.logger='["console","tensorboard"]' \
                trainer.project_name=${PROJECT_NAME} \
                trainer.experiment_name=${EXPERIMENT_NAME} \
                trainer.rollout_data_dir="${TRAIN_ROLLOUT_DIR}" \
                trainer.validation_data_dir="${VAL_ROLLOUT_DIR}" \
                trainer.n_gpus_per_node=${NTRAINER_GPUS_PER_NODE} \
                trainer.nnodes=${TRAINER_NNODES} \
                trainer.device=npu \
                trainer.default_local_dir=${DEFAULT_LOCAL_DIR} \
                trainer.val_before_train=False \
                trainer.save_freq=${SAVE_FREQ} \
                trainer.test_freq=${VAL_FREQ} \
                trainer.total_epochs=${TOTAL_EPOCHS} 2>&1 | tee -a "${LOG_PATH}/${SAVELOG_FILE_NAME}"
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

    echo "Ray Head is up! Starting worker..."

    ray start \
    --node-ip-address=${MA_CURRENT_IP} \
    --address=${MASTER_ADDR}:${MASTER_PORT} \
    --resources='{"NPU": '${NPUS_PER_NODE}'}'

    sleep infinity
fi