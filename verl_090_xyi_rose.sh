#!/bin/bash

# 管道中任一命令失败就返回非零（tee 场景下有用于取 python 退出码）
set -o pipefail

export ASCEND_HOME=/usr/local/Ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.2
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.2/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-8.5.2
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.2/lib64:/usr/local/Ascend/cann-8.5.2/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/usr/local/Ascend/cann-8.5.2/python/site-packages:/usr/local/Ascend/cann-8.5.2/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH


echo "========================================="
echo "开始启动训练"
echo "========================================="

# ========== 解析命令行参数 ==========
TENSORBOARD_OUTPUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --tensorboard_logs=*)
            TENSORBOARD_OUTPUT="${1#*=}"
            shift
            ;;
        --tensorboard_logs)
            TENSORBOARD_OUTPUT="$2"
            shift
            shift
            ;;
        *)
            echo "未知参数: $1"
            shift
            ;;
    esac
done

if [ -n "$TENSORBOARD_OUTPUT" ]; then
    # 去掉结尾多余的斜杠，避免出现 //train.log 这种路径
    TENSORBOARD_OUTPUT="${TENSORBOARD_OUTPUT%/}"
    echo "tensorboard 输出路径: $TENSORBOARD_OUTPUT"
else
    echo "未指定 tensorboard 输出路径，将使用默认路径"
    TENSORBOARD_OUTPUT="/home/ma-user/modelarts/outputs/tensorboard_logs_0"
fi
echo "========================================="

# ========== 【修复】torch_npu 硬编码查找 cann-9.0.0 的问题 ==========
# 背景：
#   开发环境原本是 cann-9.0.0，现被替换为 cann-8.5.2；
#   但 torch_npu 编译时硬编码了默认路径 /usr/local/Ascend/cann-9.0.0，
#   当 ASCEND_HOME_PATH 未传入 Ray worker 时会 fallback 到该路径并报错。
# 处理：
#   建一个软链接 cann-9.0.0 -> cann-8.5.2（幂等，存在则跳过）
if [ ! -e /usr/local/Ascend/cann-9.0.0 ]; then
    echo "创建软链接：/usr/local/Ascend/cann-9.0.0 -> /usr/local/Ascend/cann-8.5.2"
    ln -s /usr/local/Ascend/cann-8.5.2 /usr/local/Ascend/cann-9.0.0
fi
echo "当前 /usr/local/Ascend 内容："
ls -la /usr/local/Ascend/ | grep cann
echo "========================================="

# 设置HCCL超时等环境变量
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64000
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export TORCH_DIST_TIMEOUT=3600
export HCCL_OP_EXPANSION_MODE=AIV
export ACL_OP_COMPILER_CACHE_LEVEL=0

# 设置vLLM相关环境变量
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_DYNAMO=0
export TORCHDYNAMO_DISABLE=1
export HYDRA_FULL_ERROR=1

# ========== 修复 Triton 兼容性问题 ==========
# 优先使用 triton-ascend
export PYTHONPATH=/usr/local/python3.11.15/lib/python3.11/site-packages/triton_ascend:$PYTHONPATH
export TRITON_USE_ASCEND=1

# 跳过 mindspeed 中不兼容的 triton 导入
export MINDSKIP_TRITON_IMPORT=1

# 使用系统默认的python3
PYTHON_CMD="python3"
PIP_CMD="python3 -m pip"

echo "使用Python: $(which $PYTHON_CMD)"
$PYTHON_CMD --version
echo "========================================="

# 定义路径
SCRIPT_DIR="/home/ma-user/modelarts/user-job-dir/lhx_train"
WORK_DIR="/home/ma-user/work"
OUTPUT_DIR="$TENSORBOARD_OUTPUT"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 切换到工作目录
cd $WORK_DIR

# 清理旧的安装目录
echo "清理旧的安装目录..."
rm -rf verl_rose

# 复制压缩包（从脚本所在目录）
echo "从 $SCRIPT_DIR 复制压缩包..."
cp -f $SCRIPT_DIR/xyi/verl_rose.tar.gz ./

# 检查压缩包是否复制成功
echo "检查压缩包..."

if [ ! -f "verl_rose.tar.gz" ]; then
    echo "错误：verl_rose.tar.gz 不存在！"
    exit 1
fi
echo "✓ 所有压缩包已就绪"

echo "升级pip..."
$PIP_CMD install --upgrade pip

# ========== 解压并安装 verl ==========
echo "========================================="
echo "开始解压 verl_rose.tar.gz..."
echo "========================================="
tar -xzf verl_rose.tar.gz

# 定位解压后的目录（verl）
VERL_SRC="$WORK_DIR/verl_rose"
if [ ! -d "$VERL_SRC" ]; then
    echo "✗ 解压目录不存在：$VERL_SRC"
    echo "当前目录内容："
    ls -la "$WORK_DIR"
    exit 1
fi
echo "✓ 解压完成：$VERL_SRC"

# ========== 定位 ROSE 训练入口并注入 CANN 环境变量 ==========
# 目的：确保 Ray worker 子进程能拿到正确的 ASCEND_HOME_PATH。
# 说明：直接使用解压后 verl 仓库根目录的 run_rose.sh，保持脚本自身的
#       SCRIPT_DIR 指向仓库根目录；run_rose.sh 内部训练逻辑不在此处修改。
TRAIN_ENTRY="$VERL_SRC/run_rose.sh"

if [ ! -f "$TRAIN_ENTRY" ]; then
    echo "✗ 训练入口脚本不存在：$TRAIN_ENTRY"
    echo "请确认 $SCRIPT_DIR/xyi/verl_rose.tar.gz 中包含根目录 run_rose.sh"
    exit 1
fi

# run_rose.sh 依赖这些 ROSE 实现。提前检查 tar 包内容，避免安装和复制完
# 模型、数据后才在训练启动阶段因源码不完整而失败。
ROSE_REQUIRED_FILES=(
    "$VERL_SRC/verl/experimental/rose/prepare_embeddings.py"
    "$VERL_SRC/verl/experimental/rose/rose_agent_loop_tq.py"
    "$VERL_SRC/verl/experimental/rose/semantic_entropy.py"
    "$VERL_SRC/verl/experimental/rose/tree.py"
    "$VERL_SRC/verl/experimental/rose/tree_advantage.py"
)
for rose_file in "${ROSE_REQUIRED_FILES[@]}"; do
    if [ ! -f "$rose_file" ]; then
        echo "✗ ROSE 源码不完整，缺少：$rose_file"
        exit 1
    fi
done

if ! grep -q "class RoseRolloutConfig" "$VERL_SRC/verl/workers/config/rollout.py"; then
    echo "✗ rollout 配置中未找到 RoseRolloutConfig"
    exit 1
fi
if ! grep -q "advantage_extra_fields" "$VERL_SRC/verl/trainer/config/ppo_trainer.yaml"; then
    echo "✗ PPO 配置中未找到 advantage_extra_fields"
    exit 1
fi
echo "✓ ROSE 训练入口和依赖源码检查通过"

# 幂等注入：如果已经注入过就跳过（避免重复）
if grep -q "INJECTED_BY_LAUNCH_SCRIPT" "$TRAIN_ENTRY"; then
    echo "✓ entry 脚本已包含 CANN 环境变量注入，跳过"
else
    echo "向 entry 脚本注入 CANN 环境变量..."

    # 使用 python 做精确插入（在 "set -x" 行之后插入），避免 sed 转义问题
    $PYTHON_CMD - "$TRAIN_ENTRY" <<'PYEOF'
import sys, io

path = sys.argv[1]
with io.open(path, 'r', encoding='utf-8') as f:
    content = f.read()

inject = '''# ===== INJECTED_BY_LAUNCH_SCRIPT: CANN 环境变量 =====
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.2
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.2
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.2/opp
export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-8.5.2
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.2/lib64:/usr/local/Ascend/cann-8.5.2/lib:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:$LD_LIBRARY_PATH
# ===== END INJECTED_BY_LAUNCH_SCRIPT =====
'''

lines = content.splitlines(keepends=True)
out = []
injected = False
for i, line in enumerate(lines):
    out.append(line)
    # 兼容 "set -x" 与 run_rose.sh 使用的 "set -xeuo pipefail"。
    stripped = line.strip()
    words = stripped.split()
    enables_xtrace = len(words) >= 2 and words[0] == 'set' and words[1].startswith('-') and 'x' in words[1]
    if (not injected) and enables_xtrace:
        out.append('\n' + inject + '\n')
        injected = True

if not injected:
    # 未启用 xtrace 时保留 shebang，并在它后面插入。
    if out and out[0].startswith('#!'):
        out = [out[0], '\n' + inject + '\n'] + out[1:]
    else:
        out = [inject + '\n'] + out

with io.open(path, 'w', encoding='utf-8') as f:
    f.write(''.join(out))

print("  ✓ 注入完成")
PYEOF

    if [ $? -ne 0 ]; then
        echo "✗ 注入 entry 脚本失败"
        exit 1
    fi
fi

# 打印 entry 脚本头部，确认注入成功
echo "----- entry 脚本前 25 行 -----"
head -25 "$TRAIN_ENTRY"
echo "------------------------------"

# ========== 【核心】安装 verl：先清残留，再 editable 安装，最后强制校验 ==========
# 说明：
#   之前遇到过 Hydra 报 "Key 'custom' is not in struct" 的错误，
#   根因是 site-packages 里残留了旧版 verl，Python import 时优先加载了旧版。
#   这里先彻底清掉 site-packages 里的旧 verl，再装解压目录的版本，
#   最后强制校验 import 路径必须指向 $VERL_SRC，否则直接退出。
echo "========================================="
echo "安装 verl（editable）..."
echo "========================================="

# 1) 找到 site-packages 路径
SITE_PKGS=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
echo "site-packages: $SITE_PKGS"

# 2) 卸载 + 强制删除旧残留（旧目录可能属主是 ma-user，pip uninstall 删不掉，必须 rm -rf）
echo "清理旧的 verl 残留..."
$PIP_CMD uninstall verl -y 2>/dev/null
rm -rf "$SITE_PKGS/verl"
rm -rf "$SITE_PKGS"/verl-*.dist-info
rm -rf "$SITE_PKGS"/__editable__*verl*
echo "残留检查（应无输出）："
ls -la "$SITE_PKGS" | grep -i verl || echo "  ✓ 已清空"

# 3) 安装解压目录里的 verl（--no-deps，避免重装依赖时把环境搞乱）
cd "$VERL_SRC"
$PIP_CMD install -e . --no-deps
cd "$WORK_DIR"

# 4) 强制校验 import 路径，防止加载到别处旧版本
echo "验证 import 路径..."
IMPORTED_VERL=$($PYTHON_CMD -c "import verl; print(verl.__file__)" 2>/dev/null)
echo "  import verl -> $IMPORTED_VERL"
case "$IMPORTED_VERL" in
    "$VERL_SRC"*)
        echo "  ✓ import 路径正确"
        ;;
    *)
        echo "  ✗ 警告：import 路径不是解压目录！"
        echo "     期望: $VERL_SRC/verl/__init__.py"
        echo "     实际: $IMPORTED_VERL"
        echo "     请检查 PYTHONPATH 与 site-packages 残留"
        exit 1
        ;;
esac

echo "✓ verl 安装并校验完成"

# ========== 安装奖励函数与 tokenizer 所需的依赖 ==========
echo "========================================="
echo "安装奖励函数与 tokenizer 所需的依赖..."
echo "========================================="

# 1) 安装 math_verify 模块（奖励函数需要）
echo "安装 math_verify..."
$PIP_CMD install math-verify

echo "验证 math_verify 安装..."
$PYTHON_CMD -c "import math_verify; print('✓ math_verify installed successfully')" 2>&1
if [ $? -eq 0 ]; then
    echo "✓ math_verify 安装成功"
else
    echo "✗ math_verify 安装失败，尝试从源码安装..."
    $PIP_CMD install git+https://github.com/huggingface/math-verify.git
fi

# 2) 安装 tokenizer 依赖（Qwen3 tokenizer 需要 sentencepiece 或 tiktoken）
echo "安装 tokenizer 依赖（sentencepiece / tiktoken）..."
$PIP_CMD install sentencepiece tiktoken

echo "验证 tokenizer 依赖..."
$PYTHON_CMD -c "import sentencepiece; print('✓ sentencepiece', sentencepiece.__version__)" 2>&1
$PYTHON_CMD -c "import tiktoken; print('✓ tiktoken', tiktoken.__version__)" 2>&1

echo "========================================="

$PIP_CMD list

# ========== 准备模型 ==========
# 【重要】这里做自动探测，兼容两种源目录结构：
#   情况 A：$SCRIPT_DIR/Qwen3-0.6B/config.json       -> 源目录本身就是模型目录
#   情况 B：$SCRIPT_DIR/Qwen3-0.6B/main/config.json  -> 源目录多嵌套一层 main
# 之前遇到过情况 B 导致 model/Qwen3-0.6B/main/ 下又套了一层 main，
# 训练时 transformers 在 MODEL_PATH 里找不到 tokenizer.json，
# 最终误报 "sentencepiece or tiktoken 缺失"。
echo "========================================="
echo "准备模型..."
echo "========================================="

mkdir -p $WORK_DIR/model

MODEL_SOURCE_BASE="$SCRIPT_DIR/Qwen3-0.6B"
MODEL_TARGET="$WORK_DIR/model/Qwen3-0.6B/main"

# ---- 探测模型文件实际所在层 ----
if [ -f "$MODEL_SOURCE_BASE/config.json" ]; then
    MODEL_SOURCE="$MODEL_SOURCE_BASE"
    echo "✓ 检测到模型文件在：$MODEL_SOURCE_BASE"
elif [ -f "$MODEL_SOURCE_BASE/main/config.json" ]; then
    MODEL_SOURCE="$MODEL_SOURCE_BASE/main"
    echo "✓ 检测到模型文件在：$MODEL_SOURCE_BASE/main（源目录多嵌套一层 main）"
else
    echo "✗ 在 $MODEL_SOURCE_BASE 未找到 config.json，模型目录结构异常"
    echo "源目录内容："
    ls -la "$MODEL_SOURCE_BASE" 2>/dev/null
    echo "尝试查找 config.json："
    find "$MODEL_SOURCE_BASE" -name "config.json" -type f 2>/dev/null | head -5
    exit 1
fi

# ---- 清理旧目标 ----
if [ -d "$MODEL_TARGET" ]; then
    echo "目标模型已存在，正在删除..."
    rm -rf "$MODEL_TARGET"
fi

# ---- 复制 ----
mkdir -p "$WORK_DIR/model/Qwen3-0.6B"
echo "正在复制模型到：$MODEL_TARGET"
echo "这可能需要几分钟时间..."
cp -r "$MODEL_SOURCE" "$MODEL_TARGET"

# ---- 强制校验：config.json / tokenizer.json 必须存在 ----
if [ ! -f "$MODEL_TARGET/config.json" ]; then
    echo "✗ 复制后未找到 config.json：$MODEL_TARGET"
    echo "目录内容："
    ls -la "$MODEL_TARGET" 2>/dev/null
    exit 1
fi
if [ ! -f "$MODEL_TARGET/tokenizer.json" ]; then
    echo "✗ 复制后未找到 tokenizer.json：$MODEL_TARGET"
    echo "目录内容："
    ls -la "$MODEL_TARGET" 2>/dev/null
    exit 1
fi

echo "✓ 模型复制成功！"
echo "模型路径：$MODEL_TARGET"
echo "模型内容："
ls -la "$MODEL_TARGET" | head -15

# ---- 复制后立即验证 tokenizer 能否加载（提前暴露问题） ----
echo "验证 tokenizer 能否加载..."
$PYTHON_CMD -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_TARGET', trust_remote_code=True)
print('✓ tokenizer OK:', type(tok).__name__, 'vocab_size=', tok.vocab_size)
" 2>&1
if [ $? -ne 0 ]; then
    echo "✗ tokenizer 加载失败，请检查模型目录是否完整"
    exit 1
fi

# ========== 准备数据集和奖励函数 ==========
# 训练脚本中路径为：
#   BASE_PATH='/home/ma-user/work'
#   数据：${BASE_PATH}/dataset/xyi/*.json
#   奖励：${BASE_PATH}/rewards/xyi/reward_math_verifier.py
echo "========================================="
echo "准备数据集和奖励函数..."
echo "========================================="

XYI_SOURCE="$SCRIPT_DIR/xyi"
DATASET_TARGET="$WORK_DIR/dataset/xyi"
REWARD_TARGET="$WORK_DIR/rewards/xyi"

if [ ! -d "$XYI_SOURCE" ]; then
    echo "✗ 源 xyi 目录不存在：$XYI_SOURCE"
    exit 1
fi
echo "✓ 找到源 xyi 目录：$XYI_SOURCE"

# 创建目标目录
mkdir -p "$DATASET_TARGET"
mkdir -p "$REWARD_TARGET"

# 复制数据文件到 dataset/xyi
echo "复制数据文件到：$DATASET_TARGET"
cp -f "$XYI_SOURCE/train.json" "$DATASET_TARGET/" 2>/dev/null
cp -f "$XYI_SOURCE/val_high_pass.json" "$DATASET_TARGET/" 2>/dev/null
cp -f "$XYI_SOURCE/val_low_pass.json" "$DATASET_TARGET/" 2>/dev/null

# 复制奖励函数到 rewards/xyi
echo "复制奖励函数到：$REWARD_TARGET"
cp -f "$XYI_SOURCE/reward_math_verifier.py" "$REWARD_TARGET/" 2>/dev/null

# 兼容：也保留一份 xyi 目录到 work（可选，防止其他引用）
XYI_TARGET="$WORK_DIR/xyi"
if [ ! -d "$XYI_TARGET" ]; then
    cp -r "$XYI_SOURCE" "$XYI_TARGET"
fi

# ========== 验证数据文件 ==========
echo "========================================="
echo "验证数据文件和奖励函数..."
echo "========================================="

# 检查训练数据
TRAIN_DATA="$DATASET_TARGET/train.json"
if [ -f "$TRAIN_DATA" ]; then
    echo "✓ 训练数据存在：$TRAIN_DATA"
    ls -lh "$TRAIN_DATA"
else
    echo "✗ 训练数据不存在：$TRAIN_DATA"
    echo "请检查文件是否在 $XYI_SOURCE/train.json"
    exit 1
fi

# 检查验证数据
VAL_HIGH="$DATASET_TARGET/val_high_pass.json"
VAL_LOW="$DATASET_TARGET/val_low_pass.json"
if [ -f "$VAL_HIGH" ]; then
    echo "✓ 验证数据(high_pass)存在：$VAL_HIGH"
    ls -lh "$VAL_HIGH"
else
    echo "✗ 验证数据(high_pass)不存在：$VAL_HIGH"
    echo "请检查文件是否在 $XYI_SOURCE/val_high_pass.json"
    exit 1
fi

if [ -f "$VAL_LOW" ]; then
    echo "✓ 验证数据(low_pass)存在：$VAL_LOW"
    ls -lh "$VAL_LOW"
else
    echo "✗ 验证数据(low_pass)不存在：$VAL_LOW"
    echo "请检查文件是否在 $XYI_SOURCE/val_low_pass.json"
    exit 1
fi

# 检查奖励函数
REWARD_FILE="$REWARD_TARGET/reward_math_verifier.py"
if [ -f "$REWARD_FILE" ]; then
    echo "✓ 奖励函数存在：$REWARD_FILE"
    ls -lh "$REWARD_FILE"
else
    echo "✗ 奖励函数不存在：$REWARD_FILE"
    echo "请检查文件是否在 $XYI_SOURCE/reward_math_verifier.py"
    exit 1
fi
echo "========================================="

# ========== 设置 tensorboard 环境变量 ==========
export TENSORBOARD_DIR="$OUTPUT_DIR"
export TENSORBOARD_LOG_DIR="$OUTPUT_DIR"
export TB_LOG_DIR="$OUTPUT_DIR"

echo "========================================="
echo "设置 tensorboard 环境变量："
echo "  TENSORBOARD_DIR=$TENSORBOARD_DIR"
echo "  TENSORBOARD_LOG_DIR=$TENSORBOARD_LOG_DIR"
echo "  TB_LOG_DIR=$TB_LOG_DIR"
echo "========================================="

# 创建 tensorboard 日志目录
mkdir -p "$OUTPUT_DIR"
echo "✓ tensorboard 日志目录已创建：$OUTPUT_DIR"

# ========== 运行训练 ==========
echo "========================================="
echo "准备运行训练..."
echo "========================================="

# 定义日志文件路径
LOG_FILE="$OUTPUT_DIR/train.log"

echo "训练日志将保存到：$LOG_FILE"
echo "Tensorboard 日志将保存到：$OUTPUT_DIR"
echo "========================================="
echo "开始训练..."
echo "========================================="

# 赋予执行权限，并做一次语法检查（TRAIN_ENTRY 变量已在前面定位）
chmod +x "$TRAIN_ENTRY"
if ! bash -n "$TRAIN_ENTRY"; then
    echo "✗ 训练入口脚本语法检查未通过：$TRAIN_ENTRY"
    exit 1
fi
echo "✓ 训练入口脚本语法检查通过"

# 切换到训练入口脚本所在目录执行（保持相对路径引用）
TRAIN_ENTRY_DIR=$(dirname "$TRAIN_ENTRY")
cd "$TRAIN_ENTRY_DIR"

# 指定 tensorboard 输出路径并执行训练入口脚本
# 使用 bash 显式执行，避免 shebang 问题
TENSORBOARD_DIR="$OUTPUT_DIR" bash "$TRAIN_ENTRY" 2>&1 | tee "$LOG_FILE"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

# ========== 验证 tensorboard 日志 ==========
echo "========================================="
echo "验证 tensorboard 日志..."
echo "========================================="

# ============================================================
# 【调试用 SLEEP】—— 保留你原本的 16 小时等待
# 需要调试时，训练会先睡 16h，期间你可以进容器验证：
#   1) 模型目录：ls -la /home/ma-user/work/model/Qwen3-0.6B/main/
#   2) tokenizer：python3 -c "from transformers import AutoTokenizer;
#        t=AutoTokenizer.from_pretrained('/home/ma-user/work/model/Qwen3-0.6B/main',
#        trust_remote_code=True); print('OK', t.vocab_size)"
#   3) 环境变量：env | grep -iE "ascend|cann"
# 调试完毕后注释掉这行，或改小到 sleep 30
# ============================================================
# sleep 16h

# 检查 OUTPUT_DIR 中是否有 tensorboard 日志
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null | grep -v "train.log")" ]; then
    echo "✓ 在 $OUTPUT_DIR 找到 tensorboard 日志！"
    echo "目录内容："
    ls -la "$OUTPUT_DIR"
    echo "tensorboard 日志文件："
    find "$OUTPUT_DIR" -type f -name "events.out.tfevents.*" -exec ls -lh {} \;
else
    echo "⚠ 在 $OUTPUT_DIR 未找到 tensorboard 日志"
    echo "检查是否在其他位置生成了 tensorboard 日志："
    find "$WORK_DIR" -type f -name "events.out.tfevents.*" 2>/dev/null | head -10
    echo "如果找到，请手动复制到 $OUTPUT_DIR"
fi

# 根据训练退出码显示结果
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "✓ 训练完成！"
    echo "训练日志：$LOG_FILE"
    echo "Tensorboard 日志：$OUTPUT_DIR"
    echo "========================================="
else
    echo "========================================="
    echo "✗ 训练失败！"
    echo "退出码：$TRAIN_EXIT_CODE"
    echo "训练日志：$LOG_FILE"
    echo "Tensorboard 日志：$OUTPUT_DIR"
    echo "========================================="

    if [ -f "$LOG_FILE" ]; then
        echo "日志文件最后100行："
        echo "========================================="
        tail -100 "$LOG_FILE"
        echo "========================================="
    fi
fi

# ========== 最终确认 ==========
echo "========================================="
echo "最终确认所有任务执行完毕！"
echo "========================================="
echo "verl 源码路径：$VERL_SRC"
echo "训练入口脚本：$TRAIN_ENTRY"
echo "训练日志路径：$LOG_FILE"
echo "Tensorboard 日志路径：$OUTPUT_DIR"
echo "数据集路径：$DATASET_TARGET"
echo "奖励函数路径：$REWARD_FILE"
echo "模型路径：$MODEL_TARGET"
echo "========================================="
echo "所有任务执行完毕！"
echo "========================================="
