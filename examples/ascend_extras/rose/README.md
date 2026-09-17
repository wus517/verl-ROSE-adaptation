# ROSE on Ascend NPU

This example runs the first supported ROSE configuration: V1 synchronous
training, single-turn text prompts, vLLM-Ascend rollout, binary rewards, and
host-CPU semantic scoring.

Prepare a row-normalized embedding table from the exact rollout checkpoint:

```bash
python3 -m verl.experimental.rose.prepare_embeddings \
    --model-path /models/Qwen3-4B-Base \
    --output-path /local_nvme/rose/qwen3_4b_embeddings.f16
```

Place the generated binary and JSON metadata on local NVMe or `/dev/shm` on
every rollout node. Then run:

```bash
MODEL_PATH=/models/Qwen3-4B-Base \
TRAIN_FILE=/data/math/train.parquet \
VAL_FILE=/data/math/test.parquet \
EMBEDDING_PATH=/local_nvme/rose/qwen3_4b_embeddings.f16 \
bash examples/ascend_extras/rose/run_qwen3_4b_fsdp2.sh
```

`rose_npu.yaml` documents the canonical overrides but is not a standalone
Hydra config. Start with `ROLLOUT_N=2` and a short response length for the
first smoke test. Do not enable fully asynchronous training, multi-turn/tool
rollout, multimodal inputs, or speculative decoding in the initial version.

The semantic scorer runs on the NPU server's host CPU. Model generation,
sampled log-probability computation, reference evaluation, and PPO updates
remain on Ascend NPUs. Profile `timing/rose_semantic_score` before considering
an independent NPU scorer.
