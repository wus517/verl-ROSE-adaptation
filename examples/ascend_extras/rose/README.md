# ROSE on Ascend NPU

This example runs the first supported ROSE configuration: V1 synchronous
training, single-turn text prompts, vLLM-Ascend rollout, binary rewards, and
host-CPU semantic scoring.

The canonical entrypoint is now in the repository root. Edit `BASE_PATH` and
`MODEL_PATH` at the top of `run_rose.bash`; it automatically validates the
dataset/reward files, prepares the normalized embedding table, detects the
visible NPU count, and starts V1 ROSE training:

```bash
bash run_rose.bash
```

The older `examples/ascend_extras/rose/run_qwen3_4b_fsdp2.sh` remains as a
fully explicit template. Start with `ROLLOUT_N=2` and a short response length
for the first smoke test by changing the corresponding defaults in the root
script. Do not enable fully asynchronous training, multi-turn/tool rollout,
multimodal inputs, or speculative decoding in the initial version.

The semantic scorer runs on the NPU server's host CPU. Model generation,
sampled log-probability computation, reference evaluation, and PPO updates
remain on Ascend NPUs. Profile `timing/rose_semantic_score` before considering
an independent NPU scorer.
