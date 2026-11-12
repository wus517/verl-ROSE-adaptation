from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_debug_tokenizer_rollout.sh"


def test_debug_launcher_runs_full_single_npu_diagnostic():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "debug_tokenizer_rollout.py" in text
    assert '--device "npu:${NPU_ID}"' in text
    assert "--load-model" in text
    assert "--generate" in text
    assert "--rose-embedding-path" in text
    assert "--rose-embedding-meta-path" in text
    assert '"enable_thinking": false' in text
    assert "tee" in text


def test_debug_launcher_does_not_start_distributed_training():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "ray start" not in text
    assert "verl.trainer.main_ppo" not in text
