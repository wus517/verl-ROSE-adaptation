# Copyright 2025 Individual Contributor: ROSE-LATR adaptation contributors
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
"""Check launch/config wiring without sourcing platform setup or starting Ray."""

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[3]


def command_arguments(path):
    """Read only the training command, never execute the platform setup script."""
    text = path.read_text()
    command = text.split("python3 -m verl.trainer.main_ppo \\\n", 1)[1].split("2>&1 | tee", 1)[0]
    return shlex.split(command.replace("\\\n", " "))


def test_new_script_preserves_original_training_overrides_except_added_search():
    original = command_arguments(ROOT / "run_rose_new.sh")
    new = command_arguments(ROOT / "run_rose_latr.sh")
    assert [
        arg
        for arg in new
        if not arg.startswith("actor_rollout_ref.rollout.rose.lookahead.") and arg != "${LOOKAHEAD_TRACE_ARGS[@]}"
    ] == original
    assert len([arg for arg in new if arg.startswith("actor_rollout_ref.rollout.rose.lookahead.")]) == 8
    subprocess.run(["bash", "-n", str(ROOT / "run_rose_latr.sh")], check=True)


def test_new_overrides_parse_in_real_hydra_config():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "actor_rollout_ref.rollout.rose.enable=true",
                "actor_rollout_ref.rollout.rose.embedding_path=/tmp/embeddings.f16",
                "actor_rollout_ref.rollout.rose.lookahead.enable=true",
                "actor_rollout_ref.rollout.rose.lookahead.windows=[20,30,50]",
                "actor_rollout_ref.rollout.rose.lookahead.edit_distance_threshold=0.4",
                "actor_rollout_ref.rollout.rose.lookahead.max_trials_per_pivot=2",
                "actor_rollout_ref.rollout.rose.lookahead.max_pivots_per_slot=4",
                "actor_rollout_ref.rollout.rose.lookahead.max_probe_tokens_per_group=700",
                "actor_rollout_ref.rollout.rose.lookahead.request_timeout_seconds=600",
                "actor_rollout_ref.rollout.rose.lookahead.trace_dir='/tmp/trace with spaces'",
                "actor_rollout_ref.rollout.rose.lookahead.trace_include_token_ids=false",
                "trainer.total_training_steps=1",
            ],
        )
    assert cfg.actor_rollout_ref.rollout.rose.lookahead.trace_dir == "/tmp/trace with spaces"
    assert cfg.trainer.total_training_steps == 1


@pytest.mark.parametrize(
    "engine,suffix",
    [(None, "ppo"), ("megatron", "ppo_megatron"), ("veomni", "ppo_veomni"), ("torchtitan", "ppo_torchtitan")],
)
def test_generated_config_matches_current_generator(engine, suffix):
    args = [sys.executable, "scripts/print_cfg.py", "--cfg", "job"]
    if engine is not None:
        args.append(f"model_engine={engine}")
    result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    generated = (ROOT / f"verl/trainer/config/_generated_{suffix}_trainer.yaml").read_text()
    assert (
        generated[generated.index("actor_rollout_ref:") :] == result.stdout[result.stdout.index("actor_rollout_ref:") :]
    )
