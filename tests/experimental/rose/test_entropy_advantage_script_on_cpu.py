# Copyright 2026 individual contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Check server commands without executing deployment, package installs or Ray."""

import re
import shlex
import subprocess
import sys
from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from verl.utils.config import validate_config, validate_entropy_advantage_config

ROOT = Path(__file__).resolve().parents[3]


def command_overrides(script):
    command = script.split("python3 -m verl.trainer.main_ppo \\\n", 1)[1].split(" 2>&1 | tee", 1)[0]
    return shlex.split(command.replace("\\\n", " "))


def test_script_preserves_all_non_ea_overrides_and_paths():
    baseline = (ROOT / "run_rose_new.sh").read_text()
    ea = (ROOT / "run_rose_ea.sh").read_text()
    assert [x for x in command_overrides(ea) if ".entropy_advantage." not in x] == command_overrides(baseline)
    for name in ("BASE_PATH", "SAVE_PATH", "MODEL_PATH", "TRAIN_FILE", "VAL_FILE_HIGH", "VAL_FILE_LOW", "REWARD_FILE"):
        pattern = rf"^{name}=.*$"
        assert re.search(pattern, baseline, re.M).group() == re.search(pattern, ea, re.M).group()
    subprocess.run(["bash", "-n", str(ROOT / "run_rose_ea.sh")], check=True)
    assert 'cd "${PROJECT_ROOT}" || exit 1' in ea
    assert 'export PYTHONPATH="${PROJECT_ROOT}' in ea
    for suffix in ("tensorboard/rose_ea/", "rollout_history/rose_ea/", "rl_ckpt/rose_ea/", "log/rose_ea/"):
        assert suffix in ea


@pytest.mark.parametrize("enable", ["True", "False"])
def test_actual_script_command_composes_with_hydra(enable):
    script = (ROOT / "run_rose_ea.sh").read_text()
    # Evaluate ONLY the explicit default-value assignments, never deployment shell code.
    variables = {
        "BASE_PATH": "/test/base",
        "SAVE_PATH": "/test/save",
        "XYI_DIR": "/test/base/xyi",
        "NNODES": "2",
        "TRAINER_NNODES": "2",
        "EXP_NAME": "test_ea",
        "EMBEDDING_PATH": "/test/embed.pt",
        "EMBEDDING_META_PATH": "/test/embed.json",
        "TRAIN_ROLLOUT_DIR": "/test/train",
        "VAL_ROLLOUT_DIR": "/test/val",
        "EA_ENABLE": enable,
    }
    for name, default in re.findall(r"^([A-Z_]+)=\$\{[A-Z_]+:-(.*)\}(?:\s*#.*)?$", script, re.M):
        variables.setdefault(name, default)

    def expand(value):
        for _ in range(8):
            updated = re.sub(r"\$\{([A-Z_]+)\}", lambda match: variables[match[1]], value)
            if updated == value:
                return value
            value = updated
        raise AssertionError("unresolved nested variable")

    overrides = [expand(x) for x in command_overrides(script)]
    with hydra.initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = hydra.compose(config_name="ppo_trainer", overrides=overrides)
    validate_entropy_advantage_config(config)
    validate_config(config, use_reference_policy=True, use_critic=False)
    assert config.actor_rollout_ref.actor.entropy_advantage.enable is (enable == "True")
    assert config.actor_rollout_ref.actor.strategy == "fsdp2"
    assert config.trainer.device == "npu"
    assert config.actor_rollout_ref.rollout.temperature == 1
    assert not config.actor_rollout_ref.model.use_remove_padding
    assert config.reward.custom_reward_function.path == "/test/base/xyi/verify_choice_answer_new_v4.py"
    assert OmegaConf.select(config, "actor_rollout_ref.actor.entropy_advantage.entropy_chunk_size") == 64


@pytest.mark.parametrize("engine", ["", "megatron", "veomni", "torchtitan"])
def test_generated_config_is_exact_current_generator_output(engine):
    args = [sys.executable, "scripts/print_cfg.py", "--cfg", "job"]
    if engine:
        args.append(f"model_engine={engine}")
    output = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True).stdout
    expected = "actor_rollout_ref" + output.split("actor_rollout_ref", 1)[1]
    suffix = f"_{engine}" if engine else ""
    generated = (ROOT / f"verl/trainer/config/_generated_ppo{suffix}_trainer.yaml").read_text()
    actual = "actor_rollout_ref" + generated.split("actor_rollout_ref", 1)[1]
    assert actual == expected
