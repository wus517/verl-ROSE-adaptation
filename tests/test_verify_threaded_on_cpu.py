"""Run with the optional math reward dependencies installed."""

import asyncio
import importlib.util
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import pytest

pytest.importorskip("math_verify")
pytest.importorskip("latex2sympy2_extended")

VERIFIER = Path(__file__).resolve().parents[1] / "verify.py"


@pytest.fixture
def verifier(tmp_path):
    # Match verl's external-file loader: the synthetic module name is not
    # importable, and the deployment filename differs from verify.py.
    path = tmp_path / "verify_choice_answer_new_v3.py"
    shutil.copyfile(VERIFIER, path)
    spec = importlib.util.spec_from_file_location("custom_reward_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    pool = getattr(module, "_reward_process_pool", None)
    if pool is not None:
        pool.shutdown(wait=True)
    # Avoid keeping the temporary module alive across parametrized tests.
    import sys

    sys.modules.pop(path.stem, None)
    if str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def test_sympy_comparison_from_naive_thread_executor(verifier):
    async def run():
        return await asyncio.get_running_loop().run_in_executor(
            None, lambda: verifier.compute_score("test", r"\boxed{x+x}", "2*x")
        )

    assert asyncio.run(run()) == 1.0


def test_parallel_scores_preserve_answer_rules(verifier):
    cases = [
        (r"\boxed{x+x}", "2*x", 1.0),
        (r"\boxed{x+1}", "x+2", 0.0),
        (r"\boxed{(C)(A)}", "A,C", 1.0),
        (r"\boxed{2;1}", "1,2", 0.0),
        (r"\boxed{30°}", "30", 1.0),
        (r"\boxed{25%}", "0.25", 0.0),
        (r"\boxed{1}</think>\boxed{2}", "2", 1.0),
        ("no box", "2", 0.0),
        (r"\boxed{2}", ["1", "2"], 1.0),
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(verifier.compute_score, "test", pred, gt) for pred, gt, _ in cases]
        assert [future.result(timeout=30) for future in futures] == [want for _, _, want in cases]


def test_legacy_process_entry_does_not_start_another_pool(verifier):
    assert verifier.compute_score_in_process("test", r"\boxed{x+x}", "2*x") == 1.0
    assert getattr(verifier, "_reward_process_pool", None) is None


def test_signal_timeout_still_interrupts_slow_work(verifier):
    slow = verifier.timeout_ours(timeout_seconds=1)(sleep)
    with pytest.raises(TimeoutError):
        slow(10)
