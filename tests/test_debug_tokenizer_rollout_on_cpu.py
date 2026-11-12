import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "debug_tokenizer_rollout.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("debug_tokenizer_rollout", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_token_ids_accepts_common_chat_template_outputs():
    module = _load_script_module()

    assert module.normalize_token_ids([1, 2, 3], source="list") == [1, 2, 3]
    assert module.normalize_token_ids([[1, 2, 3]], source="nested") == [1, 2, 3]
    assert module.normalize_token_ids(np.array([[1, 2, 3]]), source="array") == [1, 2, 3]
    assert module.normalize_token_ids({"input_ids": [[1, 2, 3]]}, source="BatchEncoding") == [1, 2, 3]


def test_normalize_token_ids_rejects_rendered_text():
    module = _load_script_module()

    with pytest.raises(TypeError, match="returned text instead of token IDs"):
        module.normalize_token_ids("<|im_start|>user", source="chat template")


def test_compare_round_trip_reports_first_difference():
    module = _load_script_module()

    comparison = module.compare_token_sequences([10, 20, 30], [10, 99, 30])

    assert comparison["equal"] is False
    assert comparison["first_difference"] == 1
    assert comparison["left_value"] == 20
    assert comparison["right_value"] == 99


def test_inspect_embedding_rows_detects_invalid_token_id():
    module = _load_script_module()
    embeddings = np.ones((4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="outside embedding vocabulary"):
        module.inspect_embedding_rows(embeddings, [0, 4])


def test_inspect_embedding_rows_reports_finite_norms():
    module = _load_script_module()
    embeddings = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)

    report = module.inspect_embedding_rows(embeddings, [0, 1])

    assert report["shape"] == [2, 2]
    assert report["all_finite"] is True
    assert report["row_norms"] == pytest.approx([5.0, 2.0])


def test_compare_selected_embeddings_reports_row_cosines():
    module = _load_script_module()
    model_rows = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    rose_rows = np.array([[3.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    report = module.compare_selected_embeddings(model_rows, rose_rows, [7, 8])

    assert report["token_ids"] == [7, 8]
    assert report["cosine_similarity"] == pytest.approx([1.0, 0.0])
    assert report["minimum_cosine"] == pytest.approx(0.0)
