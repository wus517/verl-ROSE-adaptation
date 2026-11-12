from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def normalize_token_ids(value: Any, *, source: str) -> list[int]:
    """Convert common tokenizer outputs to a flat Python integer list."""
    if isinstance(value, str):
        raise TypeError(f"{source} returned text instead of token IDs")

    # Transformers may return a BatchEncoding even when return_tensors=None.
    # It is mapping-like and stores the actual token sequence under input_ids.
    if isinstance(value, Mapping) or hasattr(value, "keys") and hasattr(value, "__getitem__"):
        if "input_ids" not in value:
            available_keys = list(value.keys()) if hasattr(value, "keys") else []
            raise TypeError(f"{source} returned mapping without input_ids: {available_keys}")
        value = value["input_ids"]

    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"{source} returned unsupported type {type(value).__name__}")

    while value and isinstance(value[0], list | tuple):
        if len(value) != 1:
            raise ValueError(f"{source} returned more than one token sequence")
        value = list(value[0])

    token_ids: list[int] = []
    for index, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int | np.integer):
            raise TypeError(
                f"{source} returned non-integer token at index {index}: {token_id!r} ({type(token_id).__name__})"
            )
        token_ids.append(int(token_id))
    return token_ids


def compare_token_sequences(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    """Return a compact, JSON-friendly comparison of two token sequences."""
    left = list(left)
    right = list(right)
    first_difference = None
    for index, (left_value, right_value) in enumerate(zip(left, right, strict=False)):
        if left_value != right_value:
            first_difference = index
            break
    if first_difference is None and len(left) != len(right):
        first_difference = min(len(left), len(right))
    return {
        "equal": first_difference is None,
        "left_length": len(left),
        "right_length": len(right),
        "first_difference": first_difference,
        "left_value": left[first_difference] if first_difference is not None and first_difference < len(left) else None,
        "right_value": right[first_difference]
        if first_difference is not None and first_difference < len(right)
        else None,
    }


def inspect_embedding_rows(embeddings: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    """Inspect selected rows of a 2-D embedding table."""
    array = np.asarray(embeddings)
    if array.ndim != 2:
        raise ValueError(f"embedding table must be rank 2, got shape {array.shape}")

    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size and (np.any(ids < 0) or np.any(ids >= array.shape[0])):
        bad = ids[(ids < 0) | (ids >= array.shape[0])]
        raise ValueError(f"token id(s) {bad[:10].tolist()} outside embedding vocabulary [0, {array.shape[0]})")

    rows = np.asarray(array[ids]) if ids.size else np.empty((0, array.shape[1]), dtype=array.dtype)
    finite = bool(np.all(np.isfinite(rows))) if rows.size else True
    norms = np.linalg.norm(rows.astype(np.float32), axis=1).tolist() if rows.size else []
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "selected_ids": ids.tolist(),
        "all_finite": finite,
        "row_norms": norms,
        "min": float(rows.min()) if rows.size else None,
        "max": float(rows.max()) if rows.size else None,
    }


def inspect_selected_embedding_rows(rows: Any, token_ids: Sequence[int], vocab_size: int) -> dict[str, Any]:
    """Inspect already-selected rows without copying a full model table to CPU."""
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[0] != len(token_ids):
        raise ValueError(
            f"selected embedding rows must have shape (len(token_ids), hidden_size), "
            f"got {array.shape} for {len(token_ids)} ids"
        )
    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size and (np.any(ids < 0) or np.any(ids >= vocab_size)):
        raise ValueError(f"selected token IDs are outside model vocabulary [0, {vocab_size})")
    return {
        "vocab_size": int(vocab_size),
        "hidden_size": int(array.shape[1]),
        "selected_ids": ids.tolist(),
        "dtype": str(array.dtype),
        "all_finite": bool(np.all(np.isfinite(array))) if array.size else True,
        "row_norms": np.linalg.norm(array.astype(np.float32), axis=1).tolist() if array.size else [],
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def compare_selected_embeddings(model_rows: Any, rose_rows: Any, token_ids: Sequence[int]) -> dict[str, Any]:
    """Compare matching model and ROSE embedding rows by cosine similarity."""
    model_array = np.asarray(model_rows, dtype=np.float32)
    rose_array = np.asarray(rose_rows, dtype=np.float32)
    ids = list(token_ids)
    if model_array.shape != rose_array.shape:
        raise ValueError(f"model rows shape {model_array.shape} does not match ROSE rows shape {rose_array.shape}")
    if model_array.ndim != 2 or model_array.shape[0] != len(ids):
        raise ValueError(f"embedding rows shape {model_array.shape} does not match {len(ids)} token IDs")
    model_norms = np.linalg.norm(model_array, axis=1)
    rose_norms = np.linalg.norm(rose_array, axis=1)
    if np.any(model_norms == 0) or np.any(rose_norms == 0):
        raise ValueError("cannot compare zero-norm embedding rows")
    similarities = np.sum(model_array * rose_array, axis=1) / (model_norms * rose_norms)
    return {
        "token_ids": ids,
        "cosine_similarity": similarities.tolist(),
        "minimum_cosine": float(similarities.min()) if similarities.size else None,
        "maximum_cosine": float(similarities.max()) if similarities.size else None,
        "mean_cosine": float(similarities.mean()) if similarities.size else None,
    }


def load_rose_embedding_table(path: Path, metadata_path: Path | None) -> tuple[np.memmap, dict[str, Any], Path]:
    """Validate and memory-map a prepared ROSE embedding table."""
    if metadata_path is None:
        metadata_path = path.with_suffix(".json")
    with metadata_path.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)
    if not metadata.get("normalized", False):
        raise ValueError("ROSE metadata says the embedding table is not normalized")
    shape = (int(metadata["vocab_size"]), int(metadata["hidden_size"]))
    dtype = np.dtype(metadata.get("dtype", "float16"))
    expected_size = math.prod(shape) * dtype.itemsize
    actual_size = path.stat().st_size
    if expected_size != actual_size:
        raise ValueError(f"ROSE embedding file has {actual_size} bytes, expected {expected_size} from metadata")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape), metadata, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Hugging Face checkpoint directory")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="One user prompt")
    prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 file containing one user prompt")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument(
        "--chat-template-kwargs",
        default="{}",
        help="JSON passed to chat templates, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda[:N], or npu[:N]")
    parser.add_argument("--load-model", action="store_true", help="Load the causal LM for input embedding inspection")
    parser.add_argument("--generate", action="store_true", help="Run one local Transformers generate call")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--rose-embedding-path", type=Path, default=None)
    parser.add_argument("--rose-embedding-meta-path", type=Path, default=None)
    parser.add_argument(
        "--rose-token-ids",
        default=None,
        help="Comma-separated IDs to inspect in the ROSE table; defaults to prompt IDs",
    )
    return parser.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 18} {title} {'=' * 18}")


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def choose_device(torch: Any, requested: str) -> Any:
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_processor(transformers: Any, model_path: str, trust_remote_code: bool) -> Any | None:
    try:
        return transformers.AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    except Exception as exc:  # Processor is optional for text-only checkpoints.
        print(f"processor: unavailable ({type(exc).__name__}: {exc})")
        return None


def inspect_rose_table(path: Path, metadata_path: Path | None, token_ids: Sequence[int]) -> None:
    section("ROSE semantic embedding")
    table, metadata, metadata_path = load_rose_embedding_table(path, metadata_path)
    print(f"embedding_path: {path}")
    print(f"metadata_path: {metadata_path}")
    print_json(metadata)
    print(f"actual_bytes: {path.stat().st_size}")
    print_json(inspect_embedding_rows(table, token_ids))
    print(
        "note: this table is used by SemanticEntropyScorer after rollout top-k IDs; "
        "it is not fed to the model tokenizer or vLLM."
    )


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.generate:
        args.load_model = True

    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    model_path = str(Path(args.model_path).expanduser())
    chat_template_kwargs = json.loads(args.chat_template_kwargs)
    if not isinstance(chat_template_kwargs, dict):
        raise TypeError("--chat-template-kwargs must decode to a JSON object")

    try:
        import torch
        import transformers
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        return 2

    # Some Ascend environments register torch.npu only after importing torch_npu.
    if args.device == "auto" or args.device.startswith("npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            if args.device.startswith("npu"):
                raise RuntimeError("--device npu requested, but torch_npu cannot be imported") from None

    device = choose_device(torch, args.device)
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.set_device(device)

    section("Environment")
    print_json(
        {
            "python": sys.version,
            "transformers": getattr(transformers, "__version__", "unknown"),
            "torch": getattr(torch, "__version__", "unknown"),
            "device": str(device),
            "model_path": model_path,
        }
    )

    section("Config and tokenizer")
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    processor = load_processor(transformers, model_path, args.trust_remote_code)
    print_json(
        {
            "config_class": type(config).__name__,
            "model_type": getattr(config, "model_type", None),
            "tokenizer_class": type(tokenizer).__name__,
            "processor_class": type(processor).__name__ if processor is not None else None,
            "tokenizer_vocab_size": len(tokenizer),
            "config_vocab_size": getattr(config, "vocab_size", None),
            "special_tokens_map": tokenizer.special_tokens_map,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )

    messages: list[dict[str, str]] = []
    if args.system_prompt is not None:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    section("Input messages")
    print_json(messages)

    section("Chat template")
    template_kwargs = {
        **chat_template_kwargs,
        "tokenize": False,
        "add_generation_prompt": True,
    }
    rendered = tokenizer.apply_chat_template(messages, **template_kwargs)
    print(f"rendered_type: {type(rendered).__name__}")
    print(f"rendered_text: {rendered!r}")
    template_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            **chat_template_kwargs,
        ),
        source="tokenizer.apply_chat_template",
    )
    print(f"template_token_count: {len(template_ids)}")
    print(f"template_token_ids: {template_ids}")
    rollout_prompt_ids = template_ids

    section("Token mapping")
    token_strings = tokenizer.convert_ids_to_tokens(template_ids)
    print_json(
        [
            {"index": i, "id": token_id, "token": token}
            for i, (token_id, token) in enumerate(zip(template_ids, token_strings, strict=False))
        ]
    )

    section("Decode round trip")
    decoded_with_specials = tokenizer.decode(template_ids, skip_special_tokens=False)
    decoded_without_specials = tokenizer.decode(template_ids, skip_special_tokens=True)
    print(f"decoded_with_specials: {decoded_with_specials!r}")
    print(f"decoded_without_specials: {decoded_without_specials!r}")
    reencoded = normalize_token_ids(
        tokenizer(decoded_with_specials, add_special_tokens=False)["input_ids"],
        source="tokenizer(decoded_text)",
    )
    print_json(compare_token_sequences(template_ids, reencoded))

    if processor is not None and hasattr(processor, "apply_chat_template"):
        section("Processor chat template comparison")
        try:
            processor_ids = normalize_token_ids(
                processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                    **chat_template_kwargs,
                ),
                source="processor.apply_chat_template",
            )
            print_json(compare_token_sequences(template_ids, processor_ids))
            print(f"processor_token_ids: {processor_ids}")
        except Exception as exc:
            print(f"processor comparison failed: {type(exc).__name__}: {exc}")

    section("verl Continuous Token prompt")
    try:
        from verl.utils.tokenizer.continuous_token_wiring import create_continuous_token_builder

        builder = create_continuous_token_builder(
            tokenizer,
            hf_model_type=getattr(config, "model_type", None),
            chat_template_kwargs=chat_template_kwargs,
            processor=processor,
        )
        continuous_ids = normalize_token_ids(
            builder.build_initial_tokens(messages),
            source="ContinuousTokenBuilder.build_initial_tokens",
        )
        print(f"builder_class: {type(builder).__name__}")
        print(f"supports_multimodal: {builder.supports_multimodal()}")
        print(f"continuous_token_ids: {continuous_ids}")
        print(f"continuous_decoded: {tokenizer.decode(continuous_ids, skip_special_tokens=False)!r}")
        print("comparison_with_tokenizer_template:")
        print_json(compare_token_sequences(template_ids, continuous_ids))
        rollout_prompt_ids = continuous_ids
    except Exception as exc:
        print(f"Continuous Token inspection failed: {type(exc).__name__}: {exc}")

    if args.rose_embedding_path is not None:
        rose_ids = rollout_prompt_ids
        if args.rose_token_ids:
            rose_ids = [int(value.strip()) for value in args.rose_token_ids.split(",") if value.strip()]
        inspect_rose_table(args.rose_embedding_path, args.rose_embedding_meta_path, rose_ids)

    if not args.load_model:
        print(
            "\nModel load skipped. Add --load-model to inspect model input embeddings; "
            "add --generate for one local generate call."
        )
        return 0

    section("Model load and input embedding")
    dtype = getattr(torch, args.dtype) if args.dtype else None
    model_kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype is not None:
        model_kwargs["dtype"] = dtype
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.to(device)
    model.eval()
    input_embeddings = model.get_input_embeddings()
    input_weight = input_embeddings.weight.detach()
    print_json(
        {
            "model_class": type(model).__name__,
            "model_device": str(input_weight.device),
            "input_embedding_shape": list(input_weight.shape),
            "input_embedding_dtype": str(input_weight.dtype),
            "note": "finite-value checks below cover prompt/generated rows, not the full table",
        }
    )

    if args.rose_embedding_path is not None:
        section("Model embedding vs ROSE embedding")
        rose_table, rose_metadata, _ = load_rose_embedding_table(
            args.rose_embedding_path,
            args.rose_embedding_meta_path,
        )
        if input_weight.shape[0] < len(tokenizer):
            raise ValueError(
                f"model input embedding has {input_weight.shape[0]} rows, fewer than tokenizer size {len(tokenizer)}"
            )
        if rose_table.shape[0] != len(tokenizer):
            print(f"WARNING: ROSE vocab size {rose_table.shape[0]} differs from tokenizer size {len(tokenizer)}")
        if rose_table.shape[1] != input_weight.shape[1]:
            raise ValueError(
                f"ROSE hidden size {rose_table.shape[1]} differs from model input embedding "
                f"hidden size {input_weight.shape[1]}"
            )
        rose_id_tensor = torch.tensor(rose_ids, dtype=torch.long, device=input_weight.device)
        model_rows = input_weight[rose_id_tensor].detach().float().cpu().numpy()
        rose_rows = np.asarray(rose_table[rose_ids], dtype=np.float32)
        print(f"metadata_model_path: {rose_metadata.get('model_path')!r}")
        print_json(compare_selected_embeddings(model_rows, rose_rows, rose_ids))
    print(f"model_input_source: {'verl Continuous Token' if rollout_prompt_ids is not template_ids else 'tokenizer'}")
    print(f"model_input_ids: {rollout_prompt_ids}")
    input_ids = torch.tensor([rollout_prompt_ids], dtype=torch.long, device=input_weight.device)
    with torch.no_grad():
        input_vectors = input_embeddings(input_ids)
    print_json(
        {
            "lookup_shape": list(input_vectors.shape),
            "lookup_dtype": str(input_vectors.dtype),
            "lookup_all_finite": bool(torch.isfinite(input_vectors).all().item()),
            "lookup_min": float(input_vectors.min().item()),
            "lookup_max": float(input_vectors.max().item()),
        }
    )

    if args.generate:
        section("One local Transformers generation")
        with torch.no_grad():
            forward_output = model(input_ids=input_ids, use_cache=False)
        next_logits = forward_output.logits[0, -1].float()
        top_values, top_ids_tensor = torch.topk(next_logits, k=min(10, next_logits.shape[-1]))
        top_ids = [int(value) for value in top_ids_tensor.detach().cpu().tolist()]
        top_logits = [float(value) for value in top_values.detach().cpu().tolist()]
        top_probabilities = torch.softmax(top_values, dim=-1).detach().cpu().tolist()
        print("first_step_logits:")
        print_json(
            {
                "shape": list(forward_output.logits.shape),
                "all_finite": bool(torch.isfinite(next_logits).all().item()),
                "top_candidates": [
                    {
                        "id": token_id,
                        "token": tokenizer.convert_ids_to_tokens(token_id),
                        "decoded": tokenizer.decode([token_id], skip_special_tokens=False),
                        "logit": logit,
                        "probability_within_top10": probability,
                    }
                    for token_id, logit, probability in zip(top_ids, top_logits, top_probabilities, strict=True)
                ],
            }
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
        }
        if args.do_sample:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
        with torch.no_grad():
            generated = model.generate(input_ids=input_ids, **generation_kwargs)
        generated_ids = normalize_token_ids(generated, source="model.generate")
        new_ids = generated_ids[len(rollout_prompt_ids) :]
        print(f"generated_total_ids: {generated_ids}")
        print(f"generated_new_ids: {new_ids}")
        print(f"generated_text: {tokenizer.decode(new_ids, skip_special_tokens=False)!r}")
        print(f"generated_text_without_specials: {tokenizer.decode(new_ids, skip_special_tokens=True)!r}")
        selected_weights = input_weight[new_ids].detach().float().cpu().numpy()
        print_json(inspect_selected_embedding_rows(selected_weights, new_ids, input_weight.shape[0]))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DIAGNOSTIC FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
