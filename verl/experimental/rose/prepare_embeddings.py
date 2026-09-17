import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_embeddings(model_path: str, output_path: str, trust_remote_code: bool = False) -> tuple[Path, Path]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".json")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    embeddings = model.get_input_embeddings().weight.detach().float().cpu()
    vocab_size = len(tokenizer)
    if embeddings.shape[0] < vocab_size:
        raise ValueError(f"embedding rows {embeddings.shape[0]} are fewer than tokenizer vocabulary {vocab_size}")
    embeddings = torch.nn.functional.normalize(embeddings[:vocab_size], p=2, dim=1)
    embeddings.numpy().astype(np.float16).tofile(output)

    metadata = {
        "model_path": str(model_path),
        "vocab_size": vocab_size,
        "hidden_size": int(embeddings.shape[1]),
        "dtype": "float16",
        "normalized": True,
        "embedding_sha256": _sha256(output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a normalized mmap embedding table for ROSE.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    prepare_embeddings(args.model_path, args.output_path, args.trust_remote_code)


if __name__ == "__main__":
    main()
