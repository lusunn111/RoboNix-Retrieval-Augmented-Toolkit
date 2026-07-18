from __future__ import annotations

import dataclasses
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pickle
import time
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import triton
import triton.language as tl

from openpi.models import tokenizer as _tokenizer


_BASE_WEIGHTS_FILENAME = "base_weights.pkl"
_MANIFEST_FILENAME = "manifest.json"
_LANGUAGE_EMBEDS_DIRNAME = "language_embeds"
_LANGUAGE_EMBEDDING_WEIGHT_KEY = "language_embedding_weight"


def _spec_triton_env(name: str, default: str) -> str:
    spec_name = f"SPEC_TRITON_{name}"
    legacy_env_name = f"STAR_TRITON_{name}"
    return os.environ.get(spec_name, os.environ.get(legacy_env_name, default))


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    if not module_path.exists():
        raise FileNotFoundError(f"Missing required module: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spec_runtime_module(module_filename: str, module_name: str) -> ModuleType:
    return _load_module(Path(__file__).with_name(module_filename), module_name)


def _empty_pi0_triton_weights(prompt_len: int) -> dict[str, torch.Tensor]:
    return {
        "vision_patch_embedding_w": torch.zeros(14, 14, 3, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_patch_embedding_b": torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "vision_position_embedding": torch.zeros(256, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_w": torch.zeros(27, 1152, 3 * 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_qkv_b": torch.zeros(27, 3 * 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_w": torch.zeros(27, 1152, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_attn_o_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_w": torch.zeros(27, 1152, 4304, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_up_b": torch.zeros(27, 4304, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_w": torch.zeros(27, 4304, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_ffn_down_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_w": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_attn_norm_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_w": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_pre_ffn_norm_b": torch.zeros(27, 1152, dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_w": torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "vision_final_norm_b": torch.zeros(1152, dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_w": torch.zeros(1152, 2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_multi_modal_projector_b": torch.zeros(2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_qkv_w": torch.zeros(18, 2048, 2560, dtype=torch.bfloat16, device="cpu"),
        "encoder_attn_o_w": torch.zeros(18, 2048, 2048, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_gate_w": torch.zeros(18, 2048, 16384, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_up_w": torch.zeros(18, 2048, 16384, dtype=torch.bfloat16, device="cpu"),
        "encoder_ffn_down_w": torch.zeros(18, 16384, 2048, dtype=torch.bfloat16, device="cpu"),
        "decoder_state_in_proj_w": torch.zeros(32, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_state_in_proj_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_w": torch.zeros(32, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_in_proj_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_time_mlp_in_w": torch.zeros(2048, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_time_mlp_in_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_fused_in_proj_w": torch.zeros(32, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_fused_time_biases": torch.zeros(10, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_mlp_w": torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_mlp_b": torch.zeros(1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_qkv_w": torch.zeros(18, 1024, 2560, dtype=torch.bfloat16, device="cpu"),
        "decoder_attn_o_w": torch.zeros(18, 2048, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_gate_w": torch.zeros(18, 1024, 4096, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_up_w": torch.zeros(18, 1024, 4096, dtype=torch.bfloat16, device="cpu"),
        "decoder_ffn_down_w": torch.zeros(18, 4096, 1024, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_fused_out_proj_w": torch.zeros(1024, 32, dtype=torch.bfloat16, device="cpu"),
        "decoder_action_fused_out_proj_b": torch.zeros(32, dtype=torch.bfloat16, device="cpu"),
        "language_embeds": torch.zeros(prompt_len, 2048, dtype=torch.bfloat16, device="cpu"),
}


def _base_weights_have_exact_verify_keys(weights: Mapping[str, Any]) -> bool:
    required = (
        "decoder_action_in_proj_w",
        "decoder_action_in_proj_b",
        "decoder_action_time_mlp_in_w",
        "decoder_action_time_mlp_in_b",
    )
    return all(key in weights for key in required)


def _embedding_rows(embedding_weight: np.ndarray | torch.Tensor, token_ids: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(token_ids, np.ndarray):
        token_ids = torch.from_numpy(token_ids.astype(np.int64, copy=False))
    token_ids = token_ids.to(dtype=torch.long, device="cpu")
    if isinstance(embedding_weight, torch.Tensor):
        return embedding_weight.detach().cpu()[token_ids].to(torch.float32)
    return torch.from_numpy(np.asarray(embedding_weight)[token_ids.numpy()]).to(torch.float32)


def _prepare_language_embeds_local(prompt: str, embedding_weight: np.ndarray | torch.Tensor) -> torch.Tensor:
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=48)
    prompt_tokens, prompt_mask = tokenizer.tokenize(prompt)
    valid_tokens = np.asarray(prompt_tokens, dtype=np.int64)[np.asarray(prompt_mask, dtype=bool)]
    embeds = _embedding_rows(embedding_weight, valid_tokens)
    embeds.mul_(math.sqrt(float(embeds.shape[-1])))
    return embeds.to(torch.bfloat16).cpu()


def _prepare_language_embeds_hf(
    prompt: str,
    embedding_weight: np.ndarray | torch.Tensor,
    *,
    hf_tokenizer_id: str,
    hf_endpoint: str,
) -> torch.Tensor:
    previous_endpoint = os.environ.get("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = str(hf_endpoint)
    try:
        auto_tokenizer = importlib.import_module("transformers").AutoTokenizer
        tokenizer = auto_tokenizer.from_pretrained(hf_tokenizer_id)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF tokenizer '{hf_tokenizer_id}' via HF_ENDPOINT={hf_endpoint}. "
            "This repo is gated; make sure a valid HF token is configured."
        ) from exc
    finally:
        if previous_endpoint is None:
            os.environ.pop("HF_ENDPOINT", None)
        else:
            os.environ["HF_ENDPOINT"] = previous_endpoint

    prompt_text = [prompt.strip().replace("_", " ") + "\n"]
    valid_tokens = tokenizer(
        prompt_text,
        max_length=48,
        return_tensors="pt",
    )["input_ids"].squeeze(0).to(dtype=torch.int64)
    embeds = _embedding_rows(embedding_weight, valid_tokens)
    embeds.mul_(math.sqrt(float(embeds.shape[-1])))
    return embeds.to(torch.bfloat16).cpu()


def prepare_language_embeds(
    *,
    prompt: str,
    embedding_weight: np.ndarray | torch.Tensor,
    tokenizer_source: str,
    hf_endpoint: str,
    hf_tokenizer_id: str,
) -> torch.Tensor:
    if tokenizer_source == "local":
        return _prepare_language_embeds_local(prompt, embedding_weight)
    if tokenizer_source == "hf":
        return _prepare_language_embeds_hf(
            prompt,
            embedding_weight,
            hf_tokenizer_id=hf_tokenizer_id,
            hf_endpoint=hf_endpoint,
        )
    if tokenizer_source == "auto":
        try:
            return _prepare_language_embeds_hf(
                prompt,
                embedding_weight,
                hf_tokenizer_id=hf_tokenizer_id,
                hf_endpoint=hf_endpoint,
            )
        except Exception:
            return _prepare_language_embeds_local(prompt, embedding_weight)
    raise ValueError(f"Unsupported tokenizer_source: {tokenizer_source}")


def _base_weights_from_dump(*, convert_module: ModuleType, dump_weights: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    weights = _empty_pi0_triton_weights(1)
    convert_module.convert_weights(weights, dump_weights)
    weights.pop("language_embeds", None)
    weights[_LANGUAGE_EMBEDDING_WEIGHT_KEY] = torch.as_tensor(
        dump_weights["PaliGemma"]["llm"]["embedder"]["input_embedding"]["value"],
        dtype=torch.bfloat16,
        device="cpu",
    ).contiguous()
    return weights


def _prompt_cache_key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _base_weights_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / _BASE_WEIGHTS_FILENAME


def _manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / _MANIFEST_FILENAME


def _language_embed_rel_path(prompt: str) -> Path:
    return Path(_LANGUAGE_EMBEDS_DIRNAME) / f"{_prompt_cache_key(prompt)}.pt"


def build_prompt_cache(
    *,
    jax_checkpoint_dir: str | Path,
    cache_dir: str | Path,
    prompts: Iterable[str],
    tokenizer_source: str = "auto",
    hf_endpoint: str = "https://hf-mirror.com",
    hf_tokenizer_id: str = "google/paligemma-3b-pt-224",
    force_base: bool = False,
    force_prompts: bool = False,
) -> dict[str, str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_weights_path = _base_weights_path(cache_dir)
    manifest_path = _manifest_path(cache_dir)
    embed_dir = cache_dir / _LANGUAGE_EMBEDS_DIRNAME
    embed_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"prompts": {}}
    prompt_manifest = manifest.setdefault("prompts", {})

    unique_prompts = list(dict.fromkeys(str(prompt) for prompt in prompts))
    missing_prompts: list[str] = []
    for prompt in unique_prompts:
        entry = prompt_manifest.get(prompt)
        embed_missing = True
        if entry is not None:
            embed_path = cache_dir / str(entry.get("embed_path", ""))
            embed_missing = not embed_path.exists()
        if force_prompts or entry is None or embed_missing:
            missing_prompts.append(prompt)

    need_base = force_base or not base_weights_path.exists()
    if not need_base and base_weights_path.exists():
        try:
            with base_weights_path.open("rb") as handle:
                existing_base = pickle.load(handle)
            need_base = not _base_weights_have_exact_verify_keys(existing_base)
        except Exception:
            need_base = True
    if need_base or missing_prompts:
        convert_module = _spec_runtime_module(
            "convert_for_triton.py",
            "spec_convert_from_jax_for_prompt_cache",
        )
        dump_weights = convert_module.load_jax_weights(str(jax_checkpoint_dir))
        if need_base:
            base_weights = _base_weights_from_dump(convert_module=convert_module, dump_weights=dump_weights)
            with base_weights_path.open("wb") as handle:
                pickle.dump(base_weights, handle)
        if missing_prompts:
            embedding_weight = dump_weights["PaliGemma"]["llm"]["embedder"]["input_embedding"]["value"]
            for prompt in missing_prompts:
                language_embeds = prepare_language_embeds(
                    prompt=prompt,
                    embedding_weight=embedding_weight,
                    tokenizer_source=tokenizer_source,
                    hf_endpoint=hf_endpoint,
                    hf_tokenizer_id=hf_tokenizer_id,
                )
                rel_embed_path = _language_embed_rel_path(prompt)
                abs_embed_path = cache_dir / rel_embed_path
                abs_embed_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(language_embeds, abs_embed_path)
                prompt_manifest[prompt] = {
                    "embed_path": str(rel_embed_path),
                    "prompt_len": int(language_embeds.shape[0]),
                }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "base_weights_path": str(base_weights_path),
        "manifest_path": str(manifest_path),
    }


def build_prompt_cache_from_base(
    *,
    base_weights_path: str | Path,
    cache_dir: str | Path,
    prompts: Iterable[str],
    tokenizer_source: str = "auto",
    hf_endpoint: str = "https://hf-mirror.com",
    hf_tokenizer_id: str = "google/paligemma-3b-pt-224",
    force_prompts: bool = False,
) -> dict[str, str]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(cache_dir)
    embed_dir = cache_dir / _LANGUAGE_EMBEDS_DIRNAME
    embed_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"prompts": {}}
    prompt_manifest = manifest.setdefault("prompts", {})

    unique_prompts = list(dict.fromkeys(str(prompt) for prompt in prompts))
    missing_prompts: list[str] = []
    for prompt in unique_prompts:
        entry = prompt_manifest.get(prompt)
        embed_missing = True
        if entry is not None:
            embed_path = cache_dir / str(entry.get("embed_path", ""))
            embed_missing = not embed_path.exists()
        if force_prompts or entry is None or embed_missing:
            missing_prompts.append(prompt)

    if missing_prompts:
        with Path(base_weights_path).open("rb") as handle:
            base_weights = pickle.load(handle)
        embedding_weight = base_weights.get(_LANGUAGE_EMBEDDING_WEIGHT_KEY)
        if embedding_weight is None:
            raise KeyError(
                f"{Path(base_weights_path)} does not contain {_LANGUAGE_EMBEDDING_WEIGHT_KEY!r}; "
                "re-convert the base artifact with convert_for_triton.py."
            )
        for prompt in missing_prompts:
            language_embeds = prepare_language_embeds(
                prompt=prompt,
                embedding_weight=embedding_weight,
                tokenizer_source=tokenizer_source,
                hf_endpoint=hf_endpoint,
                hf_tokenizer_id=hf_tokenizer_id,
            )
            rel_embed_path = _language_embed_rel_path(prompt)
            abs_embed_path = cache_dir / rel_embed_path
            abs_embed_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(language_embeds, abs_embed_path)
            prompt_manifest[prompt] = {
                "embed_path": str(rel_embed_path),
                "prompt_len": int(language_embeds.shape[0]),
            }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "base_weights_path": str(base_weights_path),
        "manifest_path": str(manifest_path),
    }


def convert_jax_checkpoint(
    *,
    jax_checkpoint_dir: str | Path,
    output_path: str | Path,
    prompt: str,
    tokenizer_source: str = "auto",
    hf_endpoint: str = "https://hf-mirror.com",
    hf_tokenizer_id: str = "google/paligemma-3b-pt-224",
) -> Path:
    convert_module = _spec_runtime_module(
        "convert_for_triton.py",
        "spec_convert_from_jax",
    )
    dump_weights = convert_module.load_jax_weights(str(jax_checkpoint_dir))
    embedding_weight = dump_weights["PaliGemma"]["llm"]["embedder"]["input_embedding"]["value"]
    language_embeds = prepare_language_embeds(
        prompt=prompt,
        embedding_weight=embedding_weight,
        tokenizer_source=tokenizer_source,
        hf_endpoint=hf_endpoint,
        hf_tokenizer_id=hf_tokenizer_id,
    )

    weights = _empty_pi0_triton_weights(int(language_embeds.shape[0]))
    convert_module.convert_weights(weights, dump_weights)
    weights["language_embeds"].copy_(language_embeds)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(weights, handle)
    return output_path


def ensure_triton_checkpoint(
    *,
    jax_checkpoint_dir: str | Path | None,
    output_path: str | Path,
    prompt: str,
    force: bool = False,
    tokenizer_source: str = "auto",
    hf_endpoint: str = "https://hf-mirror.com",
    hf_tokenizer_id: str = "google/paligemma-3b-pt-224",
) -> Path:
    output_path = Path(output_path)
    if force or not output_path.exists():
        if jax_checkpoint_dir is None:
            raise ValueError("triton_jax_checkpoint_dir is required when creating a Triton checkpoint.")
        return convert_jax_checkpoint(
            jax_checkpoint_dir=jax_checkpoint_dir,
            output_path=output_path,
            prompt=prompt,
            tokenizer_source=tokenizer_source,
            hf_endpoint=hf_endpoint,
            hf_tokenizer_id=hf_tokenizer_id,
        )
    return output_path


def _load_draft_checkpoint_payload(draft_checkpoint_path: str | Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(Path(draft_checkpoint_path), map_location="cpu")
    meta: dict[str, Any] = {}
    sd: Any
    if isinstance(ckpt, dict) and "draft_head" in ckpt:
        meta = dict(ckpt.get("meta", {}) or {})
        sd = ckpt.get("draft_head", {})
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("draft checkpoint must be a state_dict or a dict with key `draft_head`")
    if not isinstance(sd, dict):
        raise ValueError("draft_head state_dict missing or invalid")
    return meta, {str(k): v.detach().cpu() for k, v in sd.items() if isinstance(v, torch.Tensor)}


def convert_spec_draft_checkpoint(
    *,
    draft_checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    meta, state_dict = _load_draft_checkpoint_payload(draft_checkpoint_path)
    required_keys = {
        "_state_token.weight",
        "_state_token.bias",
        "_action_queries.weight",
        "_gemma_block.self_attn.q_proj.weight",
        "_gemma_block.self_attn.k_proj.weight",
        "_gemma_block.self_attn.v_proj.weight",
        "_gemma_block.self_attn.o_proj.weight",
        "_gemma_block.mlp.gate_proj.weight",
        "_gemma_block.mlp.up_proj.weight",
        "_gemma_block.mlp.down_proj.weight",
        "_gemma_block.input_layernorm.weight",
        "_gemma_block.post_attention_layernorm.weight",
        "_action_head.weight",
        "_action_head.bias",
    }
    missing = sorted(required_keys.difference(state_dict))
    if missing:
        raise KeyError(f"draft checkpoint missing required tensors: {missing}")

    artifact = {
        "meta": dict(meta),
        "draft_state_in_proj_w": state_dict["_state_token.weight"].contiguous(),
        "draft_state_in_proj_b": state_dict["_state_token.bias"].contiguous(),
        "draft_action_queries": state_dict["_action_queries.weight"].contiguous(),
        "draft_qkv_w": torch.cat(
            [
                state_dict["_gemma_block.self_attn.q_proj.weight"],
                state_dict["_gemma_block.self_attn.k_proj.weight"],
                state_dict["_gemma_block.self_attn.v_proj.weight"],
            ],
            dim=0,
        ).contiguous(),
        "draft_attn_o_w": state_dict["_gemma_block.self_attn.o_proj.weight"].contiguous(),
        "draft_ffn_gate_w": state_dict["_gemma_block.mlp.gate_proj.weight"].contiguous(),
        "draft_ffn_up_w": state_dict["_gemma_block.mlp.up_proj.weight"].contiguous(),
        "draft_ffn_down_w": state_dict["_gemma_block.mlp.down_proj.weight"].contiguous(),
        "draft_input_layernorm_w": state_dict["_gemma_block.input_layernorm.weight"].contiguous(),
        "draft_post_attention_layernorm_w": state_dict["_gemma_block.post_attention_layernorm.weight"].contiguous(),
        "draft_action_head_w": state_dict["_action_head.weight"].contiguous(),
        "draft_action_head_b": state_dict["_action_head.bias"].contiguous(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    return output_path


def ensure_spec_draft_checkpoint(
    *,
    draft_checkpoint_path: str | Path,
    output_path: str | Path,
    force: bool = False,
) -> Path:
    output_path = Path(output_path)
    if force or not output_path.exists():
        return convert_spec_draft_checkpoint(
            draft_checkpoint_path=draft_checkpoint_path,
            output_path=output_path,
        )
    return output_path


def create_pi0_inference(
    *,
    checkpoint: dict[str, torch.Tensor],
    num_views: int,
    chunk_size: int,
):
    infer_module = _spec_runtime_module(
        "pi0_infer.py",
        "spec_pi0_infer",
    )
    return _StagedPi0Runtime(
        infer_module=infer_module,
        checkpoint=checkpoint,
        num_views=int(num_views),
        chunk_size=int(chunk_size),
    )


def create_pi0_spec_inference(
    *,
    checkpoint: dict[str, torch.Tensor],
    draft_checkpoint: Mapping[str, Any],
    num_views: int,
    chunk_size: int,
):
    infer_module = _spec_runtime_module(
        "pi0_spec_infer.py",
        "spec_pi0_spec_infer",
    )
    return infer_module.Pi0SpecInference(
        checkpoint=checkpoint,
        draft_checkpoint=draft_checkpoint,
        num_views=int(num_views),
        chunk_size=int(chunk_size),
    )


def load_pi0_inference(
    *,
    checkpoint_path: str | Path,
    num_views: int,
    chunk_size: int,
):
    with Path(checkpoint_path).open("rb") as handle:
        checkpoint = pickle.load(handle)
    return create_pi0_inference(
        checkpoint=checkpoint,
        num_views=num_views,
        chunk_size=chunk_size,
    )


def normalize_triton_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0 * 2.0 - 1.0


@triton.jit
def _normalize_uint8_images_kernel(src_ptr, dst_ptr, total_values: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_values
    values = tl.load(src_ptr + offsets, mask=mask, other=0).to(tl.float32)
    normalized = values * (2.0 / 255.0) - 1.0
    tl.store(dst_ptr + offsets, normalized, mask=mask)


@dataclasses.dataclass
class _TritonInputPrepareBuffers:
    key: tuple[Any, ...]
    image_u8: torch.Tensor
    images: torch.Tensor
    state: torch.Tensor
    noise: torch.Tensor
    noise_fp32: torch.Tensor


class TritonInputPreparer:
    """Caches GPU staging buffers for the transformed pi0 policy inputs."""

    def __init__(self, *, device: str | torch.device) -> None:
        self._device = torch.device(device)
        self._buffers: _TritonInputPrepareBuffers | None = None

    @staticmethod
    def _active_uint8_images(transformed: Mapping[str, Any]) -> list[tuple[str, np.ndarray]] | None:
        active: list[tuple[str, np.ndarray]] = []
        for name, image in transformed["image"].items():
            if not bool(transformed["image_mask"][name]):
                continue
            image_np = np.asarray(image)
            if image_np.dtype != np.uint8 or image_np.ndim != 3:
                return None
            active.append((str(name), np.ascontiguousarray(image_np)))
        if not active:
            return []
        first_shape = tuple(active[0][1].shape)
        if any(tuple(image_np.shape) != first_shape for _, image_np in active):
            return None
        return active

    def _ensure_buffers(
        self,
        *,
        active_images: list[tuple[str, np.ndarray]],
        state_shape: tuple[int, ...],
        action_horizon: int,
        action_dim: int,
    ) -> _TritonInputPrepareBuffers:
        image_shape = tuple(active_images[0][1].shape)
        key = (
            str(self._device),
            tuple(name for name, _ in active_images),
            len(active_images),
            image_shape,
            tuple(int(x) for x in state_shape),
            int(action_horizon),
            int(action_dim),
        )
        if self._buffers is None or self._buffers.key != key:
            image_batch_shape = (len(active_images), *image_shape)
            self._buffers = _TritonInputPrepareBuffers(
                key=key,
                image_u8=torch.empty(image_batch_shape, device=self._device, dtype=torch.uint8),
                images=torch.empty(image_batch_shape, device=self._device, dtype=torch.bfloat16),
                state=torch.empty(state_shape, device=self._device, dtype=torch.bfloat16),
                noise=torch.empty((int(action_horizon), int(action_dim)), device=self._device, dtype=torch.bfloat16),
                noise_fp32=torch.empty((int(action_horizon), int(action_dim)), device=self._device, dtype=torch.float32),
            )
        return self._buffers

    def prepare(
        self,
        *,
        transformed: Mapping[str, Any],
        action_horizon: int,
        action_dim: int,
        noise: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self._device.type != "cuda" or not torch.cuda.is_available():
            return None
        active_images = self._active_uint8_images(transformed)
        if active_images is None:
            return None
        if not active_images:
            raise ValueError("No active camera views found after applying the pi0_libero input transforms.")

        state_np = np.ascontiguousarray(np.asarray(transformed["state"], dtype=np.float32))
        buffers = self._ensure_buffers(
            active_images=active_images,
            state_shape=tuple(state_np.shape),
            action_horizon=int(action_horizon),
            action_dim=int(action_dim),
        )
        for image_idx, (_, image_np) in enumerate(active_images):
            buffers.image_u8[image_idx].copy_(torch.from_numpy(image_np), non_blocking=True)

        total_values = int(buffers.image_u8.numel())
        _normalize_uint8_images_kernel[(triton.cdiv(total_values, 256),)](
            buffers.image_u8,
            buffers.images,
            total_values=total_values,
            BLOCK_SIZE=256,
        )
        buffers.state.copy_(torch.from_numpy(state_np), non_blocking=True)

        if noise is None:
            buffers.noise_fp32.normal_()
            buffers.noise.copy_(buffers.noise_fp32)
        elif isinstance(noise, torch.Tensor):
            noise_tensor = noise
            if noise_tensor.ndim == 3 and int(noise_tensor.shape[0]) == 1:
                noise_tensor = noise_tensor[0]
            if tuple(noise_tensor.shape) != tuple(buffers.noise.shape):
                raise ValueError(f"noise shape={tuple(noise_tensor.shape)} must match {tuple(buffers.noise.shape)}")
            buffers.noise.copy_(noise_tensor, non_blocking=True)
        else:
            noise_np = np.ascontiguousarray(np.asarray(noise, dtype=np.float32))
            if noise_np.ndim == 3 and int(noise_np.shape[0]) == 1:
                noise_np = np.ascontiguousarray(noise_np[0])
            if tuple(noise_np.shape) != tuple(buffers.noise.shape):
                raise ValueError(f"noise shape={tuple(noise_np.shape)} must match {tuple(buffers.noise.shape)}")
            buffers.noise.copy_(torch.from_numpy(noise_np), non_blocking=True)

        return buffers.images, buffers.state, buffers.noise


_INPUT_PREPARER_BY_DEVICE: dict[str, TritonInputPreparer] = {}


def _default_input_preparer(device: torch.device) -> TritonInputPreparer:
    key = str(device)
    preparer = _INPUT_PREPARER_BY_DEVICE.get(key)
    if preparer is None:
        preparer = TritonInputPreparer(device=device)
        _INPUT_PREPARER_BY_DEVICE[key] = preparer
    return preparer


def _prepare_triton_inputs_from_transformed_slow(
    *,
    transformed: Mapping[str, Any],
    device: str | torch.device,
    action_horizon: int,
    action_dim: int,
    noise: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    image_tensors: list[torch.Tensor] = []
    for name, image in transformed["image"].items():
        if bool(transformed["image_mask"][name]):
            image_np = normalize_triton_image(np.asarray(image))
            image_tensors.append(torch.from_numpy(image_np).to(device=device, dtype=torch.bfloat16))
    if not image_tensors:
        raise ValueError("No active camera views found after applying the pi0_libero input transforms.")

    state = torch.from_numpy(np.asarray(transformed["state"], dtype=np.float32)).to(device=device, dtype=torch.bfloat16)
    if noise is None:
        noise_tensor = torch.randn((int(action_horizon), int(action_dim)), device=device, dtype=torch.float32)
    elif isinstance(noise, torch.Tensor):
        noise_tensor = noise.to(device=device, dtype=torch.float32)
    else:
        noise_tensor = torch.from_numpy(np.asarray(noise, dtype=np.float32)).to(device=device)
    if noise_tensor.ndim == 3 and int(noise_tensor.shape[0]) == 1:
        noise_tensor = noise_tensor[0]
    return (
        torch.stack(image_tensors, dim=0).contiguous(),
        state.contiguous(),
        noise_tensor.to(dtype=torch.bfloat16).contiguous(),
    )


def prepare_triton_inputs_from_transformed(
    *,
    transformed: Mapping[str, Any],
    device: str | torch.device,
    action_horizon: int,
    action_dim: int,
    noise: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    if _spec_triton_env("INPUT_PREPARE_FAST", "1") != "0":
        prepared = _default_input_preparer(device).prepare(
            transformed=transformed,
            action_horizon=int(action_horizon),
            action_dim=int(action_dim),
            noise=noise,
        )
        if prepared is not None:
            return prepared
    return _prepare_triton_inputs_from_transformed_slow(
        transformed=transformed,
        device=device,
        action_horizon=action_horizon,
        action_dim=action_dim,
        noise=noise,
    )


@dataclasses.dataclass(frozen=True)
class TritonPreparedObservation:
    """Prepared tensors for a staged Triton policy call.

    This is intentionally lightweight today: it defines the boundary between
    observation preparation and runtime execution without changing the current
    full-path kernel implementation. Future draft/verify stages can extend this
    contract without changing the server entrypoint again.
    """

    images: torch.Tensor
    state: torch.Tensor
    tokenized_prompt: torch.Tensor | None = None
    tokenized_prompt_mask: torch.Tensor | None = None


_RTCACHE_COMPOSITE_DEFAULTS = {
    "libero_goal": {
        "threshold": 0.143210,
        "disp_min": 0.000009,
        "disp_max": 0.139051,
        "radius_min": 0.000001,
        "radius_max": 0.016873,
    },
    "libero_spatial": {
        "threshold": 0.217119,
        "disp_min": 0.000027,
        "disp_max": 0.128629,
        "radius_min": 0.000019,
        "radius_max": 0.015654,
    },
    "libero_object": {
        "threshold": 0.188199,
        "disp_min": 0.000098,
        "disp_max": 0.116458,
        "radius_min": 0.000010,
        "radius_max": 0.014151,
    },
    "libero_10": {
        "threshold": 0.400000,
        "disp_min": 0.000008,
        "disp_max": 0.102298,
        "radius_min": 0.000001,
        "radius_max": 0.012479,
    },
}


def _rtcache_suite_key(suite_name: str | None) -> str:
    suite = str(suite_name or "").strip()
    if not suite:
        return "libero_goal"
    if suite.startswith("libero_"):
        return suite
    return f"libero_{suite}"


def _rtcache_normalize(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(float(value)):
        return float("nan")
    denom = max(float(hi) - float(lo), 1e-12)
    return float(np.clip((float(value) - float(lo)) / denom, 0.0, 1.0))


def _rtcache_circle_radius(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or int(pts.shape[0]) < 3 or int(pts.shape[1]) < 2:
        return float("nan")
    pts = pts[:, :3] if int(pts.shape[1]) >= 3 else pts
    centered = pts - pts.mean(axis=0, keepdims=True)
    if float(np.linalg.norm(centered)) < 1e-12:
        return 0.0
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        axes = vh[:2].T if int(vh.shape[0]) >= 2 else np.eye(int(centered.shape[1]), 2)
        xy = centered @ axes
        a = np.column_stack([2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(int(xy.shape[0]))])
        b = xy[:, 0] ** 2 + xy[:, 1] ** 2
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
        radius2 = float(sol[2] + sol[0] ** 2 + sol[1] ** 2)
        return float(math.sqrt(max(radius2, 0.0)))
    except Exception:
        return float("nan")


class TritonRuntimeSession:
    def __init__(self, *, prompt: str, runtime: Any) -> None:
        self._prompt = str(prompt)
        self._runtime = runtime

    @property
    def prompt(self) -> str:
        return self._prompt

    def prepare_observation(
        self,
        *,
        images: torch.Tensor,
        state: torch.Tensor,
        tokenized_prompt: torch.Tensor | None = None,
        tokenized_prompt_mask: torch.Tensor | None = None,
    ) -> TritonPreparedObservation:
        return TritonPreparedObservation(
            images=images.contiguous(),
            state=state.contiguous(),
            tokenized_prompt=None if tokenized_prompt is None else tokenized_prompt.contiguous(),
            tokenized_prompt_mask=None if tokenized_prompt_mask is None else tokenized_prompt_mask.contiguous(),
        )

    def run_full(self, *, prepared: TritonPreparedObservation, noise: torch.Tensor) -> torch.Tensor:
        return self._runtime.forward(prepared.images, prepared.state, noise)

    def run_full_with_timing(
        self,
        *,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if hasattr(self._runtime, "run_full_with_timing"):
            return self._runtime.run_full_with_timing(prepared.images, prepared.state, noise)
        return self.run_full(prepared=prepared, noise=noise), {}


class SpecTritonRuntimeSession(TritonRuntimeSession):
    def __init__(self, *, prompt: str, runtime: Any, draft_runtime: Any) -> None:
        super().__init__(prompt=prompt, runtime=runtime)
        self.draft_runtime = draft_runtime

    def capture_full_cache_snapshot(self):
        if not hasattr(self.draft_runtime, "capture_full_cache_snapshot"):
            raise AttributeError("Spec Triton runtime does not expose cache snapshot capture.")
        return self.draft_runtime.capture_full_cache_snapshot(self._runtime)

    def run_draft_with_timing(
        self,
        *,
        prepared: TritonPreparedObservation,
    ):
        if hasattr(self.draft_runtime, "run_draft_with_timing"):
            try:
                return self.draft_runtime.run_draft_with_timing(
                    prepared.images,
                    prepared.state,
                    full_runtime=self._runtime,
                )
            except TypeError:
                return self.draft_runtime.run_draft_with_timing(prepared.images, prepared.state)
        if hasattr(self.draft_runtime, "run_draft"):
            try:
                return (
                    self.draft_runtime.run_draft(
                        prepared.images,
                        prepared.state,
                        full_runtime=self._runtime,
                    ),
                    {},
                )
            except TypeError:
                return self.draft_runtime.run_draft(prepared.images, prepared.state), {}
        raise AttributeError("Spec Triton runtime does not expose draft execution.")

    def run_verify_with_timing(
        self,
        *,
        cache_snapshot: Any,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: tuple[float, ...],
    ):
        if hasattr(self.draft_runtime, "run_verify_with_timing"):
            try:
                return self.draft_runtime.run_verify_with_timing(
                    cache_snapshot,
                    prepared.images,
                    prepared.state,
                    noise,
                    x0_draft,
                    t_list,
                    full_runtime=self._runtime,
                )
            except TypeError:
                return self.draft_runtime.run_verify_with_timing(
                    cache_snapshot,
                    prepared.images,
                    prepared.state,
                    noise,
                    x0_draft,
                    t_list,
                )
        if hasattr(self.draft_runtime, "run_verify"):
            try:
                return (
                    self.draft_runtime.run_verify(
                        cache_snapshot,
                        prepared.images,
                        prepared.state,
                        noise,
                        x0_draft,
                        t_list,
                        full_runtime=self._runtime,
                    ),
                    {},
                )
            except TypeError:
                return self.draft_runtime.run_verify(
                    cache_snapshot,
                    prepared.images,
                    prepared.state,
                    noise,
                    x0_draft,
                    t_list,
                ), {}
        raise AttributeError("Spec Triton runtime does not expose verify execution.")

    def run_verify_semantics_with_timing(
        self,
        *,
        cache_snapshot: Any,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: tuple[float, ...],
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
    ):
        if not hasattr(self.draft_runtime, "run_verify_semantics_with_timing"):
            raise AttributeError("Spec Triton runtime does not expose verify semantics execution.")
        try:
            return self.draft_runtime.run_verify_semantics_with_timing(
                cache_snapshot,
                prepared.images,
                prepared.state,
                noise,
                x0_draft,
                t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
                full_runtime=self._runtime,
            )
        except TypeError:
            return self.draft_runtime.run_verify_semantics_with_timing(
                cache_snapshot,
                prepared.images,
                prepared.state,
                noise,
                x0_draft,
                t_list,
                tau_radius=tau_radius,
                dist_dims=dist_dims,
                max_exec_steps=max_exec_steps,
                last_gripper=last_gripper,
                gripper_switch_threshold=gripper_switch_threshold,
                enable_gripper_verify=enable_gripper_verify,
                enable_gripper_post_verify=enable_gripper_post_verify,
            )


class CompiledSpecVerifyRuntime:
    def __init__(self, *, spec_model: Any, device: str | torch.device) -> None:
        self._spec_model = spec_model
        self._device = torch.device(device)
        self._last_cache_snapshot = None
        self._prepare_prefix = self._prepare_prefix_impl

    def _time_region(self, fn) -> tuple[Any, float]:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        t0 = time.perf_counter()
        out = fn()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        return out, (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _as_action_batch(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            return actions.unsqueeze(0)
        return actions

    @staticmethod
    def _as_state_batch(state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            return state.unsqueeze(0)
        return state

    @staticmethod
    def _image_view_count(images: torch.Tensor) -> int:
        if images.ndim == 4:
            return int(images.shape[0])
        if images.ndim == 5 and int(images.shape[0]) == 1:
            return int(images.shape[1])
        raise ValueError(f"compiled Spec encoder currently supports batch_size=1, got images shape={tuple(images.shape)}")

    @staticmethod
    def _retrieval_vector_from_prefix(prefix_embs: torch.Tensor, *, view_count: int) -> torch.Tensor | None:
        if int(view_count) < 2:
            return None
        if prefix_embs.ndim != 3 or int(prefix_embs.shape[0]) != 1:
            return None
        tokens_per_view = 256
        required = int(view_count) * tokens_per_view
        if int(prefix_embs.shape[1]) < required:
            return None
        first_view = prefix_embs[:, :tokens_per_view, :].to(dtype=torch.float32).mean(dim=1)
        second_view = prefix_embs[:, tokens_per_view : 2 * tokens_per_view, :].to(dtype=torch.float32).mean(dim=1)
        return torch.cat([first_view, second_view], dim=1).contiguous()

    def _prepare_prefix_impl(
        self,
        images: torch.Tensor,
        tokenized_prompt: torch.Tensor,
        tokenized_prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim == 4:
            batch_size = 1
            view_count = int(images.shape[0])
            image_batch = images.permute(0, 3, 1, 2).contiguous()
        elif images.ndim == 5 and int(images.shape[0]) == 1:
            batch_size = 1
            view_count = int(images.shape[1])
            image_batch = images[0].permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(
                f"compiled Spec encoder currently supports batch_size=1, got images shape={tuple(images.shape)}"
            )

        image_batch = image_batch.to(device=self._device)
        tokenized_prompt = tokenized_prompt.to(device=self._device, dtype=torch.long)
        tokenized_prompt_mask = tokenized_prompt_mask.to(device=self._device, dtype=torch.bool)
        if tokenized_prompt.ndim == 1:
            tokenized_prompt = tokenized_prompt.unsqueeze(0)
        if tokenized_prompt_mask.ndim == 1:
            tokenized_prompt_mask = tokenized_prompt_mask.unsqueeze(0)
        image_masks = tuple(
            torch.ones((batch_size,), device=self._device, dtype=torch.bool)
            for _ in range(view_count)
        )
        images = tuple(image_batch[view_idx : view_idx + 1] for view_idx in range(view_count))
        return self._spec_model._encoder_stage(
            images,
            image_masks,
            tokenized_prompt,
            tokenized_prompt_mask,
        )

    def prepare_prefix_with_timing(
        self,
        *,
        prepared: TritonPreparedObservation,
        full_runtime: Any,
        prefix_runtime: Any | None = None,
    ):
        del prefix_runtime
        del full_runtime
        if prepared.tokenized_prompt is None or prepared.tokenized_prompt_mask is None:
            raise ValueError("compiled Spec encoder requires tokenized_prompt and tokenized_prompt_mask")

        def _run():
            prefix_embs, prefix_pad_masks, prefix_att_masks = self._prepare_prefix(
                prepared.images,
                prepared.tokenized_prompt,
                prepared.tokenized_prompt_mask,
            )
            retrieval_vector = self._retrieval_vector_from_prefix(
                prefix_embs,
                view_count=self._image_view_count(prepared.images),
            )
            return types.SimpleNamespace(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_att_masks=prefix_att_masks,
                retrieval_vector=retrieval_vector,
            )

        return self._time_region(_run)

    @staticmethod
    def _past_key_values_to_snapshot(
        past_key_values: Any,
        *,
        prefix_len: int,
        prefix_pad_masks: torch.Tensor,
    ):
        try:
            from openpi.models_pytorch.spec_pi0_pytorch import _DynamicCache
        except Exception:
            _DynamicCache = None  # type: ignore[assignment]

        if _DynamicCache is not None and isinstance(past_key_values, _DynamicCache):
            legacy_items = [past_key_values[layer_idx] for layer_idx in range(len(past_key_values))]
        else:
            legacy_items = list(past_key_values)

        encoder_k: list[torch.Tensor] = []
        encoder_v: list[torch.Tensor] = []
        for key, value in legacy_items:
            if key.ndim == 4:
                key = key[0, 0, :prefix_len]
                value = value[0, 0, :prefix_len]
            elif key.ndim == 3:
                key = key[0, :prefix_len]
                value = value[0, :prefix_len]
            else:
                key = key[:prefix_len]
                value = value[:prefix_len]
            encoder_k.append(key.detach().clone())
            encoder_v.append(value.detach().clone())
        return types.SimpleNamespace(
            encoder_seq_len=int(prefix_len),
            encoder_x=None,
            encoder_k=torch.stack(encoder_k, dim=0).contiguous(),
            encoder_v=torch.stack(encoder_v, dim=0).contiguous(),
            prefix_pad_masks=prefix_pad_masks.detach().clone(),
        )

    def capture_full_cache_snapshot(self):
        if self._last_cache_snapshot is None:
            raise ValueError("compiled Spec cache snapshot is unavailable before a full round")
        return self._last_cache_snapshot

    def run_full_with_timing(
        self,
        *,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        full_runtime: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from openpi.models_pytorch.spec_pi0_pytorch import clone_past_key_values

        prefix, encoder_ms = self.prepare_prefix_with_timing(prepared=prepared, full_runtime=full_runtime)
        state = self._as_state_batch(prepared.state).to(device=self._device, dtype=torch.float32)
        noise = self._as_action_batch(noise).to(device=self._device, dtype=torch.float32)
        prefix_embs = prefix.prefix_embs.to(device=self._device, dtype=torch.float32)
        prefix_pad_masks = prefix.prefix_pad_masks.to(device=self._device, dtype=torch.bool)
        prefix_att_masks = prefix.prefix_att_masks.to(device=self._device, dtype=torch.bool)

        def _run_prefill():
            with torch.inference_mode():
                return clone_past_key_values(
                    self._spec_model._vlm_prefill_stage(prefix_embs, prefix_pad_masks, prefix_att_masks)
                )

        past_key_values, prefill_ms = self._time_region(_run_prefill)
        self._last_cache_snapshot = self._past_key_values_to_snapshot(
            past_key_values,
            prefix_len=int(prefix_pad_masks.shape[1]),
            prefix_pad_masks=prefix_pad_masks,
        )

        def _run_action():
            with torch.inference_mode():
                return self._spec_model._full_action_stage(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    noise,
                    int(self._spec_model.spec_args.full_num_steps),
                )

        actions, decoder_ms = self._time_region(_run_action)
        return actions, {
            "encoder_ms": float(encoder_ms),
            "vlm_prefill_ms": float(prefill_ms),
            "decoder_ms": float(decoder_ms),
            "total_ms": float(encoder_ms + prefill_ms + decoder_ms),
        }

    @staticmethod
    def _snapshot_to_past_key_values(*, cache_snapshot: Any, batch_size: int, device: torch.device):
        if getattr(cache_snapshot, "encoder_k", None) is None or getattr(cache_snapshot, "encoder_v", None) is None:
            raise ValueError("compiled Spec verify requires encoder_k and encoder_v in the cache snapshot")
        if int(batch_size) != 1:
            raise ValueError("compiled Spec verify currently supports batch_size=1 only")
        prefix_len = int(cache_snapshot.encoder_seq_len)
        encoder_k = cache_snapshot.encoder_k[:, :prefix_len].to(device=device)
        encoder_v = cache_snapshot.encoder_v[:, :prefix_len].to(device=device)
        legacy_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(int(encoder_k.shape[0])):
            key = encoder_k[layer_idx].unsqueeze(0).unsqueeze(0).contiguous()
            value = encoder_v[layer_idx].unsqueeze(0).unsqueeze(0).contiguous()
            legacy_cache.append((key, value))
        try:
            from openpi.models_pytorch.spec_pi0_pytorch import _DynamicCache

            if _DynamicCache is not None:
                return _DynamicCache.from_legacy_cache(tuple(legacy_cache))
        except Exception:
            pass
        return legacy_cache

    def run_draft_with_timing(
        self,
        *,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        last_actions: torch.Tensor,
        prefix_runtime: Any,
        full_runtime: Any,
        encoder_runtime: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if encoder_runtime is not None and hasattr(encoder_runtime, "prepare_prefix_with_timing"):
            prefix, encoder_ms = encoder_runtime.prepare_prefix_with_timing(
                prepared=prepared,
                full_runtime=full_runtime,
                prefix_runtime=prefix_runtime,
            )
        elif hasattr(prefix_runtime, "_prepare_prefix_with_timing"):
            prefix, encoder_ms = prefix_runtime._prepare_prefix_with_timing(
                prepared.images,
                prepared.state,
                full_runtime=full_runtime,
            )
        elif hasattr(prefix_runtime, "prepare_prefix"):
            prefix, encoder_ms = self._time_region(
                lambda: prefix_runtime.prepare_prefix(
                    prepared.images,
                    prepared.state,
                    full_runtime=full_runtime,
                )
            )
        else:
            raise AttributeError("compiled Spec draft requires prefix_runtime.prepare_prefix support")

        state = self._as_state_batch(prepared.state).to(device=self._device, dtype=torch.float32)
        noise = self._as_action_batch(noise).to(device=self._device, dtype=torch.float32)
        last_actions = self._as_action_batch(last_actions).to(device=self._device, dtype=torch.float32)
        prefix_embs = prefix.prefix_embs.to(device=self._device, dtype=torch.float32)
        prefix_pad_masks = prefix.prefix_pad_masks.to(device=self._device, dtype=torch.bool)
        prefix_att_masks = prefix.prefix_att_masks.to(device=self._device, dtype=torch.bool)

        def _run():
            with torch.inference_mode():
                return self._spec_model._compute_draft(
                    noise=noise,
                    prefix_embs=prefix_embs,
                    prefix_pad_masks=prefix_pad_masks,
                    prefix_att_masks=prefix_att_masks,
                    robot_state=state,
                    last_actions=last_actions,
                )

        x0_draft, draft_ms = self._time_region(_run)
        return x0_draft, {
            "encoder_ms": float(encoder_ms),
            "draft_ms": float(draft_ms),
        }

    def run_verify_semantics_with_timing(
        self,
        *,
        cache_snapshot: Any,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        x0_draft: torch.Tensor,
        t_list: tuple[float, ...],
        tau_radius: float,
        dist_dims: int,
        max_exec_steps: int,
        last_gripper: torch.Tensor | None,
        gripper_switch_threshold: float,
        enable_gripper_verify: bool,
        enable_gripper_post_verify: bool,
    ) -> tuple[Any, dict[str, float]]:
        del t_list, tau_radius, dist_dims, max_exec_steps, gripper_switch_threshold, enable_gripper_verify, enable_gripper_post_verify
        output_device = x0_draft.device
        state = prepared.state
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(device=self._device, dtype=torch.float32)
        noise = self._as_action_batch(noise).to(device=self._device, dtype=torch.float32)
        x0_draft = self._as_action_batch(x0_draft).to(device=self._device, dtype=torch.float32)
        if last_gripper is not None:
            last_gripper = last_gripper.to(device=self._device, dtype=torch.float32)
        prefix_pad_masks = getattr(cache_snapshot, "prefix_pad_masks", None)
        if prefix_pad_masks is None:
            prefix_pad_masks = torch.ones(
                (int(state.shape[0]), int(cache_snapshot.encoder_seq_len)),
                device=self._device,
                dtype=torch.bool,
            )
        else:
            prefix_pad_masks = prefix_pad_masks.to(device=self._device, dtype=torch.bool)
        past_key_values = self._snapshot_to_past_key_values(
            cache_snapshot=cache_snapshot,
            batch_size=int(state.shape[0]),
            device=self._device,
        )

        def _run():
            with torch.inference_mode():
                return self._spec_model._action_stage(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    noise,
                    x0_draft,
                    last_gripper,
                )

        (actions, metrics, accepted_prefix_len), action_verify_ms = self._time_region(_run)
        result = types.SimpleNamespace(
            actions=actions.to(device=output_device),
            metrics=metrics.to(device=output_device),
            accepted_prefix_len=accepted_prefix_len.to(device=output_device),
        )
        return result, {"action_verify_ms": float(action_verify_ms)}


class SpecTritonPolicyRuntime:
    def __init__(
        self,
        *,
        runtime_pool: "SpecTritonRuntimePool",
        action_horizon: int,
        action_dim: int,
        max_exec_steps: int,
        device: str,
        draft_history_len: int = 6,
        t_list: tuple[float, ...] = (0.10, 0.05),
        tau_radius: float = 0.3,
        dist_dims: int = 7,
        gripper_switch_threshold: float = 0.0,
        enable_gripper_verify: bool = True,
        enable_gripper_post_verify: bool = True,
        gripper_full_window: int = 1,
        full_fallback: bool = True,
        force_full_each_round: bool = False,
        periodic_full_every_n_draft_rounds: int = 0,
        compiled_encoder_runtime: Any | None = None,
        compiled_draft_runtime: Any | None = None,
        compiled_verify_runtime: Any | None = None,
        retrieval_draft_provider: Any | None = None,
        rtcache_rerank_min_accept_len: int = 8,
        rtcache_noverify_policy: str = "off",
        rtcache_noverify_max_consecutive: int = 2,
        rtcache_suite_name: str | None = None,
        rtcache_composite_window_size: int = 5,
        rtcache_composite_alpha: float = 0.5,
    ) -> None:
        self._runtime_pool = runtime_pool
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._max_exec_steps = int(max_exec_steps)
        self._device = str(device)
        self._draft_history_len = int(max(1, int(draft_history_len)))
        self._t_list = tuple(float(t) for t in t_list)
        self._tau_radius = float(tau_radius)
        self._dist_dims = int(dist_dims)
        self._gripper_switch_threshold = float(gripper_switch_threshold)
        self._enable_gripper_verify = bool(enable_gripper_verify)
        self._enable_gripper_post_verify = bool(enable_gripper_post_verify)
        self._gripper_full_window = int(gripper_full_window)
        self._full_fallback = bool(full_fallback)
        self._force_full_each_round = bool(force_full_each_round)
        self._periodic_full_every_n_draft_rounds = int(periodic_full_every_n_draft_rounds)
        self._compiled_encoder_runtime = compiled_encoder_runtime
        self._compiled_draft_runtime = compiled_draft_runtime
        self._compiled_verify_runtime = compiled_verify_runtime
        self._retrieval_draft_provider = retrieval_draft_provider
        self._rtcache_rerank_min_accept_len = int(max(1, int(rtcache_rerank_min_accept_len)))
        self._rtcache_noverify_policy = str(rtcache_noverify_policy or "off")
        self._rtcache_noverify_max_consecutive = int(max(0, int(rtcache_noverify_max_consecutive)))
        self._rtcache_suite_name = _rtcache_suite_key(rtcache_suite_name)
        self._rtcache_composite_window_size = int(max(2, int(rtcache_composite_window_size)))
        self._rtcache_composite_alpha = float(np.clip(float(rtcache_composite_alpha), 0.0, 1.0))
        self.reset_runtime_state()

    def reload_manifest(self) -> None:
        self._runtime_pool.reload_manifest()

    @staticmethod
    def _as_action_batch(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            return actions.unsqueeze(0)
        return actions

    @staticmethod
    def _maybe_unbatch_actions(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 3 and int(actions.shape[0]) == 1:
            return actions[0]
        return actions

    def _runtime_state_device(self) -> torch.device:
        if self._action_chunk_cache is not None:
            return self._action_chunk_cache.device
        if self._last_actions is not None:
            return self._last_actions.device
        if self._last_gripper is not None:
            return self._last_gripper.device
        return torch.device(self._device)

    def _init_runtime_state(self, batch_size: int) -> None:
        device = self._runtime_state_device()
        if self._compiled_draft_runtime is not None:
            if (
                self._last_actions is None
                or int(self._last_actions.shape[0]) != int(batch_size)
                or int(self._last_actions.shape[2]) != int(self._action_dim)
            ):
                self._last_actions = torch.zeros(
                    (int(batch_size), int(self._draft_history_len), int(self._action_dim)),
                    device=device,
                    dtype=torch.float32,
                )
            elif self._last_actions.device != device:
                self._last_actions = self._last_actions.to(device=device, dtype=torch.float32)
        if int(self._action_dim) < 7:
            self._last_gripper = None
            return
        if (
            self._last_gripper is None
            or int(self._last_gripper.shape[0]) != int(batch_size)
        ):
            self._last_gripper = torch.zeros(
                (int(batch_size),),
                device=device,
                dtype=torch.float32,
            )
        elif self._last_gripper.device != device:
            self._last_gripper = self._last_gripper.to(device=device, dtype=torch.float32)

    def _advance_runtime_state(self, executed_steps: int | None) -> None:
        if self._action_chunk_cache is None:
            return
        batch_size = int(self._action_chunk_cache.shape[0])
        if self._compiled_draft_runtime is not None and (
            self._last_actions is None
            or int(self._last_actions.shape[0]) != batch_size
            or int(self._last_actions.shape[2]) != int(self._action_chunk_cache.shape[2])
        ):
            self._last_actions = torch.zeros(
                (batch_size, int(self._draft_history_len), int(self._action_chunk_cache.shape[2])),
                device=self._action_chunk_cache.device,
                dtype=torch.float32,
            )
        if self._last_actions is not None and self._last_actions.device != self._action_chunk_cache.device:
            self._last_actions = self._last_actions.to(device=self._action_chunk_cache.device, dtype=torch.float32)
        if self._last_gripper is not None and self._last_gripper.device != self._action_chunk_cache.device:
            self._last_gripper = self._last_gripper.to(device=self._action_chunk_cache.device, dtype=torch.float32)
        if self._last_gripper is not None and int(self._last_gripper.shape[0]) != batch_size:
            self._last_gripper = torch.zeros(
                (batch_size,),
                device=self._action_chunk_cache.device,
                dtype=torch.float32,
            )
        if self._last_actions is None and self._last_gripper is None:
            return
        horizon = int(self._action_chunk_cache.shape[1])
        step_adv = int(self._max_exec_steps) if executed_steps is None else int(executed_steps)
        step_adv = max(0, min(step_adv, horizon))
        self._action_cache_ptr = min(int(self._action_cache_ptr) + step_adv, horizon)
        if self._action_cache_ptr <= 0:
            return
        executed_anchor = self._action_chunk_cache[:, self._action_cache_ptr - 1, :].to(dtype=torch.float32)
        if self._last_actions is not None:
            self._last_actions = torch.cat(
                [self._last_actions[:, 1:, :], executed_anchor[:, None, :]],
                dim=1,
            ).detach()
        if self._last_gripper is not None and int(executed_anchor.shape[1]) >= 7:
            self._last_gripper = executed_anchor[:, 6].detach()

    def reset_runtime_state(self) -> None:
        self._last_actions: torch.Tensor | None = None
        self._last_gripper: torch.Tensor | None = None
        self._action_chunk_cache: torch.Tensor | None = None
        self._action_cache_ptr: int = 0
        self._draft_rounds_since_full: int = 0
        self._pending_full_fallback: bool = False
        self._gripper_full_rounds_left: int = 0
        self._full_cache_snapshot = None
        self._rtcache_noverify_streak: int = 0
        self._rtcache_eef_history: list[np.ndarray] = []

    def _accept_full_round_actions(self, actions: torch.Tensor, *, cache_snapshot) -> None:
        self._draft_rounds_since_full = 0
        self._rtcache_noverify_streak = 0
        if int(self._gripper_full_rounds_left) > 0:
            self._gripper_full_rounds_left = max(0, int(self._gripper_full_rounds_left) - 1)
            self._pending_full_fallback = bool(self._gripper_full_rounds_left > 0)
        else:
            self._pending_full_fallback = False
        self._full_cache_snapshot = cache_snapshot
        self._action_chunk_cache = actions.detach().clone()
        self._action_cache_ptr = 0

    def _schedule_gripper_full_fallback(self) -> None:
        self._pending_full_fallback = True
        self._gripper_full_rounds_left = max(1, int(self._gripper_full_window))

    def _run_full_with_timing(self, *, session: Any, prepared: TritonPreparedObservation, noise: torch.Tensor):
        if self._compiled_encoder_runtime is not None:
            return self._compiled_encoder_runtime.run_full_with_timing(
                prepared=prepared,
                noise=noise,
                full_runtime=session._runtime,
            )
        return session.run_full_with_timing(prepared=prepared, noise=noise)

    @staticmethod
    def _prepare_session_observation(
        *,
        session: Any,
        images: torch.Tensor,
        state: torch.Tensor,
        tokenized_prompt: torch.Tensor | None,
        tokenized_prompt_mask: torch.Tensor | None,
    ):
        try:
            return session.prepare_observation(
                images=images,
                state=state,
                tokenized_prompt=tokenized_prompt,
                tokenized_prompt_mask=tokenized_prompt_mask,
            )
        except TypeError:
            return session.prepare_observation(images=images, state=state)

    def _capture_full_cache_snapshot(self, *, session: Any):
        if self._compiled_encoder_runtime is not None and hasattr(
            self._compiled_encoder_runtime,
            "capture_full_cache_snapshot",
        ):
            return self._compiled_encoder_runtime.capture_full_cache_snapshot()
        return session.capture_full_cache_snapshot()

    def _run_draft_with_timing(
        self,
        *,
        session: Any,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        last_actions: torch.Tensor | None,
        prompt: str,
        trace_task_id: int | None = None,
    ):
        if self._retrieval_draft_provider is not None:
            retrieval_vector = None
            encoder_ms = None
            if self._compiled_encoder_runtime is not None and hasattr(
                self._compiled_encoder_runtime,
                "prepare_prefix_with_timing",
            ):
                prefix, encoder_ms = self._compiled_encoder_runtime.prepare_prefix_with_timing(
                    prepared=prepared,
                    full_runtime=session._runtime,
                    prefix_runtime=getattr(session, "draft_runtime", None),
                )
                retrieval_vector = getattr(prefix, "retrieval_vector", None)
            return self._retrieval_draft_provider.run_draft_with_timing(
                prepared=prepared,
                prompt=prompt,
                trace_task_id=trace_task_id,
                action_horizon=self._action_horizon,
                action_dim=self._action_dim,
                device=noise.device,
                retrieval_vector=retrieval_vector,
                retrieval_encoder_ms=encoder_ms,
            )
        if self._compiled_draft_runtime is not None:
            if last_actions is None:
                raise RuntimeError("compiled Spec draft requires initialized last_actions history")
            kwargs = {
                "prepared": prepared,
                "noise": self._as_action_batch(noise),
                "last_actions": last_actions,
                "prefix_runtime": getattr(session, "draft_runtime", None),
                "full_runtime": session._runtime,
            }
            if self._compiled_encoder_runtime is not None:
                kwargs["encoder_runtime"] = self._compiled_encoder_runtime
            return self._compiled_draft_runtime.run_draft_with_timing(**kwargs)
        if self._compiled_encoder_runtime is not None:
            raise AttributeError("compiled Spec encoder requires the compiled Spec draft backend")
        return session.run_draft_with_timing(prepared=prepared)

    def _run_retrieval_candidates_with_timing(
        self,
        *,
        session: Any,
        prepared: TritonPreparedObservation,
        noise: torch.Tensor,
        prompt: str,
        trace_task_id: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float], list[dict[str, Any]]]:
        if self._retrieval_draft_provider is None:
            raise RuntimeError("retrieval candidate path requires a retrieval draft provider")
        retrieval_vector = None
        encoder_ms = None
        if self._compiled_encoder_runtime is not None and hasattr(
            self._compiled_encoder_runtime,
            "prepare_prefix_with_timing",
        ):
            prefix, encoder_ms = self._compiled_encoder_runtime.prepare_prefix_with_timing(
                prepared=prepared,
                full_runtime=session._runtime,
                prefix_runtime=getattr(session, "draft_runtime", None),
            )
            retrieval_vector = getattr(prefix, "retrieval_vector", None)
        if hasattr(self._retrieval_draft_provider, "run_candidates_with_timing"):
            return self._retrieval_draft_provider.run_candidates_with_timing(
                prepared=prepared,
                prompt=prompt,
                trace_task_id=trace_task_id,
                action_horizon=self._action_horizon,
                action_dim=self._action_dim,
                device=noise.device,
                retrieval_vector=retrieval_vector,
                retrieval_encoder_ms=encoder_ms,
            )
        x0_draft, draft_timing = self._retrieval_draft_provider.run_draft_with_timing(
            prepared=prepared,
            prompt=prompt,
            trace_task_id=trace_task_id,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            device=noise.device,
            retrieval_vector=retrieval_vector,
            retrieval_encoder_ms=encoder_ms,
        )
        return x0_draft, draft_timing, [{"rank": 0, "score": draft_timing.get("rtcache_top_score", float("nan"))}]

    @staticmethod
    def _rtcache_candidate_score(meta: dict[str, Any]) -> float:
        try:
            score = float(meta.get("score", float("nan")))
        except (TypeError, ValueError):
            score = float("nan")
        return score if math.isfinite(score) else float("-inf")

    @staticmethod
    def _rtcache_candidate_radius(metrics: torch.Tensor) -> float:
        try:
            radius = float(metrics[0].item())
        except Exception:
            return float("inf")
        return radius if math.isfinite(radius) else float("inf")

    @staticmethod
    def _rtcache_accept_len(accepted_prefix_len: torch.Tensor) -> float:
        try:
            return float(accepted_prefix_len.to(dtype=torch.float32).mean().item())
        except Exception:
            return 0.0

    @staticmethod
    def _apply_rtcache_candidate_meta(timing: dict[str, float], meta: dict[str, Any]) -> None:
        timing["rtcache_selected_rank"] = float(meta.get("rank", 0))
        timing["rtcache_selected_score"] = float(meta.get("score", float("nan")))
        timing["rtcache_record_index"] = float(meta.get("index", -1))
        timing["rtcache_record_task_id"] = float(meta.get("record_task_id", -1))
        timing["rtcache_record_episode_idx"] = float(meta.get("record_episode_idx", -1))
        timing["rtcache_record_infer_id"] = float(meta.get("record_infer_id", -1))

    def _rtcache_composite_timing(self, *, prepared: TritonPreparedObservation) -> dict[str, float]:
        timing = {
            "rtcache_composite_ready": 0.0,
            "rtcache_composite": float("nan"),
            "rtcache_composite_threshold": float("nan"),
            "rtcache_composite_displacement": float("nan"),
            "rtcache_composite_radius": float("nan"),
            "rtcache_composite_norm_displacement": float("nan"),
            "rtcache_composite_norm_radius": float("nan"),
            "rtcache_noverify_streak": float(self._rtcache_noverify_streak),
        }
        if self._retrieval_draft_provider is None or self._rtcache_noverify_policy == "off":
            return timing
        raw_state_fn = getattr(self._retrieval_draft_provider, "raw_state_from_prepared", None)
        if not callable(raw_state_fn):
            return timing
        try:
            raw_state = np.asarray(raw_state_fn(prepared), dtype=np.float64).reshape(-1)
        except Exception:
            return timing
        if int(raw_state.shape[0]) < 3:
            return timing
        self._rtcache_eef_history.append(raw_state[:3].copy())
        window = int(self._rtcache_composite_window_size)
        if len(self._rtcache_eef_history) > window:
            self._rtcache_eef_history = self._rtcache_eef_history[-window:]
        if len(self._rtcache_eef_history) < min(3, window):
            return timing

        points = np.asarray(self._rtcache_eef_history[-window:], dtype=np.float64)
        displacement = float(np.linalg.norm(points[:-1] - points[-1], axis=1).sum()) if int(points.shape[0]) > 1 else 0.0
        radius = _rtcache_circle_radius(points)
        config = _RTCACHE_COMPOSITE_DEFAULTS.get(
            self._rtcache_suite_name,
            _RTCACHE_COMPOSITE_DEFAULTS["libero_goal"],
        )
        norm_displacement = _rtcache_normalize(displacement, config["disp_min"], config["disp_max"])
        norm_radius = _rtcache_normalize(radius, config["radius_min"], config["radius_max"])
        if not math.isfinite(norm_displacement) or not math.isfinite(norm_radius):
            return timing
        alpha = float(self._rtcache_composite_alpha)
        composite = float(alpha * norm_radius + (1.0 - alpha) * norm_displacement)
        timing.update(
            {
                "rtcache_composite_ready": 1.0,
                "rtcache_composite": composite,
                "rtcache_composite_threshold": float(config["threshold"]),
                "rtcache_composite_displacement": displacement,
                "rtcache_composite_radius": radius,
                "rtcache_composite_norm_displacement": norm_displacement,
                "rtcache_composite_norm_radius": norm_radius,
                "rtcache_noverify_streak": float(self._rtcache_noverify_streak),
            }
        )
        return timing

    def _rtcache_should_noverify(self, composite_timing: dict[str, float]) -> bool:
        if self._rtcache_noverify_policy != "composite_2to1":
            return False
        if int(self._rtcache_noverify_max_consecutive) <= 0:
            return False
        if int(self._rtcache_noverify_streak) >= int(self._rtcache_noverify_max_consecutive):
            return False
        if float(composite_timing.get("rtcache_composite_ready", 0.0)) < 0.5:
            return False
        composite = float(composite_timing.get("rtcache_composite", float("nan")))
        threshold = float(composite_timing.get("rtcache_composite_threshold", float("nan")))
        return math.isfinite(composite) and math.isfinite(threshold) and composite > threshold

    def sample_actions_with_timing(
        self,
        *,
        prompt: str,
        images: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        tokenized_prompt: torch.Tensor | None = None,
        tokenized_prompt_mask: torch.Tensor | None = None,
        executed_steps: int | None = None,
        trace_task_id: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        from openpi.models_pytorch.spec_pi0_pytorch import _populate_full_round_timing
        from openpi.models_pytorch.spec_pi0_pytorch import _set_legacy_timing_compat_fields
        from openpi.models_pytorch.spec_pi0_pytorch import _should_run_full_pipeline_round
        from openpi.models_pytorch.spec_pi0_pytorch import _should_schedule_full_fallback

        batch_size = 1 if state.ndim == 1 else int(state.shape[0])
        self._init_runtime_state(batch_size)
        self._advance_runtime_state(executed_steps)

        session = self._runtime_pool.start_session(prompt)
        prepared = self._prepare_session_observation(
            session=session,
            images=images,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )
        run_full_pipeline_round = _should_run_full_pipeline_round(
            cache_ready=self._full_cache_snapshot is not None,
            full_fallback=bool(self._full_fallback),
            pending_full_fallback=bool(self._pending_full_fallback),
            force_full_each_round=bool(self._force_full_each_round),
            periodic_full_every_n_draft_rounds=int(self._periodic_full_every_n_draft_rounds),
            draft_rounds_since_full=int(self._draft_rounds_since_full),
        )

        if run_full_pipeline_round:
            full_out = self._run_full_with_timing(session=session, prepared=prepared, noise=noise)
            if isinstance(full_out, tuple):
                actions, stage_timing = full_out
            else:
                actions, stage_timing = full_out, {}
            full_snapshot = self._capture_full_cache_snapshot(session=session)
            action_batch = self._as_action_batch(actions)
            self._accept_full_round_actions(action_batch, cache_snapshot=full_snapshot)
            timing = {
                "encoder_ms": float(stage_timing.get("encoder_ms", float("nan"))),
            }
            _populate_full_round_timing(
                timing,
                verify_mode="radius",
                action_horizon=self._action_horizon,
                max_exec_steps=self._max_exec_steps,
                full_prefill_ms=float(stage_timing.get("vlm_prefill_ms", 0.0)),
                full_action_ms=float(stage_timing.get("decoder_ms", stage_timing.get("total_ms", 0.0))),
                gripper_verify_enabled=bool(self._enable_gripper_verify),
            )
            timing["total_ms"] = float(stage_timing.get("total_ms", float("nan")))
            return self._maybe_unbatch_actions(action_batch), timing

        if self._compiled_draft_runtime is not None and self._retrieval_draft_provider is None:
            if self._last_actions is None:
                self._init_runtime_state(batch_size)
            if self._last_actions is None:
                raise RuntimeError("compiled Spec draft requires initialized last_actions history")
        verify_runner = self._compiled_verify_runtime
        if verify_runner is None:
            verify_runner = session

        candidate_meta: list[dict[str, Any]] = []
        composite_timing: dict[str, float] = {}
        rerank_extra_verifies = 0
        rerank_verify_ms = 0.0
        if self._retrieval_draft_provider is not None:
            x0_candidates, draft_timing, candidate_meta = self._run_retrieval_candidates_with_timing(
                session=session,
                prepared=prepared,
                noise=noise,
                prompt=prompt,
                trace_task_id=trace_task_id,
            )
            if x0_candidates.ndim == 2:
                x0_candidates = x0_candidates.unsqueeze(0)
            if x0_candidates.ndim != 3:
                raise ValueError(f"Expected retrieval candidates with shape [K,H,D], got {tuple(x0_candidates.shape)}")
            composite_timing = self._rtcache_composite_timing(prepared=prepared)
            draft_timing.update(composite_timing)

            if self._rtcache_should_noverify(composite_timing):
                actions = x0_candidates[:1].detach().clone()
                meta = candidate_meta[0] if candidate_meta else {"rank": 0}
                accepted_len = float(min(int(actions.shape[1]), max(1, int(self._max_exec_steps))))
                self._action_chunk_cache = actions.detach().clone()
                self._action_cache_ptr = 0
                self._draft_rounds_since_full = int(self._draft_rounds_since_full) + 1
                self._rtcache_noverify_streak = int(self._rtcache_noverify_streak) + 1
                timing = {
                    "encoder_ms": float(draft_timing.get("encoder_ms", float("nan"))),
                    "vlm_prefill_ms": 0.0,
                    "draft_ms": float(draft_timing.get("draft_ms", float("nan"))),
                    "action_verify_ms": 0.0,
                    "full_fallback_ms": 0.0,
                    "used_full_fallback": 0.0,
                    "scheduled_full_fallback": 0.0,
                    "is_full_pipeline_round": 0.0,
                    "include_in_draft_accept_metrics": 0.0,
                    "radius_dist": float("nan"),
                    "radius_dist_mean": float("nan"),
                    "verify_mode_random": 0.0,
                    "accepted_prefix_len_mean": accepted_len,
                    "gripper_switch_cut_rate": 0.0,
                    "scheduled_full_fallback_gripper": 0.0,
                    "gripper_verify_stop_rate": 0.0,
                    "gripper_verify_enabled": 1.0 if self._enable_gripper_verify else 0.0,
                    "accepted_prefix_len": accepted_len,
                }
                for key, value in draft_timing.items():
                    if str(key).startswith("rtcache_"):
                        timing[str(key)] = float(value)
                self._apply_rtcache_candidate_meta(timing, meta)
                timing["rtcache_noverify"] = 1.0
                timing["rtcache_noverify_streak"] = float(self._rtcache_noverify_streak)
                timing["rtcache_best_accept_len"] = accepted_len
                timing["rtcache_rerank_verify_ms"] = 0.0
                timing["rtcache_rerank_extra_verifies"] = 0.0
                timing["total_ms"] = float(
                    timing["encoder_ms"]
                    + timing["vlm_prefill_ms"]
                    + timing["draft_ms"]
                    + timing["action_verify_ms"]
                )
                _set_legacy_timing_compat_fields(timing)
                return self._maybe_unbatch_actions(actions), timing

            self._rtcache_noverify_streak = 0
            last_gripper = None
            if self._last_gripper is not None and int(x0_candidates.shape[2]) >= 7:
                last_gripper = self._last_gripper.to(device=x0_candidates.device, dtype=torch.float32)

            candidate_evals: list[dict[str, Any]] = []
            for rank in range(int(x0_candidates.shape[0])):
                x0_draft_i = x0_candidates[rank : rank + 1].contiguous()
                verify_result_i, verify_timing_i = verify_runner.run_verify_semantics_with_timing(
                    cache_snapshot=self._full_cache_snapshot,
                    prepared=prepared,
                    noise=self._as_action_batch(noise),
                    x0_draft=x0_draft_i,
                    t_list=self._t_list,
                    tau_radius=self._tau_radius,
                    dist_dims=self._dist_dims,
                    max_exec_steps=self._max_exec_steps,
                    last_gripper=last_gripper,
                    gripper_switch_threshold=self._gripper_switch_threshold,
                    enable_gripper_verify=self._enable_gripper_verify,
                    enable_gripper_post_verify=self._enable_gripper_post_verify,
                )
                verify_ms = float(verify_timing_i.get("action_verify_ms", 0.0))
                rerank_verify_ms += verify_ms
                meta = candidate_meta[rank] if rank < len(candidate_meta) else {"rank": rank}
                accept_len = self._rtcache_accept_len(verify_result_i.accepted_prefix_len)
                radius = self._rtcache_candidate_radius(verify_result_i.metrics)
                candidate_evals.append(
                    {
                        "rank": rank,
                        "meta": meta,
                        "verify_result": verify_result_i,
                        "verify_timing": verify_timing_i,
                        "accept_len": accept_len,
                        "radius": radius,
                        "score": self._rtcache_candidate_score(meta),
                    }
                )
                if accept_len >= float(self._rtcache_rerank_min_accept_len):
                    break

            if not candidate_evals:
                raise RuntimeError("Retrieval rerank produced no candidate evaluations.")

            def _candidate_key(item: dict[str, Any]) -> tuple[float, float, float]:
                return (
                    float(item["accept_len"]),
                    -float(item["radius"]),
                    float(item["score"]),
                )

            selected = max(candidate_evals, key=_candidate_key)
            rerank_extra_verifies = max(0, len(candidate_evals) - 1)
            verify_result = selected["verify_result"]
            verify_timing = {"action_verify_ms": float(rerank_verify_ms)}
            draft_timing["rtcache_selected_rank"] = float(selected["rank"])
            draft_timing["rtcache_best_accept_len"] = float(selected["accept_len"])
            draft_timing["rtcache_rerank_verify_ms"] = float(rerank_verify_ms)
            draft_timing["rtcache_rerank_extra_verifies"] = float(rerank_extra_verifies)
            draft_timing["rtcache_noverify"] = 0.0
            draft_timing["rtcache_noverify_streak"] = 0.0
            self._apply_rtcache_candidate_meta(draft_timing, selected["meta"])
        else:
            x0_draft, draft_timing = self._run_draft_with_timing(
                session=session,
                prepared=prepared,
                noise=noise,
                last_actions=self._last_actions,
                prompt=prompt,
                trace_task_id=trace_task_id,
            )
            last_gripper = None
            if self._last_gripper is not None and int(x0_draft.shape[2]) >= 7:
                last_gripper = self._last_gripper.to(device=x0_draft.device, dtype=torch.float32)
            verify_result, verify_timing = verify_runner.run_verify_semantics_with_timing(
                cache_snapshot=self._full_cache_snapshot,
                prepared=prepared,
                noise=self._as_action_batch(noise),
                x0_draft=x0_draft,
                t_list=self._t_list,
                tau_radius=self._tau_radius,
                dist_dims=self._dist_dims,
                max_exec_steps=self._max_exec_steps,
                last_gripper=last_gripper,
                gripper_switch_threshold=self._gripper_switch_threshold,
                enable_gripper_verify=self._enable_gripper_verify,
                enable_gripper_post_verify=self._enable_gripper_post_verify,
            )
        actions = verify_result.actions
        metrics = verify_result.metrics
        accepted_prefix_len = verify_result.accepted_prefix_len

        should_schedule_full_fallback = _should_schedule_full_fallback(
            full_fallback=bool(self._full_fallback),
            accepted_prefix_len=accepted_prefix_len,
            gripper_switch_cut_mask=metrics[3:4].to(dtype=torch.bool),
        )
        if should_schedule_full_fallback:
            if bool(metrics[3].item()):
                self._schedule_gripper_full_fallback()
            else:
                self._pending_full_fallback = True
                self._gripper_full_rounds_left = 0
        else:
            self._action_chunk_cache = actions.detach().clone()
            self._action_cache_ptr = 0

        self._draft_rounds_since_full = int(self._draft_rounds_since_full) + 1
        timing = {
            "encoder_ms": float(draft_timing.get("encoder_ms", float("nan"))),
            "vlm_prefill_ms": 0.0,
            "draft_ms": float(draft_timing.get("draft_ms", float("nan"))),
            "action_verify_ms": float(verify_timing.get("action_verify_ms", float("nan"))),
            "full_fallback_ms": 0.0,
            "used_full_fallback": 0.0,
            "scheduled_full_fallback": 1.0 if should_schedule_full_fallback else 0.0,
            "is_full_pipeline_round": 0.0,
            "include_in_draft_accept_metrics": 1.0,
        }
        for key, value in draft_timing.items():
            if str(key).startswith("rtcache_"):
                timing[str(key)] = float(value)
        timing["total_ms"] = float(
            timing["encoder_ms"]
            + timing["vlm_prefill_ms"]
            + timing["draft_ms"]
            + timing["action_verify_ms"]
        )
        timing["radius_dist"] = float(metrics[0].item())
        timing["radius_dist_mean"] = float(metrics[0].item())
        timing["verify_mode_random"] = 0.0
        timing["accepted_prefix_len_mean"] = float(metrics[1].item())
        _set_legacy_timing_compat_fields(timing)
        timing["gripper_switch_cut_rate"] = float(metrics[2].item())
        timing["scheduled_full_fallback_gripper"] = float(metrics[3].item())
        timing["gripper_verify_stop_rate"] = float(metrics[4].item())
        timing["gripper_verify_enabled"] = 1.0 if self._enable_gripper_verify else 0.0
        timing["accepted_prefix_len"] = float(accepted_prefix_len.to(dtype=torch.float32).mean().item())
        return self._maybe_unbatch_actions(actions), timing


class _StagedPi0Runtime:
    """Wrapper around the Triton pi0 runtime with staged timing support."""

    def __init__(
        self,
        *,
        infer_module: ModuleType,
        checkpoint: dict[str, torch.Tensor],
        num_views: int,
        chunk_size: int,
    ) -> None:
        self._infer_module = infer_module
        self._inner = infer_module.Pi0Inference(
            checkpoint=checkpoint,
            num_views=int(num_views),
            chunk_size=int(chunk_size),
        )
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.weights = self._inner.weights
        self.buffers = self._inner.buffers
        self._encoder_seq_len = int(self.buffers["encoder_x"].shape[0])
        self._prompt_offset = int(self.num_views * 256)
        self._encoder_graph = self._capture_stage_graph(self._record_encoder_stage)
        self._prefill_graph = self._capture_stage_graph(self._record_prefill_stage)
        self._decoder_graph = self._capture_stage_graph(self._record_decoder_stage)

    def _record_encoder_stage(self) -> None:
        self._infer_module.vision_encoder(self.weights, self.buffers, self.num_views)

    def _record_prefill_stage(self) -> None:
        self.buffers["encoder_x"][self._prompt_offset :].copy_(self.weights["language_embeds"])
        self._infer_module.transformer_encoder(self.weights, self.buffers, self._encoder_seq_len)

    def _record_decoder_stage(self) -> None:
        self._infer_module.transformer_decoder(self.weights, self.buffers, self._encoder_seq_len)

    @staticmethod
    def _time_cuda_region(fn) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    def _capture_stage_graph(self, record_fn):
        if not torch.cuda.is_available():
            return None

        warmup_stream = torch.cuda.Stream()
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                record_fn()
        warmup_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_stream = torch.cuda.Stream()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph):
                record_fn()
        capture_stream.synchronize()
        return graph

    def _replay_or_run(self, graph, record_fn) -> float:
        if graph is None:
            return self._time_cuda_region(record_fn)
        return self._time_cuda_region(graph.replay)

    def forward(self, images: torch.Tensor, state: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._inner.forward(images, state, noise)

    def run_full_with_timing(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.buffers["observation_images_normalized"].copy_(images)
        self.buffers["observation_state_normalized"].copy_(state)
        self.buffers["diffusion_noise"].copy_(noise)

        encoder_ms = self._replay_or_run(self._encoder_graph, self._record_encoder_stage)
        prefill_ms = self._replay_or_run(self._prefill_graph, self._record_prefill_stage)
        decoder_ms = self._replay_or_run(self._decoder_graph, self._record_decoder_stage)
        total_ms = float(encoder_ms + prefill_ms + decoder_ms)

        return self.buffers["diffusion_noise"], {
            "encoder_ms": float(encoder_ms),
            "vlm_prefill_ms": float(prefill_ms),
            "decoder_ms": float(decoder_ms),
            "total_ms": total_ms,
        }


class TritonRuntimePool:
    def __init__(
        self,
        *,
        base_weights_path: str | Path,
        manifest_path: str | Path | None,
        num_views: int,
        chunk_size: int,
        tokenizer_source: str = "auto",
        hf_endpoint: str = "https://hf-mirror.com",
        hf_tokenizer_id: str = "google/paligemma-3b-pt-224",
        runtime_factory=None,
    ) -> None:
        self._base_weights_path = Path(base_weights_path)
        self._manifest_path = Path(manifest_path) if manifest_path is not None else None
        self._cache_dir = self._manifest_path.parent if self._manifest_path is not None else None
        self._num_views = int(num_views)
        self._chunk_size = int(chunk_size)
        self._tokenizer_source = str(tokenizer_source)
        self._hf_endpoint = str(hf_endpoint)
        self._hf_tokenizer_id = str(hf_tokenizer_id)
        self._runtime_factory = runtime_factory or self._default_runtime_factory
        self._runtime_by_prompt_len: dict[int, Any] = {}
        self._language_embeds_by_prompt: dict[str, torch.Tensor] = {}

        with self._base_weights_path.open("rb") as handle:
            self._base_weights = pickle.load(handle)
        self._language_embedding_weight = self._base_weights.pop(_LANGUAGE_EMBEDDING_WEIGHT_KEY, None)
        if "language_embeds" in self._base_weights:
            raise ValueError(
                "Expected shared Triton base weights without language_embeds. Re-convert the base checkpoint "
                "without --prompt, or use a cache manifest generated by build_prompt_cache()."
            )
        self.reload_manifest()

    @staticmethod
    def _evict_runtime_cache(cache: dict[int, Any], keep_prompt_len: int) -> None:
        stale_prompt_lens = [prompt_len for prompt_len in cache if int(prompt_len) != int(keep_prompt_len)]
        if not stale_prompt_lens:
            return
        for prompt_len in stale_prompt_lens:
            del cache[prompt_len]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evict_stale_runtimes(self, keep_prompt_len: int) -> None:
        self._evict_runtime_cache(self._runtime_by_prompt_len, keep_prompt_len)

    def reload_manifest(self) -> None:
        if self._manifest_path is None:
            self._prompt_manifest = {}
            return
        manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self._prompt_manifest = dict(manifest.get("prompts", {}))

    def _default_runtime_factory(self, *, checkpoint: dict[str, torch.Tensor], **_kwargs):
        return create_pi0_inference(
            checkpoint=checkpoint,
            num_views=self._num_views,
            chunk_size=self._chunk_size,
        )

    def _load_language_embeds(self, prompt: str) -> torch.Tensor:
        cached = self._language_embeds_by_prompt.get(prompt)
        if cached is not None:
            return cached

        entry = self._prompt_manifest.get(prompt)
        if entry is None:
            self.reload_manifest()
            entry = self._prompt_manifest.get(prompt)
        if entry is None:
            raise KeyError(f"Prompt not found in Triton cache manifest: {prompt}")

        if self._cache_dir is None:
            raise KeyError(f"Prompt cache manifest is unavailable for prompt: {prompt}")
        embed_path = self._cache_dir / str(entry["embed_path"])
        language_embeds = torch.load(embed_path, map_location="cpu")
        if not isinstance(language_embeds, torch.Tensor):
            raise TypeError(f"Expected a tensor at {embed_path}, got {type(language_embeds)!r}")
        language_embeds = language_embeds.to(dtype=torch.bfloat16, device="cpu").contiguous()
        self._language_embeds_by_prompt[prompt] = language_embeds
        return language_embeds

    @staticmethod
    def _copy_language_embeds(runtime: Any, language_embeds: torch.Tensor) -> None:
        target = runtime.weights["language_embeds"]
        target.copy_(language_embeds.to(device=target.device, dtype=target.dtype))

    def get_runtime(self, prompt: str):
        language_embeds = self._load_language_embeds(str(prompt))
        prompt_len = int(language_embeds.shape[0])
        self._evict_stale_runtimes(prompt_len)
        runtime = self._runtime_by_prompt_len.get(prompt_len)
        if runtime is None:
            checkpoint = dict(self._base_weights)
            checkpoint["language_embeds"] = language_embeds
            runtime = self._runtime_factory(
                checkpoint=checkpoint,
                num_views=self._num_views,
                chunk_size=self._chunk_size,
            )
            self._runtime_by_prompt_len[prompt_len] = runtime
        self._copy_language_embeds(runtime, language_embeds)
        return runtime

    def start_session(self, prompt: str) -> TritonRuntimeSession:
        return TritonRuntimeSession(prompt=str(prompt), runtime=self.get_runtime(prompt))

    def forward(self, *, prompt: str, images: torch.Tensor, state: torch.Tensor, noise: torch.Tensor):
        session = self.start_session(prompt)
        prepared = session.prepare_observation(images=images, state=state)
        return session.run_full(prepared=prepared, noise=noise)


class SpecTritonRuntimePool(TritonRuntimePool):
    def __init__(
        self,
        *,
        base_weights_path: str | Path,
        manifest_path: str | Path | None,
        draft_checkpoint_path: str | Path,
        num_views: int,
        chunk_size: int,
        tokenizer_source: str = "auto",
        hf_endpoint: str = "https://hf-mirror.com",
        hf_tokenizer_id: str = "google/paligemma-3b-pt-224",
        runtime_factory=None,
        spec_runtime_factory=None,
    ) -> None:
        super().__init__(
            base_weights_path=base_weights_path,
            manifest_path=manifest_path,
            num_views=num_views,
            chunk_size=chunk_size,
            tokenizer_source=tokenizer_source,
            hf_endpoint=hf_endpoint,
            hf_tokenizer_id=hf_tokenizer_id,
            runtime_factory=runtime_factory,
        )
        self._draft_checkpoint_path = Path(draft_checkpoint_path)
        with self._draft_checkpoint_path.open("rb") as handle:
            self._draft_checkpoint = pickle.load(handle)
        self._spec_runtime_factory = spec_runtime_factory or self._default_spec_runtime_factory
        self._spec_runtime_by_prompt_len: dict[int, Any] = {}

    def _evict_stale_runtimes(self, keep_prompt_len: int) -> None:
        super()._evict_stale_runtimes(keep_prompt_len)
        self._evict_runtime_cache(self._spec_runtime_by_prompt_len, keep_prompt_len)

    def _default_spec_runtime_factory(
        self,
        *,
        checkpoint: dict[str, torch.Tensor],
        draft_checkpoint: Mapping[str, Any],
        **_kwargs,
    ):
        return create_pi0_spec_inference(
            checkpoint=checkpoint,
            draft_checkpoint=draft_checkpoint,
            num_views=self._num_views,
            chunk_size=self._chunk_size,
        )

    def _maybe_copy_language_embeds(self, runtime: Any, language_embeds: torch.Tensor) -> None:
        weights = getattr(runtime, "weights", None)
        if isinstance(weights, Mapping) and "language_embeds" in weights:
            self._copy_language_embeds(runtime, language_embeds)

    def get_spec_runtime(self, prompt: str):
        language_embeds = self._load_language_embeds(str(prompt))
        prompt_len = int(language_embeds.shape[0])
        self._evict_stale_runtimes(prompt_len)
        runtime = self._spec_runtime_by_prompt_len.get(prompt_len)
        if runtime is None:
            checkpoint = dict(self._base_weights)
            checkpoint["language_embeds"] = language_embeds
            runtime = self._spec_runtime_factory(
                checkpoint=checkpoint,
                draft_checkpoint=self._draft_checkpoint,
                num_views=self._num_views,
                chunk_size=self._chunk_size,
            )
            self._spec_runtime_by_prompt_len[prompt_len] = runtime
        self._maybe_copy_language_embeds(runtime, language_embeds)
        return runtime

    def start_session(self, prompt: str) -> SpecTritonRuntimeSession:
        prompt = str(prompt)
        return SpecTritonRuntimeSession(
            prompt=prompt,
            runtime=self.get_runtime(prompt),
            draft_runtime=self.get_spec_runtime(prompt),
        )
