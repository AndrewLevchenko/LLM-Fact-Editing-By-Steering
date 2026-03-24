from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

load_dotenv() # для загрузки токена из SECRETS при работе в KAGGLE

@dataclass(slots=True)
class ModelConfig:
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    hf_token: str | None = None

    device_map: str | dict[str, Any] = "auto"
    trust_remote_code: bool = False

    torch_dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False # я таки куплю RTX3090

    use_fast_tokenizer: bool = False
    padding_side: str = "left" # при батчевой работе удобнее найти последний токен
    local_files_only: bool = False # первый запуск - всегда c False (ну или иметь скачанную модель)


def _resolve_hf_token(config: ModelConfig) -> str | None:
    if config.hf_token:
        return config.hf_token

    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    key = dtype_name.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported torch_dtype='{dtype_name}'. "
            f"Supported: {sorted(mapping.keys())}"
        )

    return mapping[key]


def _build_quantization_config(config: ModelConfig) -> BitsAndBytesConfig | None:
    if config.load_in_8bit and config.load_in_4bit:
        raise ValueError("Only one of load_in_8bit / load_in_4bit can be True.")

    # восьмибитный конфиг - занимает чуть в районе 9 Гб видеопамяти
    if config.load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)

    # четырехбитный конфиг - бедность не порок - занимает чуть меньше 5 Гб видеопамяти
    if config.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(config.torch_dtype),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    return None


def load_tokenizer(config: ModelConfig) -> AutoTokenizer:
    token = _resolve_hf_token(config)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        token=token,
        use_fast=config.use_fast_tokenizer,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )

    tokenizer.padding_side = config.padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_model(config: ModelConfig) -> AutoModelForCausalLM:
    token = _resolve_hf_token(config)
    quantization_config = _build_quantization_config(config)

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": config.model_name,
        "token": token,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "dtype": config.torch_dtype,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = config.torch_dtype
    if token is not None:
        from huggingface_hub import login
        login(token=token)

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    return model


def load_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = load_tokenizer(config)
    model = load_model(config)

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if getattr(model.config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer


@torch.inference_mode()
def tokenize_batch(
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str | torch.device | None = None,
    padding: bool = True,
    truncation: bool = True,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}

    return encoded


@torch.inference_mode()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    skip_special_tokens: bool = True,
) -> str:


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated[0]
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[input_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens).strip()


@torch.inference_mode()
def instruct_generate_text(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompt: str,
        system_prompt: str="You are helpful assistant.",
        max_new_tokens: int = 50,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        skip_special_tokens: bool = True,
) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user","content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt",return_dict=True).to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    output_ids = generated[0]
    input_len = inputs["input_ids"].shape[1]
    new_tokens = output_ids[input_len:]

    return tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens).strip()
