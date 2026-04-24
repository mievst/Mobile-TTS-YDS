import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

from package.utils import summarize_outputs, save_outputs
from package.model_assets import (
    get_config, text_projection, text_embedding_lookup, 
    talker_codec_embedding_lookup, cp_codec_embedding_lookup, 
    build_prefill_embedding, softmax
)

LOGGER = logging.getLogger(__name__)


def sample_talker_token(
    logits: np.ndarray,
    assets: dict[str, Any],
    previous_tokens: list[int],
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    min_new_tokens: int,
    step: int
) -> int:
    config = assets["config"]
    talker_vocab_size = int(get_config(config, "talker.vocab_size"))
    cp_vocab_size = int(get_config(config, "code_predictor.vocab_size"))
    codec_eos_token_id = int(get_config(config, "talker.codec_eos_token_id"))
    
    x = logits.reshape(-1)[-talker_vocab_size:].astype(np.float32, copy=True)

    for token in previous_tokens:
        if 0 <= token < len(x):
            if x[token] > 0: x[token] /= repetition_penalty
            else: x[token] *= repetition_penalty

    for i in range(cp_vocab_size, talker_vocab_size):
        if i != codec_eos_token_id:
            x[i] = -np.inf

    if step < min_new_tokens and 0 <= codec_eos_token_id < len(x):
        x[codec_eos_token_id] = -np.inf

    if temperature > 0:
        x = x / temperature

    if top_k > 0 and top_k < len(x):
        idx = np.argpartition(-x, top_k)[:top_k]
        mask = np.full_like(x, -np.inf)
        mask[idx] = x[idx]
        x = mask

    probs = softmax(x)
    return int(np.random.choice(len(probs), p=probs))


def sample_cp_token(logits: np.ndarray, assets: dict[str, Any], temperature: float, top_k: int) -> int:
    config = assets["config"]
    cp_vocab_size = int(get_config(config, "code_predictor.vocab_size"))
    x = logits.reshape(-1)[-cp_vocab_size:].astype(np.float32, copy=True)

    if top_k > 0 and top_k < len(x):
        idx = np.argpartition(-x, top_k)[:top_k]
        mask = np.full_like(x, -np.inf)
        mask[idx] = x[idx]
        x = mask

    if temperature > 0:
        x = x / temperature

    probs = softmax(x)
    return int(np.random.choice(len(probs), p=probs))


def stack_prefill_kv(prefill_map: dict[str, Any], num_layers: int) -> tuple[np.ndarray, np.ndarray]:
    keys = []
    values = []
    for layer in range(num_layers):
        keys.append(prefill_map[f"present_key_{layer}"])
        values.append(prefill_map[f"present_value_{layer}"])
    return np.stack(keys, axis=0).astype(np.float32, copy=False), np.stack(values, axis=0).astype(np.float32, copy=False)


def build_position_ids_3d(step_position: int) -> np.ndarray:
    return np.array([[[step_position]], [[step_position]], [[step_position]]], dtype=np.int64)


def run_code_predictor_groups(
    cp_session: ort.InferenceSession,
    last_hidden: np.ndarray,
    group0_token: int,
    assets: dict[str, Any],
    temperature: float,
    top_k: int,
    save_dir: Optional[Path] = None,
    timestep_idx: Optional[int] = None
) -> list[int]:
    config = assets["config"]
    cp_num_layers = int(get_config(config, "code_predictor.num_hidden_layers"))
    cp_num_kv_heads = int(get_config(config, "code_predictor.num_key_value_heads"))
    cp_head_dim = int(get_config(config, "code_predictor.head_dim"))
    
    group0_embed = talker_codec_embedding_lookup(group0_token, assets)
    codes: list[int] = []
    cp_past_keys = np.empty((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
    cp_past_values = np.empty((cp_num_layers, 1, cp_num_kv_heads, 0, cp_head_dim), dtype=np.float32)
    prev_cp_token = None

    for group_idx in range(1, 16):
        if group_idx == 1:
            inputs_embeds = np.stack([last_hidden, group0_embed], axis=0)[None, :, :].astype(np.float32)
        else:
            assert prev_cp_token is not None
            prev_cp_embed = cp_codec_embedding_lookup(group_idx - 2, prev_cp_token, assets)
            inputs_embeds = prev_cp_embed[None, None, :].astype(np.float32)

        generation_steps = np.array([group_idx - 1], dtype=np.int64)
        
        feed = {
            "inputs_embeds": inputs_embeds,
            "generation_steps": generation_steps,
            "past_keys": cp_past_keys,
            "past_values": cp_past_values,
        }

        output_names = [x.name for x in cp_session.get_outputs()]
        outputs = cp_session.run(output_names, feed)
        cp_map = dict(zip(output_names, outputs))

        cp_logits = cp_map["logits"]
        cp_token = sample_cp_token(cp_logits, assets, temperature=temperature, top_k=top_k)
        codes.append(cp_token)
        prev_cp_token = cp_token

        cp_past_keys = cp_map["present_keys"].astype(np.float32, copy=False)
        cp_past_values = cp_map["present_values"].astype(np.float32, copy=False)

        if save_dir is not None and timestep_idx is not None:
            save_outputs(save_dir, f"cp_t{timestep_idx}_g{group_idx}", output_names, outputs)

    return codes


def generate_codes(
    prefill_session: ort.InferenceSession,
    decode_session: ort.InferenceSession,
    cp_session: ort.InferenceSession,
    token_ids: np.ndarray,
    assets: dict[str, Any],
    language: str,
    speaker_id: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    min_new_tokens: int,
    save_dir: Optional[Path] = None
) -> np.ndarray:
    config = assets["config"]
    num_layers = int(get_config(config, "talker.num_hidden_layers"))
    codec_eos_token_id = int(get_config(config, "talker.codec_eos_token_id"))

    inputs_embeds, trailing_text_hidden = build_prefill_embedding(
        token_ids=token_ids,
        assets=assets,
        language=language,
        speaker_id=speaker_id
    )
    
    prefill_len = inputs_embeds.shape[1]
    attention_mask = np.ones((1, prefill_len), dtype=np.int64)
    position_ids = np.zeros((3, 1, prefill_len), dtype=np.int64)
    
    for ax in range(3):
        position_ids[ax, 0, :] = np.arange(prefill_len, dtype=np.int64)

    LOGGER.info("Running talker_prefill")
    prefill_output_names = [x.name for x in prefill_session.get_outputs()]
    
    prefill_outputs = prefill_session.run(
        output_names=prefill_output_names,
        input_feed={
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    )
    
    prefill_map = summarize_outputs("prefill", prefill_output_names, prefill_outputs)
    
    if save_dir is not None:
        save_outputs(save_dir, "prefill", prefill_output_names, prefill_outputs)

    logits = prefill_map["logits"].astype(np.float32, copy=False)
    hidden_states = prefill_map["hidden_states"].astype(np.float32, copy=False)
    past_keys, past_values = stack_prefill_kv(prefill_map, num_layers=num_layers)
    
    cfg_pad_token_id = int(get_config(config, "tts.tts_pad_token_id"))
    tts_pad_proj = text_projection(text_embedding_lookup(cfg_pad_token_id, assets), assets)
    generated_codes = []
    generated_group0_tokens = []

    for step in range(max_new_tokens):
        group0_token = sample_talker_token(
            logits=logits,
            assets=assets,
            previous_tokens=generated_group0_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            step=step,
        )

        LOGGER.debug("step=%d group0_token=%d", step, group0_token)

        if group0_token == codec_eos_token_id:
            LOGGER.info("Reached codec EOS at step %d", step)
            break

        generated_group0_tokens.append(group0_token)
        last_hidden = hidden_states[0, -1, :].astype(np.float32, copy=False)

        cp_codes = run_code_predictor_groups(
            cp_session=cp_session,
            last_hidden=last_hidden,
            group0_token=group0_token,
            assets=assets,
            temperature=temperature,
            top_k=top_k,
            save_dir=save_dir,
            timestep_idx=step,
        )

        codes_this_step = [group0_token] + cp_codes
        generated_codes.append(codes_this_step)
        next_input = talker_codec_embedding_lookup(group0_token, assets).copy()
        
        for g in range(1, 16):
            cp_embed = cp_codec_embedding_lookup(g - 1, codes_this_step[g], assets)
            next_input[: cp_embed.shape[0]] += cp_embed

        if step < trailing_text_hidden.shape[0]:
            next_input += trailing_text_hidden[step]
        else:
            next_input += tts_pad_proj

        next_input_embeds = next_input[None, None, :].astype(np.float32)
        new_len = prefill_len + step + 1
        decode_attention_mask = np.ones((1, new_len), dtype=np.int64)
        decode_position_ids = build_position_ids_3d(prefill_len + step)

        LOGGER.info("Running talker_decode step %d/%d", step + 1, max_new_tokens)

        decode_output_names = [x.name for x in decode_session.get_outputs()]
        
        decode_outputs = decode_session.run(
            output_names=decode_output_names,
            input_feed={
                "inputs_embeds": next_input_embeds,
                "attention_mask": decode_attention_mask,
                "position_ids": decode_position_ids,
                "past_keys": past_keys.astype(np.float32),
                "past_values": past_values.astype(np.float32),
            }
        )
        
        decode_map = summarize_outputs(f"decode_step_{step + 1}", decode_output_names, decode_outputs)
        
        if save_dir is not None:
            save_outputs(save_dir, f"decode_step_{step + 1}", decode_output_names, decode_outputs)

        logits = decode_map["logits"].astype(np.float32, copy=False)
        hidden_states = decode_map["hidden_states"].astype(np.float32, copy=False)
        past_keys = decode_map["present_keys"].astype(np.float32, copy=False)
        past_values = decode_map["present_values"].astype(np.float32, copy=False)

    T = len(generated_codes)
    LOGGER.info("Generated timesteps: %d", T)

    result = np.zeros((1, 16, T), dtype=np.int64)
    for t in range(T):
        for g in range(16):
            result[0, g, t] = generated_codes[t][g]

    return result


def run_vocoder(vocoder_session: ort.InferenceSession, codes: np.ndarray) -> np.ndarray:
    vocoder_input_name = vocoder_session.get_inputs()[0].name
    LOGGER.info("Running vocoder with codes shape=%s", codes.shape)
    output_names = [x.name for x in vocoder_session.get_outputs()]
    outputs = vocoder_session.run(output_names, {vocoder_input_name: codes.astype(np.int64, copy=False)})
    waveform = outputs[0]
    LOGGER.info("Vocoder waveform shape=%s dtype=%s", waveform.shape, waveform.dtype)
    return waveform
