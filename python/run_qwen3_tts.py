from __future__ import annotations

import time
import random
import logging
import argparse
from pathlib import Path

import numpy as np

from package.utils import setup_logging, load_session, log_session_io, save_wav, get_model_size
from package.model_assets import load_tokenizer, load_assets, log_assets, build_custom_voice_prompt_ids, get_embeddings_size
from package.inference_engine import generate_codes, run_vocoder


LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Python reference runner")
    parser.add_argument("--model-dir", required=True, help="Path to ONNX model directory")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--instruct", default=None, help="Optional instruction prompt")
    parser.add_argument("--language", default="auto", help="Language code")
    parser.add_argument("--speaker-id", type=int, default=-1, help="Speaker id")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max audio timesteps")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--min-new-tokens", type=int, default=2, help="Suppress EOS for N steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", default=None, help="Output/debug directory")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--log-file", default=None, help="Log file path")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else model_dir / "_debug_outputs"
    save_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "prefill": model_dir / "talker_prefill.onnx",
        # "prefill": Path("/Users/ruaoccj/Qwen3-TTS-Android/scripts/quantized/float16/talker_prefill_fp16.onnx"),
        # "decode": model_dir / "talker_decode.onnx",
        "decode": Path("/Users/ruaoccj/Qwen3-TTS-Android/scripts/quantized/int4/talker_decode_quantized_int4.onnx"),
        "cp": model_dir / "code_predictor.onnx",
        "vocoder": model_dir / "vocoder.onnx"
    }
    
    for path in model_files.values():
        LOGGER.info(f"{path} model = {get_model_size(path)}")
        if not path.exists():
            LOGGER.error("Missing model file: %s", path)
            return

    LOGGER.info("Loading sessions on CPU")
    t0 = time.perf_counter()
    
    prefill_session = load_session(model_files["prefill"])
    decode_session = load_session(model_files["decode"])
    cp_session = load_session(model_files["cp"])
    vocoder_session = load_session(model_files["vocoder"])
    LOGGER.info("All sessions loaded in %.2fs", time.perf_counter() - t0)

    log_session_io("talker_prefill", prefill_session)
    log_session_io("talker_decode", decode_session)
    log_session_io("code_predictor", cp_session)
    log_session_io("vocoder", vocoder_session)

    tokenizer = load_tokenizer(model_dir)
    
    assets = load_assets(Path("/Users/ruaoccj/Qwen3-TTS-Android/scripts/quantized/embeddings_fp16"))
    # assets = load_assets(Path("/Users/ruaoccj/Qwen3-TTS-Android/models/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX/embeddings"))
    # log_assets(assets)
    emb_size = get_embeddings_size(assets)
    LOGGER.info(f"Embeddings size = {emb_size} MB")
    

    token_ids = build_custom_voice_prompt_ids(
        tokenizer=tokenizer,
        text=args.text,
        instruct=args.instruct,
    )

    codes = generate_codes(
        prefill_session=prefill_session,
        decode_session=decode_session,
        cp_session=cp_session,
        token_ids=token_ids,
        assets=assets,
        language=args.language,
        speaker_id=args.speaker_id,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        min_new_tokens=args.min_new_tokens,
        save_dir=save_dir,
    )

    LOGGER.info("Generated codes shape=%s", codes.shape)
    np.savez_compressed(save_dir / "codes.npz", codes=codes)

    waveform = run_vocoder(vocoder_session, codes)
    wav_path = save_dir / "output.wav"
    save_wav(wav_path, waveform, sample_rate=24000)
    np.savez_compressed(save_dir / "waveform.npz", waveform=waveform)

    LOGGER.info("Saved waveform to %s", wav_path)
    LOGGER.info("Pipeline finished successfully")


if __name__ == "__main__":
    main()
