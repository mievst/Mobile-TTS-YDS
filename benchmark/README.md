# Qwen3-TTS Benchmark

Benchmarks for Qwen3-TTS text-to-speech models, comparing vLLM-Omni streaming serving against HuggingFace Transformers offline inference.

## Prerequisites

```bash
pip install matplotlib aiohttp soundfile numpy tqdm
pip install qwen_tts
```

## Quick Start

Run the full benchmark (HF baseline) with a single command:

```bash
cd benchmarks/qwen3-tts
bash run_benchmark.sh
```

Results (JSON + PNG plots) are saved to `results/`.

### Common options

```bash


# Use a different model (e.g. 1.7B)
MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice bash run_benchmark.sh

# Custom GPU, prompt count
GPU_DEVICE=1 NUM_PROMPTS=20 bash run_benchmark.sh
```

## Manual Steps

### Run HuggingFace baseline

```bash
python benchmarks/qwen3-tts/bench_tts_hf.py \
    --model "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" \
    --num-prompts 0 \
    --gpu-device 99 \
    --result-dir results/
```

### 4) Generate comparison plots

```bash
python benchmarks/qwen3-tts/plot_results.py \
    --results results/bench_hf_transformers_*.json \
    --labels "hf_transformers" \
    --output results/comparison.png
```

## Metrics

- **TTFP (Time to First Audio Packet)**: Time from request to first audio chunk (streaming latency)
- **E2E (End-to-End Latency)**: Total time from request to complete audio response
- **RTF (Real-Time Factor)**: E2E latency / audio duration. RTF < 1.0 means faster-than-real-time synthesis
- **Throughput**: Total audio seconds generated per wall-clock second
