# Результаты экспериментов

Сводные результаты по квантизации **Qwen3-TTS-12Hz-0.6B-CustomVoice** для запуска на мобильном устройстве (Android, ONNX orchestration layer).

Метрики:

- **size** — размер модели (bundle).
- **latency** — средняя сквозная (`e2e`) задержка синтеза и `rtf` (real-time factor, `latency / duration`).
- **DNSMOS** — объективное качество речи (0–5, чем выше тем лучше), `arxiv:2010.15258`.
- **WER/CER** — качество распознавания синтеза (RU/EN), пока **не опубликовано** в репозитории.

> ⚠️ WER/CER (ASR-оценка через Whisper) и **on-device (Android) латенси** в репозиторий не выгружены — см. раздел «[Что не попало в репозиторий](#что-не-попало-в-репозиторий)».

## 1. Итоговая таблица

Сводка по вариантам квантизации. `latency`/`rtf`/`DNSMOS` — серверные замеры (см. подробности ниже).

| Вариант квантизации | Размер модели | On-device latency | DNSMOS | WER/CER RU | WER/CER EN |
|---|---|---|---|---|---|
| Базовая (FP32, transformers) | ~5.7 GB (ONNX FP32) | — | 3.173 | — | — |
| `talker_decode` → INT4 | — (меньше) | — | 3.11 | — | — |
| `talker_prefill` INT8 + `talker_decode` INT4 | — | — | 3.03 | — | — |
| `embeddings` FP16, остальное INT8 | — | — | 3.0 | — | — |
| **Финальный**: `embeddings` FP16, `talker_decode` INT4 (extra options), остальные INT8 | **>2× меньше** | — | 3.0 | — | — |

> «—» — данные не опубликованы в репозитории.

## 2. Детальные замеры (PPT финальной презентации)

Серверные замеры вариантов квантизации ONNX (метрики: `duration`, `mean e2e latency`, `mean rtf`, `DNSMOS`).

| Эксперимент | duration (s) | mean latency (s) | mean rtf | DNSMOS |
|---|---|---|---|---|
| **Базовая модель** (бейзлайн) | 1873 | 182.22 | 10.8 | **3.173** |
| `talker_decode` → INT4 | 121 | 12.36 | 3.12 | **3.11** |
| `talker_prefill` INT8 + `talker_decode` INT4 | 135 | 16.78 | 3.13 | **3.03** |
| `embeddings` FP16, остальное INT8 | 127 | 15.25 | 3.38 | **3.0** |
| **Финальный**: `embeddings` FP16, `talker_decode` INT4 (extra opts), остальные INT8 | 121 | **12.36** | **3.12** | **3.0** |

Вывод из презентации: финальный вариант даёт **ускорение в ~14×** (latency `182.22 s → 12.36 s`) и **уменьшение модели более чем в 2 раза** при минимальной потере DNSMOS (`3.173 → 3.0`).

## 3. Ранние замеры (Report.pdf)

Первые замеры скорости оригинальной модели (transformers-бэкенд `qwen_tts` + `vllm-omni`), из `Report.pdf` (начало экспериментального плана).

Окружение: CPU — Intel Core i5-8300H; GPU — NVIDIA RTX A4000 (сервер YSDA `beleriand`). GPU-замеры могли быть неточны из-за загрузки сервера.

| Device | duration (s) | mean e2e latency (s) | mean rtf |
|---|---|---|---|
| CPU | 2461 | 245.8 | 62.094 |
| GPU (transformers) | 146 | 14.5 | 3.679 |
| GPU (vllm-omni) | 119 | 11.9 | 3.328 |

## Что не попало в репозиторий

- **WER/CER (RU/EN)** — бенчмарк `benchmark/bench_tts_quality.py` (DNSMOS + Whisper WER/CER) реализован, прогоны описаны в `benchmark/README.md`, но **результаты не сохранены** (`benchmark/tmp_results/` отсутствует, исключено `.gitignore`-ом).
- **On-device (Android) латенси** — демо работает (видео `assets/demo-video.mp4`), формальные замеры на устройстве не выгружены.
