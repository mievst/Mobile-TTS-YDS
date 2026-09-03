# Запуск квантованной Int8 TTS на телефоне

Репозиторий решения для эффективного запуска **Qwen3-TTS-12Hz-0.6B-CustomVoice** на Android-устройстве. Основной фокус — исследование **квантизации** (INT8/INT4/FP16) для значительного уменьшения размера модели и ускорения инференса при сохранении качества генерации, с собственным ONNX orchestration layer.

## 🎬 Демо

Инференс Qwen3-TTS «из текста в аудио» прямо на Android-устройстве:

[`assets/demo-video.mp4`](assets/demo-video.mp4)

Примеры синтеза — в [`assets/qwen-tts-audio/`](assets/qwen-tts-audio/). 👂 Прослушать наглядно, как квантизация влияет на качество:

| Вариант | Аудио |
|---|---|
| Базовый (оригинал) | [`original.wav`](assets/qwen-tts-audio/original.wav) |
| `embeddings` FP16 | [`Embeddings FP16.wav`](assets/qwen-tts-audio/Embeddings%20FP16.wav) |
| `talker_decode` до INT4 | [`talker_decode до int4.wav`](assets/qwen-tts-audio/talker_decode%20до%20int4.wav) |
| `talker_prefill` INT8 + `talker_decode` INT4 | [`talker_prefill int8 + talker_decode int 4.wav`](assets/qwen-tts-audio/talker_prefill%20int8%20%2B%20talker_decode%20int%204.wav) |
| Артефакт наивной квантизации «в лоб» (модель «рассмешили») | [`смех.wav`](assets/qwen-tts-audio/смех.wav), [`смех2.wav`](assets/qwen-tts-audio/смех2.wav) |
| **Финальный результат** | [`итоговая.wav`](assets/qwen-tts-audio/итоговая.wav) |

## 📊 Результаты

Сводная таблица (вариант квантизации / размер / латенси / DNSMOS / WER-CER) и подробные замеры — в **[`RESULTS.md`](RESULTS.md)**.

Ключевой итог: финальный вариант квантизации даёт **ускорение ~14×** (latency `182 s → ~12 s`) и **сокращение модели более чем в 2 раза** при незначительной потере DNSMOS (`3.173 → 3.0`).

## 👥 Состав команды и зоны ответственности

| Участник | Зона ответственности |
|---|---|
| [**Миронов Арсений**](https://github.com/Napkin-AI) | Квантизация (INT8/INT4/FP16), замеры скорости, автоматизация бенчмарков латенси, профилирование и точечная оптимизация слоёв, финальные замеры |
| [**Викулов Максим**](https://github.com/mavikulov) | Разработка Android-приложения, конвертация модели в ONNX, имплементация ONNX orchestration layer, полный on-device пайплайн, code cleanup |
| [**Евстифеев Михаил**](https://github.com/mievst) | Методология оценки качества (DNSMOS, WER/CER), подготовка тестовых данных, планирование экспериментов, организация репозитория, итоговый отчёт |

Распределение активностей по неделям экспериментального плана — см. [`Report.pdf`](Report.pdf) и презентацию [`assets/Запуск квантованной Int8 TTS на телефоне..pptx`](assets/Запуск%20квантованной%20Int8%20TTS%20на%20телефоне..pptx).

## 📁 Структура проекта

| Каталог / файл | Назначение |
|---|---|
| `QwenMobileTTS/` | **Android-приложение** (пайплайн, orchestration, инференс) |
| `python/` | Конвертация, квантизация и запуск модели (ONNX) |
| `benchmark/` | Оценка качества (DNSMOS, WER/CER) и производительности |
| `assets/` | Демо-видео, примеры аудио, финальная презентация |
| `RESULTS.md` | Сводные результаты экспериментов |
| `ARCHITECTURE.md` | Архитектура мобильного приложения |
| `Report.pdf` | Планирование и первый эксперимент |
| `requirements.txt`, `pyproject.toml` | Python-зависимости / настройки (uv) |

## 🚀 Быстрый старт

```bash
git clone https://github.com/mievst/Mobile-TTS-YDS.git
cd Mobile-TTS-YDS

# установка зависимостей
pip install -r requirements.txt
# или через uv
uv sync
```

## 📚 Подробнее

- Вся информация о мобильном приложении — [`QwenMobileTTS/README.md`](QwenMobileTTS/README.md)
- Вся информация о бенчмарках — [`benchmark/README.md`](benchmark/README.md)

## 📌 Статус проекта

Проект **завершён** по экспериментальному плану (23.03.2026 – 25.04.2026): ONNX-конвертация, квантизация, on-device демо и оценка качества доведены до финального варианта. Результаты зафиксированы в презентации и `RESULTS.md`. Репозиторий содержит все артефакты, кроме формальных `WER/CER` и on-device latency замеров — см. «[Что не попало в репозиторий](RESULTS.md#что-не-попало-в-репозиторий)».
