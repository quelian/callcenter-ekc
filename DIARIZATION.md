# Разделение ролей по голосу

`pyannote` удален. В системе два локальных пути:

- `heavy_diarization` (основной в режиме `Max Quality RU`): `speechbrain` speaker embeddings + агломеративная кластеризация.
- `fallback_light`: легкий pipeline на `resemblyzer`.

Режимов облака нет: пайплайн полностью локальный и рассчитан на русскую речь.

## Heavy path

1. Из аудио выделяются голосовые регионы.
2. Строятся скользящие окна по регионам.
3. Для окон считаются embeddings (`speechbrain/spkrec-ecapa-voxceleb`).
4. Окна кластеризуются в 2 спикера.
5. Метки переносятся на ASR-реплики по максимальному overlap таймкодов.
6. Кластеры маппятся в **Оператор / Заявитель** с русскими priors, затем сглаживаются.

Если heavy путь не срабатывает (ошибка/таймаут/мало окон), автоматически включается `fallback_light`.

## Fallback path (light)

1. Whisper дает сегменты речи с таймкодами.
2. Для длинных сегментов строятся фиксированные голосовые окна.
3. По окнам считаются speaker embeddings (`resemblyzer`) + energy-фильтр.
4. Эмбеддинги кластеризуются в 2 спикера.
5. Кластеры подписываются как **Оператор / Заявитель** через скриптовые якоря и score.

Если и light путь неудачен, применяется усиленный текстовый fallback.

## Диагностика качества

В UI показываются:

- `confidence` (`high` / `medium` / `low`);
- `diarization_path` (`heavy_diarization` / `fallback_light`);
- `split_mode` (`native` / `forced`);
- метрики (`cluster_quality`, `inter`, `intra`, `window_count`, `valid_segments`);
- latency этапов (`asr_seconds`, `diarization_seconds`, `merge_seconds`).

## ASR профиль и влияние на роли

Качество ролей сильно зависит от качества транскрибации.

- `medium_ru` — средний режим (`medium`);
- `ideal_ru` — максимальное качество (`large-v3`, Systran) с принудительным `language=ru`;
- `ultima_ru` — **Ultima**: fine-tuned `whisper-large-v3-russian` с Hugging Face (см. `CALLQA_WHISPER_MODEL_ULTIMA`); **тот же premium decode**, что у `ideal_ru` (см. `transcription.py`).

## Пост-редактор (medium_ru / ideal_ru / ultima_ru)

Для `medium_ru`, `ideal_ru` и `ultima_ru` доступен один и тот же локальный пост-редактор:

1. deterministic-нормализация текста;
2. нейросетевая пунктуация (`deepmultilingualpunctuation`);
3. логическая коррекция ролей (`turn-taking`, операторский пролог, сглаживание).

Безопасность:

- есть timeout пост-редактора;
- есть anti-drift откат, если редактор слишком меняет текст;
- voice-role шаг не отключается даже при деградации post-edit.

Рекомендуемая комбинация для большинства звонков:

- модель `medium`
- профиль `medium_ru`
- `compute_type=int8`
- при необходимости включите `Пост-редактор (RU, local)` — быстрее, чем `ideal_ru`, но текст и роли аккуратнее

Для максимального качества ролей:

- профиль `ideal_ru` или `ultima_ru` (в Streamlit — **«Максимум»** / **«Ultima»**, heavy по умолчанию включён)
- включенный `Max Quality RU (heavy)` там, где применимо
- включенный `Пост-редактор (RU, local)`
- таймаут heavy diarization 180-240 секунд
