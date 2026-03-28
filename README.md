# Call Center QA Analyzer (MVP)

Локальное приложение для транскрибации звонков и QA-оценки диалогов колл-центра.

**Текущая версия:** Beta 1.1 (24.03.2026) — см. [CHANGELOG.md](CHANGELOG.md).

- **Windows (скачать с GitHub, cmd, запуск):** [docs/WINDOWS.md](docs/WINDOWS.md)  
- **GitHub:** первый push и синхронизация — [docs/GITHUB_SETUP.md](docs/GITHUB_SETUP.md)
- **Пропадает начало разговора после ожидания?** Разбор причин и переменные `.env`: [docs/ASR_START_ANALYSIS.md](docs/ASR_START_ANALYSIS.md)
- **Роли и шаблоны оператора** («спасибо за обращение» и т.п.): [docs/CALL_CENTER_ROLE_ANCHORS.md](docs/CALL_CENTER_ROLE_ANCHORS.md)

## Что умеет

- Загружает аудиофайлы (`wav/mp3/m4a/ogg/flac`)
- Транскрибирует звонки локально (`faster-whisper`)
- Поддерживает RU ASR-профили: `medium_ru` / `ideal_ru` / `ultima_ru` (**Ultima** — `whisper-large-v3-russian` с Hugging Face)
- **LLM пост-редактор**: по умолчанию **встроенная** лёгкая [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) через `transformers` (скачивание с Hugging Face при первом запуске, ~3.5 ГБ); опционально внешний OpenAI-compatible API (Ollama, LM Studio)
- Опционально **локальный пунктуатор** (`deepmultilingualpunctuation`) без LLM
- Оценивает диалоги локально (`CallQualityEvaluator` в `transcription.py`) и/или облаком (`llm_cloud_eval.py`)
- Делит роли `Оператор/Заявитель` через голосовые кластеры + гибридную коррекцию
- Выделяет плюсы, минусы и итоговый рейтинг
- Экспортирует текстовый отчёт в папку **`результаты/`** в корне проекта (создаётся автоматически; в `.gitignore` не попадает в git)

### Имя оператора

- **Одиночный файл:** вручную в списке «Оператор» или пункт **«Автоопределение»** — тогда имя задаётся **только облачной моделью** (Yandex API, переключатель в боковой панели). Локальный поиск имени оператора по тексту транскрипта **не используется**. Без облака и без ручного выбора анализ не запустится.
- **Егор, Артем, Даша** (зона повышенного внимания): критерии те же; при ручном выборе в облаке — более благожелательная трактовка пограничных случаев в промпте; после оценки **+1** к каждому из 4 критериев (максимум 10), пересчёт итога, меньше и мягче формулировки в «минусах». См. `evaluation_leniency.py`, `operator_staff.OPERATORS_LENIENCY_FOCUS`.
- **Пакетная обработка:** оператор **обязателен** — выберите одно имя для всего пакета (все записи считаются звонками одного оператора). Автоопределения в пакете нет. Имя **каждого** файла должно соответствовать шаблону `ДД-ММ-ГГГГ_телефон_ММ-СС.ext` (см. интерфейс); иначе файл отклоняется и в пакет не попадает.
- **Один файл с «нестандартным» именем:** можно вручную ввести дату, телефон и ожидание одной строкой (**ММ:СС** или **ММ-СС**, либо только секунды числом) — они уйдут в Google Таблицу и зададут обрезку записи, как суффикс в стандартном имени.

### Имя заявителя

По-прежнему определяется эвристиками и/или облачной оценкой по транскрипту (логика не отключалась).

## Быстрый старт

Команды вводите в **терминале**, не внутри Python (`>>>`).

**Windows** — пошагово: **[docs/WINDOWS.md](docs/WINDOWS.md)** (ZIP с GitHub или `git clone`, `venv`, `pip`, `run_web_gui.bat`).

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run web_gui.py
```

**Админ-панель** (сайдбар): ключи Yandex и таблица Google — только после пароля. Для включения задайте `ADMIN_PANEL_PASSWORD` в `.env`; без этого админ-панель отключена. Сохранение идёт в `.env`.

Для разработки:

```bash
pip install -e .[dev]
pytest
ruff check app_config.py analysis_service.py batch_state.py env_store.py state_paths.py web_gui.py transcription.py sheets_export.py llm_cloud_eval.py tests
```

### Интерфейс загрузки файлов (русский + 5 файлов на страницу)

Виджет `st.file_uploader` в Streamlit по умолчанию на английском и показывает по 3 файла в списке. При **`streamlit run web_gui.py`** скрипт сам патчит JS в **том же venv**, из которого запущен Streamlit. Если в браузере всё ещё английский или по 3 строки — **жёстко обновите страницу** (Ctrl+Shift+R).

Вручную (например после `pip install -U streamlit`):

```bash
python scripts/patch_streamlit_file_uploader_ru.py
```

## Рекомендованные ASR настройки (8GB RAM)

- В **Streamlit** по умолчанию выбрано **Максимум** (Large-v3 + `ideal_ru` + тяжёлая диаризация); **Ultima** — та же мощная связка decode + heavy, но модель RU fine-tuned с HF (`CALLQA_WHISPER_MODEL_ULTIMA`); для слабых ПК — **Стандарт**.
- `ASR профиль: medium_ru` + модель `medium` (средний режим)
- `ASR профиль: ideal_ru` + модель `large-v3` (максимум, Systran)
- `ASR профиль: ultima_ru` + модель из `CALLQA_WHISPER_MODEL_ULTIMA` (по умолчанию `bzikst/faster-whisper-large-v3-russian`)
- Для максимальной стабильности ролей — пресет **Максимум** (там heavy по умолчанию) или heavy в CLI для нужного профиля.
- В Streamlit по умолчанию включён **LLM пост-редактор** (встроенная модель с Hugging Face, отдельное приложение не нужно). На CPU первый прогон может быть долгим — увеличьте таймаут в боковой панели. Локальный пунктуатор — запасной вариант без LLM.
- Внешний API: в UI выберите «Внешний API» или задайте `LLM_POST_EDIT_BASE_URL` / `LLM_POST_EDIT_MODEL` в `.env` (Ollama: `ollama serve` + `ollama pull …`).

CLI пример (встроенный LLM, первый раз — загрузка модели с HF):

```bash
python transcription.py /path/to/call.wav --asr-profile medium_ru --compute-type int8 \
  --llm-backend embedded --llm-post-edit-timeout-seconds 300
```

Внешний API (нужны переменные `LLM_POST_EDIT_BASE_URL` и `LLM_POST_EDIT_MODEL`):

```bash
python transcription.py /path/to/call.wav --asr-profile medium_ru --llm-backend remote
```

Только локальный пунктуатор (без LLM):

```bash
python transcription.py /path/to/call.wav --asr-profile medium_ru --enable-post-edit --post-edit-timeout-seconds 60
```

## Настройка критериев

- **Локальная эвристика** — класс `CallQualityEvaluator` в `transcription.py` (скрипт, речь, консультация, вовлечённость).
- **Облако (Yandex)** — промпт и JSON-ответ в `llm_cloud_eval.py`.
- **Текстовые стандарты ЕКЦ** — для сверки: файл `Стандарты диалогов ЕКЦ` в корне репозитория.

## Примечания

- Имена записей в формате `дата_телефон_MM-SS.ext` (последний фрагмент после `_` — **минуты-секунды ожидания**): распознавание автоматически начинается с этого момента. Ручной `--skip-first-seconds` это перекрывает. В Streamlit для парсинга используется **исходное имя загрузки**, а не временный файл на диске.
- Разделение ролей работает в режиме voice-first с fallback на текстовые правила.
- В heavy-режиме основной путь: `heavy_diarization`, fallback: легкий embedding-кластеринг.
- **LLM пост-редактор**: встроенный путь — `llm_embedded_deepseek.py` (локальная генерация); HTTP — `llm_post_edit.py`. Ответ с блоками `<<<FULL>>>` / `<<<ROLES>>>`, проверка на «уезд» текста. При ошибке/таймауте — fallback на следующий шаг.
- **Локальный пост-редактор** (если включён): пунктуация **по сегментам ASR** (длинные сегменты — батчами по предложениям), заглавные после `.!?…`, эвристики ролей только при `low`/`medium` confidence диаризации. Плоский текст и роли синхронизируются из одних и тех же сегментов.
- Для диагностики качества ролей смотрите `confidence` и `role_diagnostics` в UI/отчете. Подробнее о диаризации: [DIARIZATION.md](DIARIZATION.md).

## Облачный режим оценки (Yandex AI Studio)

В сайдбаре Streamlit переключите **«Режим оценки»** → **«Облачный AI (Яндекс)»**.

Вместо regex-эвристик LLM:
- определяет имя оператора и заявителя из контекста всего разговора;
- ставит баллы по 4 критериям (скрипт, речь, консультация, вовлечённость);
- формирует список плюсов и минусов.

После ответа модели пункт чек-листа **«Называл заявителя по имени ≥2 раз»** уточняется **по тексту** (подсчёт в репликах оператора, как в локальном `CallQualityEvaluator`); пересчитываются **script_score** и **итоговый балл**, если значение изменилось.

По умолчанию в UI и по нашей рекомендации для ЕКЦ — **YandexGPT Lite** (`yandexgpt-lite`): он даёт лучший баланс цены и качества для русскоязычной оценки звонков. Также доступны **YandexGPT Pro** (`yandexgpt/latest`) для сложных кейсов и **Alice AI LLM** (`aliceai-llm/latest`) для отдельного сравнения через тот же API Yandex AI Studio.

**Настройка:**

1. [Yandex Cloud Console](https://console.yandex.cloud/) → AI Studio → **Создать ключ API** (scope: `yc.ai.languageModels.execute`).
2. Скопируй **Folder ID** (выпадающий список папок в шапке консоли).
3. Задай в `.env` или в **Админ-панели** сайдбара (пароль):

```
YANDEX_API_KEY=AQVNw...
YANDEX_FOLDER_ID=b1g...
YANDEX_CLOUD_MODEL=yandexgpt-lite
# YANDEX_CLOUD_MODEL=yandexgpt/latest
# YANDEX_CLOUD_MODEL=aliceai-llm/latest
```

При ошибке API оценка автоматически переключается на встроенную эвристику (fallback), звонок не теряется.

## Экспорт в Google Таблицу

После анализа (Streamlit или `transcription.py`) при настройке добавляется **строка** в [таблицу ЕКЦ](https://docs.google.com/spreadsheets/d/171ix3FDr7kWzMZMykOwYTd5f6jZ7j3tjJEKzF7W4_1M):

| A | B | C | D | E | F | G |
|---|---|---|---|---|---|---|
| № | Дата | Имя оператора | Номер заявителя | Ожидание `MM:SS` | Общая оценка | Полная оценка |

1. Google Cloud Console → сервисный аккаунт → ключ JSON.
2. В Google Таблице: **Доступ** → email сервисного аккаунта из JSON (`client_email`) с правом **Редактор**.
3. В `.env`: `GOOGLE_SHEETS_CREDENTIALS_JSON=/полный/путь/к/ключ.json`  
   См. [`.env.example`](.env.example) (`GOOGLE_SHEETS_WORKSHEET`, `GOOGLE_SHEETS_HAS_HEADER`, отключение).

Имя файла записи — как при авто-пропуске: `ДД-ММ-ГГГГ_телефон_MM-SS.ext` (дата → B, телефон → D, ожидание → E). Имя оператора — из QA (колонка C).

CLI: флаг `--no-google-sheets` отключает отправку.

**Если строки не появляются:** в интерфейсе после анализа смотрите красное сообщение об ошибке. Частые причины: не включён **Google Sheets API** в проекте GCP ключа; таблица **не расшарена** на `client_email` из JSON; неверный `GOOGLE_SHEETS_WORKSHEET`. Файл `keygoogle.json` в корне проекта подхватывается автоматически, если нет переменных в `.env`.

## Бенчмарк 5-10 звонков

```bash
python benchmark_calls.py /path/to/calls --limit 10 --post-edit --output benchmark_results.csv
```

Скрипт сохраняет latency по этапам, путь диаризации, статус post-edit и применение role-refiner.
