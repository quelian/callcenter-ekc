# Windows: коротко, с нуля

Нужно: **Windows 10/11** и **Python 3.11+**.  
Скачать Python: [python.org/downloads](https://www.python.org/downloads/) — при установке включите **Add python.exe to PATH**.

---

## 1. Получить папку проекта

**Без Git:** на GitHub нажмите **Code** → **Download ZIP** → распакуйте (например в `C:\Users\Вас\callcenter-ekc`).

**С Git:** установите [Git for Windows](https://git-scm.com/download/win), откройте **cmd**:

```bat
cd %USERPROFILE%\Desktop
git clone https://github.com/quelian/callcenter-ekc.git
cd callcenter-ekc
```

*(Подставьте свой URL репозитория, если другой.)*

---

## 2. Один раз: окружение и зависимости

Откройте **cmd**, перейдите в **корень** папки проекта (где лежат `web_gui.py`, `requirements.txt`):

```bat
cd C:\путь\к\папке\проекта
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- Если **`py` не найден** — попробуйте `python` вместо `py`.  
- Команда **`source`** в cmd **не используется** (это для Mac/Linux).

---

## 3. Настройка (по необходимости)

- Скопируйте **`.env.example`** → файл **`.env`**, заполните переменные (см. комментарии в `.env.example` и [README](../README.md)).
- Для Google Таблицы положите **`keygoogle.json`** в корень проекта (или укажите путь в `.env`).

---

## 4. Запуск

Дважды щёлкните **`run_web_gui.bat`** в папке проекта  

**или** в cmd после `activate.bat`:

```bat
python -m streamlit run web_gui.py
```

Откройте в браузере: **http://localhost:8501**

---

## PowerShell

```powershell
cd C:\путь\к\проекту
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run web_gui.py
```

Если скрипт активации блокируется:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

Полное описание возможностей — в [README.md](../README.md).
