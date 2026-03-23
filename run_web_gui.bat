@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

if not exist "web_gui.py" (
    echo Ошибка: web_gui.py не найден. Запускайте скрипт из папки проекта.
    pause
    exit /b 1
)

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo Виртуальное окружение .venv не найдено.
    echo Создайте его в этой папке:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate.bat
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

echo Запуск Streamlit...
echo Откройте в браузере адрес, который покажет консоль ^(обычно http://localhost:8501^).
echo.

python -m streamlit run web_gui.py

echo.
pause
