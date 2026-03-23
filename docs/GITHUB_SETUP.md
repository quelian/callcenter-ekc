# Публикация на GitHub

## Что не попадает в репозиторий

См. `.gitignore`: `.env`, `.venv`, `keygoogle.json`, `результаты/`, `pretrained_models/` (локальные симлинки на кэш HF), `__pycache__`.

На новом ПК: см. **[WINDOWS.md](WINDOWS.md)** (Windows) или раздел «Быстрый старт» в README (macOS/Linux): `.env.example` → `.env`, при необходимости `keygoogle.json`, `pip install -r requirements.txt`.

## Синхронизация с GitHub (после правок в проекте)

Из корня репозитория:

```bash
git status
git add -A
git commit -m "Кратко: что изменили"
git push
```

Подтянуть чужие изменения с GitHub:

```bash
git pull
```

---

## Первый push (репозиторий только локальный)

1. На [github.com](https://github.com) создайте **новый репозиторий** (без README, без .gitignore — проект уже локальный).
2. В терминале в корне проекта:

```bash
git remote add origin https://github.com/ВАШ_ЛОГИН/ИМЯ_РЕПО.git
git branch -M main
git push -u origin main
```

При запросе логина GitHub используйте **Personal Access Token** (Settings → Developer settings → Tokens), а не пароль.

## Альтернатива: GitHub CLI

```bash
brew install gh
gh auth login
gh repo create callcenter-qa-app --private --source=. --remote=origin --push
```

(`--public` вместо `--private`, если нужен открытый репозиторий.)
