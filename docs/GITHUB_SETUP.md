# Публикация на GitHub

## Что не попадает в репозиторий

См. `.gitignore`: `.env`, `.venv`, `keygoogle.json`, `результаты/`, `pretrained_models/` (локальные симлинки на кэш HF), `__pycache__`.

На новом ПК: скопируйте `.env.example` → `.env`, положите `keygoogle.json` при необходимости, выполните `pip install -r requirements.txt`.

## Первый push (у вас уже есть локальный коммит)

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
