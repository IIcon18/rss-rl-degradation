# rss-rl-degradation

Деградация безопасности RL-агента с RSS-фильтром при нарушении допущений о поведении участников дорожного движения.

## Установка

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
```

Точные версии всех пакетов: `requirements-lock.txt`.

## Структура

- `rssrl/` — код (формулы RSS, нарушители, фильтр, сцены, логирование)
- `tests/` — проверки (pytest)
- `notebooks/` — пилоты по этапам; `make_XX_*.py` собирает и выполняет соответствующий ноутбук
- `results/figures/` — графики
