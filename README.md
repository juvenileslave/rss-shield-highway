# rss-shield-highway

**Назначение.** RSS-фильтр безопасности для DQN-агента в highway-env: маска действий, задержка
исполнительного тракта τ = Δt, супервизорные команды TAKEOVER / ABORT / FALLBACK. Экспериментальный
протокол исследования — `docs/protocol/`.

**Текущее состояние.** Протокол исследования — `docs/protocol/Протокол_исследования_RSS-фильтр.docx`.
Пакет верификации: 67 тестов, все строки таблицы 3 — pass. Основное обучение DQN ещё не запускалось.

## Быстрый старт

Требуется Python ≥ 3.10.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
PYTHON=python ./run_all.sh     # код выхода 0 = все строки таблицы 3 pass
```

`run_all.sh` прогоняет pytest, контрольные сценарии, профиль манёвра и собирает `results/` и `logs/`.
Сравнение с эталоном: `diff -r expected/ results/` (могут отличаться только время прогона,
строки ОС/Python и хэши, если менялся код).

## Карта репозитория

| Путь | Что это |
|---|---|
| `rss_shield/` | подкласс `MDPVehicle` (FIFO на 2 кадра, ограничители), фильтр с маской и иерархией, сценарии, интеграция с `HighwayEnv` |
| `tests/` | 67 pytest-тестов; `tests/reference.py` — независимый эталон таблицы 2 |
| `tools/make_table3.py` | сборка `results/table3.md` и `table3.csv` из журналов и pytest |
| `expected/` | эталонный прогон проверяющего |
| `runs/v9_author_run/` | зафиксированный прогон автора к v9 (`results/`, `logs/`) |
| `docs/protocol/` | протокол исследования (`Протокол_исследования_RSS-фильтр.docx`) |
| `experiments/`, `configs/` | заготовки под обучение, сиды и сетку D × ρ |
| `.github/workflows/verify.yml` | CI: Python 3.13 и 3.14, сверка `table3.csv` с эталоном |

## Правила

- `tests/reference.py` и `expected/` **не меняются**.
- Любое изменение `rss_shield/` — через PR с зелёным CI и обновлённым `expected/`; в описании PR
  указывается причина.
- `sim/` — точное ускорение highway-env для экспериментов; любое его изменение — только
  с зелёным `sim/test_fast_road.py`.
- Прямых коммитов в `main` нет: только через PR.
