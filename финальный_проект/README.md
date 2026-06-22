# Support Ticket Routing Pipeline

Конвейер для классификации и маршрутизации тикетов поддержки с использованием RAG, ReAct-агента и LLM-as-judge.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Настройка
Скопируйте .env.example в .env и укажите настройки:

```bash
cp .env.example .env
Пример .env:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_AUTH_TOKEN=ollama
LLM_MODEL=llama3.2:3b

## Запуск
Полный запуск (подготовка данных → обработка → оценка):

```bash
python run_all.py
```
Поэтапный запуск:

```bash
# Подготовка данных
python prepare_data.py

# Обработка тикетов (по умолчанию 5)
python pipeline.py --limit 5

# Оценка на gold-наборе
python eval.py
```
## Результаты
Все артефакты сохраняются в папку output/:

predictions.json — предсказания по всем тикетам

eval_results.json — метрики оценки

## Структура
support_ticket_routing/
├── run_all.py          # Полный запуск
├── prepare_data.py     # Подготовка данных
├── pipeline.py         # Основной пайплайн
├── eval.py             # Оценка
├── agent.py            # ReAct-агент
├── tools.py            # Инструменты агента
├── rag.py              # RAG (BM25 + ChromaDB)
├── schema.py           # Pydantic-схемы
├── judge.py            # LLM-as-judge
├── hallucination.py    # Проверка галлюцинаций
├── llm_client.py       # Клиент LLM
├── requirements.txt
├── .env.example
├── input/
│   ├── tickets.csv
│   ├── eval_gold.json
│   └── kb/
│       ├── doc_000.md
│       └── ...
└── output/
    ├── predictions.json
    └── eval_results.json