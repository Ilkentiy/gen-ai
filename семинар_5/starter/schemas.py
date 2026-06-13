"""
JSON-схемы инструментов для API вызова инструментов (OpenAI-совместимого).

Эту запись модель читает, чтобы решить, какой инструмент звать и с какими
аргументами. Чем точнее описание — тем реже агент ошибается.

На семинаре дописываем эти схемы руками (в бою их генерируют из Pydantic
и аннотаций типов, но сначала полезно понять, что туда попадает).
"""

TOOL_SCHEMAS = [
    # ----- пример схемы (готовый, для ориентира) -----
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Безопасный математический калькулятор. Понимает +, -, *, /, ^, "
                "sqrt, ln, log, exp, скобки. Использовать для любых вычислений "
                "над числами, полученными от других инструментов — руками не считать."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Математическое выражение, например '(21 - 9.5)' или "
                            "'log(2) / log(1 + 0.17)'. Не используй переменные, "
                            "только числа и операторы."
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_fx_rate",
            "description": (
                "Официальный курс валюты к рублю на дату по данным ЦБ РФ. "
                "Зови, если вопрос про курс USD/EUR/CNY/прочих — не придумывай курс. "
                "Возвращает сколько рублей стоит 1 единица валюты."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "currency": {
                        "type": "string",
                        "description": "ISO-код валюты: USD, EUR, CNY, GBP, JPY, TRY и т.д.",
                        "default": "USD"
                    },
                    "on_date": {
                        "type": ["string", "null"],
                        "description": "Дата в формате YYYY-MM-DD. Если не задана — сегодня.",
                        "default": None
                    },
                },
                "required": ["currency"],
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "get_key_rate",
            "description": (
                "Ключевая ставка Банка России на дату, % годовых. Для текущей — "
                "с cbr.ru, для исторической — из локального архива изменений ставки. "
                "Используй для расчётов реальной ставки, доходности вкладов, "
                "стоимости денег во времени."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "on_date": {
                        "type": ["string", "null"],
                        "description": "Дата в формате YYYY-MM-DD. Если не задана — сегодня.",
                        "default": None
                    },
                },
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "get_inflation",
            "description": (
                "Индекс потребительских цен Росстата, % г/г, на конец месяца. "
                "Для инфляции и реальной доходности. Инфляция считается "
                "год к году (по сравнению с тем же месяцем предыдущего года)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer", 
                        "description": "Год, например 2024",
                        "minimum": 2000,
                        "maximum": 2030
                    },
                    "month": {
                        "type": "integer",
                        "description": "Месяц 1..12 (1 = январь, 12 = декабрь)",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": ["year", "month"],
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "get_unemployment",
            "description": (
                "Уровень безработицы (методология МОТ) Росстата, % от рабочей силы, "
                "на конец месяца. Для «индекса нищеты» (инфляция + безработица) "
                "и анализа рынка труда."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "Год, например 2024",
                        "minimum": 2000,
                        "maximum": 2030
                    },
                    "month": {
                        "type": "integer",
                        "description": "Месяц 1..12 (1 = январь, 12 = декабрь)",
                        "minimum": 1,
                        "maximum": 12,
                    },
                },
                "required": ["year", "month"],
            },
        },
    },
    
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": (
                "Сравнить значение метрики в двух периодах. Используй для вопросов: "
                "'во сколько раз вырос', 'как изменилась', 'сравни курс в двух датах', "
                "'на сколько процентов увеличилось'. Этот инструмент сам вызывает "
                "get_fx_rate, get_key_rate, get_inflation или get_unemployment "
                "и возвращает разницу (delta) и отношение (ratio). "
                "НЕ вызывай get_* инструменты отдельно, если можно использовать compare_periods."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": [
                            "key_rate", 
                            "fx_USD", 
                            "fx_EUR", 
                            "fx_CNY", 
                            "fx_GBP", 
                            "fx_JPY", 
                            "cpi", 
                            "unemployment"
                        ],
                        "description": (
                            "Метрика для сравнения. "
                            "key_rate - ключевая ставка ЦБ, "
                            "fx_XXX - курс валюты XXX к рублю, "
                            "cpi - инфляция (ИПЦ), "
                            "unemployment - безработица"
                        )
                    },
                    "period_a": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}(-\\d{2})?$",
                        "description": (
                            "Первый период в формате YYYY-MM (для месячных данных) "
                            "или YYYY-MM-DD (для дневных данных). "
                            "Примеры: '2022-01', '2024-03-15'"
                        )
                    },
                    "period_b": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}(-\\d{2})?$",
                        "description": (
                            "Второй период в формате YYYY-MM или YYYY-MM-DD. "
                            "Будет сравнен с period_a."
                        )
                    }
                },
                "required": ["metric", "period_a", "period_b"],
                "additionalProperties": False
            },
        },
    },
]