# Лабораторная работа 1 — вариант 31

[![Покрытие тестами](https://codecov.io/gh/ivnzlchv/GenAI_lections/branch/lab1-phone-number/graph/badge.svg)](https://codecov.io/gh/ivnzlchv/GenAI_lections)

`PhoneNumberTool` нормализует и валидирует телефонные номера с помощью библиотеки
[`phonenumbers`](https://pypi.org/project/phonenumbers/). Локальные номера по
умолчанию разбираются для региона `RU`; международные номера следует передавать с
кодом страны.

## Подготовка

```bash
brew install ollama
brew services start ollama
ollama pull qwen3.5:0.8b

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Запуск

```bash
python main.py
```

Пример прямого использования:

```python
from llm_agent.tool_phone_number import PhoneNumberTool

tool = PhoneNumberTool(default_region="RU")
print(tool.normalize("8 (912) 345-67-89"))
print(tool.validate("+1 415 555 2671"))
print(tool.analyze("+44 20 7946 0958"))
```

## Тесты

```bash
python -m pytest -v
```

Тесты автоматически запускаются в GitHub Actions при каждом `push` и при создании
или обновлении pull request. После успешного запуска Codecov пересчитывает процент
покрытия и динамически обновляет бейдж.
