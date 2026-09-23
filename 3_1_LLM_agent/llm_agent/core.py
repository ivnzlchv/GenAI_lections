# llm_agent/core.py

import requests
import json
from typing import List, Dict
from decouple import config # Импортируем для чтения .env

# Импортируем наши инструменты
from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_phone_number import PhoneNumberTool

class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Основан на понятной структуре: Планирование -> Исполнение -> Ответ.
    """

    def __init__(self, model: str = "tngtech/deepseek-r1t2-chimera"):
        """
        Инициализирует агента.
        
        Args:
            model (str): Название модели для использования в Openrouter.
        """
        self.api_key = config('OPENROUTER_API_KEY') # Безопасно читаем ключ из .env
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        
        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "phone_number": PhoneNumberTool(),
        }
        self.conversation_history = []
    
    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        Это реализация вашего метода `create_plan`.
        """
        # Системный промпт, который объясняет агенту его роль и формат ответа
        system_prompt = f"""
        You are a helpful AI planning assistant. Analyze the user's request and decide if you need to use any tools.

        Available tools:
        - **calculator**: For any math-related questions (numbers, calculations). Use it with the full expression.
        - **web_search**: For finding any information about the real world (current events, facts, definitions). Use it with the user's question or a clear search query.
        - **phone_number**: For validating and normalizing phone numbers. Pass only the phone number, including its country code when known.

        Your response MUST be ONLY a JSON object of the following format.
        If one or more tools are needed to answer, return JSON of this structure:
        {{
        "plan": [
            {{"action": "tool_name", "input": "some text to pass into tool"}},
            ... //MORE ACTIONS IF NEEDED SEVERAL TOOLS. ONE ACTION FOR ONE TOOL CALL
        ]
        }}
        If no tool is needed, return an empty plan: {{"plan": []}}.
        """

        # Формируем и отправляем запрос к API Openrouter
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()  # Проверка на ошибки HTTP
            llm_response = response.json()
            
            # Извлекаем текстовый ответ от модели
            llm_text = llm_response["choices"][0]["message"]["content"]

            # Очищаем ответ от блоков кода Markdown, если они есть.
            # Используем регулярное выражение, чтобы найти и извлечь JSON из ```json...```
            import re
            # Ищем паттерн ```json...`` ` и захватываем только внутреннее содержимое
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_text, re.DOTALL)
            
            if json_match:
                # Если нашли совпадение, берем только содержимое первой группы (то, что в скобках)
                cleaned_json_text = json_match.group(1)
            else:
                # Иначе, предполагаем, что ответ уже является "чистым" JSON
                cleaned_json_text = llm_text

            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text}")
            
            # Пытаемся преобразовать ответ в JSON
            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            return plan
            
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            return [] # Возвращаем пустой план, если что-то пошло не так

    def _generate_final_response(self, user_query: str) -> str:
        """
        Генерирует финальный ответ на основе истории выполнения.
        Это реализация вашего метода `generate_final_response`.
        """
        prompt = f"""
Based on the following conversation log, provide a direct and helpful answer to the user's original question.
Be concise and use the information from the tool results to support your answer.

Original User Question: {user_query}

Conversation Log:
{chr(10).join([msg['content'] for msg in self.conversation_history])}
"""
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            final_text = response.json()["choices"][0]["message"]["content"]
            return final_text
        except requests.exceptions.RequestException as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        """
        Основной метод для обработки запроса пользователя.
        Следует вашей четкой структуре: План -> Исполнение -> Ответ.
        """
        print("Агент анализирует ваш запрос...")
        
        # --- Шаг 1: Планирование ---
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            # Можно вызвать LLM для прямого ответа или просто вернуть заглушку
            return "Для выполнения этого запроса мне не потребовались специальные инструменты."

        # --- Шаг 2: Исполнение плана ---
        print(f"План действий: {plan}")
        for step in plan:
            tool_name = step.get('action')
            tool_input = step.get('input')

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result[:100]}...")
                
                # Добавляем результат в историю, как в вашем примере
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"Ошибка: инструмент с именем '{tool_name}' не найден."
                print(error_msg)
                self.conversation_history.append({'role': 'system', 'content': error_msg})
        
        # --- Шаг 3: Генерация финального ответа ---
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        return final_response
