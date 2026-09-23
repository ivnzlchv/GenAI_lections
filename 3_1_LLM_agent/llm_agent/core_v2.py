# llm_agent/core.py

import requests
import json
import re
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_pdfinfo import PDFInfoTool
from .tool_phone_number import PhoneNumberTool

class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(self, model: str = "tngtech/deepseek-r1t2-chimera", local: bool = False, 
                 ollama_base_url: str = "http://localhost:11434", ollama_model: str = "qwen3.5:0.8b"):
        """
        Инициализирует агента.
        
        Args:
            model (str): Название модели для OpenRouter.
            local (bool): Если True, использует локальный Ollama вместо OpenRouter.
            ollama_base_url (str): Базовый URL для Ollama API.
            ollama_model (str): Название модели в Ollama.
        """
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        if not self.local:
            self.api_key = config('OPENROUTER_API_KEY')
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/v1/chat/completions"
            self.model = ollama_model
        
        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "pdf_info": PDFInfoTool(),
            "phone_number": PhoneNumberTool(),
        }
        self.conversation_history = []
    
    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """
        Универсальный метод для отправки запросов к API.
        Поддерживает как OpenRouter, так и Ollama.
        
        Args:
            payload (Dict): Тело запроса.
            headers (Dict, optional): Заголовки запроса.
            
        Returns:
            Dict: Ответ от API.
        """
        if headers is None:
            headers = {}
        
        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")
    
    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        Работает как с OpenRouter, так и с Ollama.
        """
        # Системный промпт, который объясняет агенту его роль и формат ответа
        system_prompt = f"""
        You are a helpful AI planning assistant. Analyze the user's request and decide if you need to use any tools.
        Available tools:
        - **calculator**: For any math-related questions (numbers, calculations). Use it with the full expression.
        - **web_search**: For finding any information about the real world (current events, facts, definitions). Use it with the user's question or a clear search query. USE ONLY RUSSIAN LANGUAGE QUERIES in this tool.
        - **pdf_info**: For extracting information from PDF files (metadata, page count, text content). Use it with a local file path or a URL to a PDF file.
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

        # Формируем запрос к API
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }
        
        try:
            # Для Ollama может потребоваться дополнительная настройка
            if self.local:
                # Отключаем streaming и reasoning, чтобы ответ Qwen попадал в content.
                payload["stream"] = False
                payload["reasoning_effort"] = "none"
            
            response_data = self._make_api_request(payload)
            
            # Извлекаем текстовый ответ от модели
            llm_text = response_data["choices"][0]["message"]["content"]

            # Очищаем ответ от блоков кода Markdown
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_text, re.DOTALL)
            
            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                cleaned_json_text = llm_text

            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text}")
            
            # Пытаемся преобразовать ответ в JSON
            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            if not isinstance(plan, list):
                return []

            valid_steps = [
                step
                for step in plan
                if isinstance(step, dict)
                and isinstance(step.get("action"), str)
                and isinstance(step.get("input"), str)
            ]
            return valid_steps if len(valid_steps) == len(plan) else []
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            # Пробуем альтернативный подход: извлечь JSON из текста
            try:
                # Ищем JSON в тексте без маркеров
                import re
                json_match = re.search(r'\{.*"plan".*\}', llm_text, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except:
                pass
            return []

    def _fallback_plan(self, query: str) -> List[Dict]:
        """Выбирает очевидный инструмент, если малая LLM вернула неверный план."""
        lowered_query = query.lower()

        phone_match = re.search(r"\+?[\d\s()\-]{7,}", query)
        if phone_match and any(
            hint in lowered_query for hint in ("телефон", "номер", "phone")
        ):
            return [
                {"action": "phone_number", "input": phone_match.group().strip()}
            ]

        if re.search(r"\d\s*(?:[+*/]|-(?=\s*\d))", query):
            expression_candidates = [
                part.strip()
                for part in re.findall(r"[\d\s()+\-*/.]+", query)
                if re.search(r"\d", part) and re.search(r"[+*/-]", part)
            ]
            if expression_candidates:
                return [
                    {
                        "action": "calculator",
                        "input": max(expression_candidates, key=len),
                    }
                ]

        if ".pdf" in lowered_query:
            return [{"action": "pdf_info", "input": query}]

        current_fact_hints = (
            "последн",
            "кто выиграл",
            "новост",
            "погода",
            "сейчас",
            "сегодня",
            "current",
            "latest",
        )
        if any(hint in lowered_query for hint in current_fact_hints):
            return [{"action": "web_search", "input": query}]

        return []

    def _generate_final_response(self, user_query: str) -> str:
        """
        Генерирует финальный ответ на основе истории выполнения.
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
        
        if self.local:
            payload["stream"] = False
            payload["reasoning_effort"] = "none"
        
        try:
            response_data = self._make_api_request(payload)
            final_text = response_data["choices"][0]["message"]["content"]
            return final_text
        except Exception as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        """
        Основной метод для обработки запроса пользователя.
        """
        print(f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})")
        
        # --- Шаг 1: Планирование ---
        plan = self._fallback_plan(query)

        if not plan:
            plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            # Генерируем прямой ответ через LLM
            direct_prompt = f"Ответьте на следующий вопрос кратко и информативно: {query}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": direct_prompt}]
            }
            if self.local:
                payload["stream"] = False
                payload["reasoning_effort"] = "none"
            try:
                response_data = self._make_api_request(payload)
                return response_data["choices"][0]["message"]["content"]
            except:
                return "Извините, не удалось сгенерировать ответ."

        # --- Шаг 2: Исполнение плана ---
        print(f"План действий: {plan}")
        tool_results = []
        for step in plan:
            tool_name = step.get('action')
            tool_input = step.get('input')

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result}...")
                tool_results.append((tool_name, result))
                
                # Добавляем результат в историю
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"Ошибка: инструмент с именем '{tool_name}' не найден."
                print(error_msg)
                self.conversation_history.append({'role': 'system', 'content': error_msg})
        
        # Детерминированные инструменты уже сформировали точный ответ. Повторная
        # генерация малой LLM может исказить число или нормализованный телефон.
        if tool_results and all(
            name in {"calculator", "phone_number"} for name, _ in tool_results
        ):
            return "\n".join(result for _, result in tool_results)

        # --- Шаг 3: Генерация финального ответа ---
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        return final_response

    def test_ollama_connection(self) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.
        
        Returns:
            bool: True если соединение успешно, иначе False.
        """
        if not self.local:
            return False
        
        try:
            # Проверяем доступность Ollama API
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False
