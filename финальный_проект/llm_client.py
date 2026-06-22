import os
import json
import asyncio
from typing import Type, TypeVar, Optional, Dict, Any
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    """Клиент для работы с LLM"""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 2
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.api_key = api_key or os.getenv("LLM_AUTH_TOKEN") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-v4-flash")
        self.max_retries = max_retries
        
        if not self.api_key:
            raise ValueError("LLM_AUTH_TOKEN или OPENAI_API_KEY не задан")
        
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=60.0
        )
    
    async def chat(
        self,
        messages: list,
        response_model: Type[T],
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> T:
        """Chat completion с структурированным выводом"""
        
        for attempt in range(self.max_retries):
            try:
                # Добавляем указание вернуть JSON в системный промпт
                system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
                if system_msg:
                    system_msg["content"] = system_msg["content"] + " Верни результат в формате JSON."
                else:
                    messages.insert(0, {"role": "system", "content": "Верни результат в формате JSON."})
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                return response_model(**data)
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Ошибка после {self.max_retries} попыток: {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError("Не удалось получить ответ от LLM")
    
    async def chat_simple(
        self,
        messages: list,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """Простой chat completion без структурированного вывода"""
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Ошибка после {self.max_retries} попыток: {e}")
                await asyncio.sleep(1)
        
        raise RuntimeError("Не удалось получить ответ от LLM")

llm_client = LLMClient()