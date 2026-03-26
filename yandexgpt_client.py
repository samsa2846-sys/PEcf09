"""
Клиент для работы с YandexGPT API.
Управляет запросами к Yandex Cloud для генерации текста и создания эмбеддингов.
"""

import requests
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from pathlib import Path

# Загрузка переменных окружения
env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class YandexGPTClient:
    """Клиент для работы с YandexGPT API."""
    
    def __init__(self, api_key: str = None, folder_id: str = None):
        """
        Инициализация YandexGPT клиента.
        
        Args:
            api_key: API ключ Yandex Cloud
            folder_id: ID каталога Yandex Cloud
        """
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        
        if not self.api_key:
            raise ValueError("YANDEX_API_KEY не установлен")
        if not self.folder_id:
            raise ValueError("YANDEX_FOLDER_ID не установлен")
        
        # URL для API Yandex Cloud
        self.completion_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.embedding_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
        
        print("[OK] YandexGPT клиент инициализирован")
    
    def _get_headers(self) -> Dict[str, str]:
        """Получение заголовков для запросов."""
        return {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       model: str = "yandexgpt-lite/latest",
                       temperature: float = 0.3,
                       max_tokens: int = 500) -> str:
        """
        Отправка запроса к чат-модели YandexGPT.
        
        Args:
            messages: список сообщений в формате [{"role": "user", "content": "..."}]
            model: название модели (yandexgpt-lite/latest, yandexgpt/latest и т.д.)
            temperature: температура генерации
            max_tokens: максимальное количество токенов в ответе
            
        Returns:
            текст ответа от модели
        """
        # Формируем modelUri для Yandex Cloud
        model_uri = f"gpt://{self.folder_id}/{model}"
        
        # Фильтруем сообщения - убираем пустые и проверяем формат
        filtered_messages = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                content = str(msg["content"]).strip()
                if content:  # Только непустые сообщения
                    filtered_messages.append({
                        "role": str(msg["role"]),
                        "text": content
                    })
        
        if not filtered_messages:
            raise ValueError("Нет валидных сообщений для отправки")
        
        payload = {
            "modelUri": model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": temperature,
                "maxTokens": str(max_tokens)
            },
            "messages": filtered_messages
        }
        
        try:
            response = requests.post(
                self.completion_url,
                headers=self._get_headers(),
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                error_text = response.text
                raise Exception(f"Ошибка запроса к YandexGPT: {response.status_code} - {error_text}")
            
            data = response.json()
            return data['result']['alternatives'][0]['message']['text']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка запроса к YandexGPT: {e}")
        except KeyError as e:
            raise Exception(f"Неожиданный формат ответа от YandexGPT: {e}, ответ: {response.text if 'response' in locals() else 'N/A'}")
    
    def get_embeddings(self, texts: List[str], model: str = "text-search-doc") -> List[List[float]]:
        """
        Получение векторных представлений текстов через Yandex Cloud API.
        
        Args:
            texts: список текстов для векторизации
            model: модель для embeddings (text-search-doc, text-search-query)
            
        Returns:
            список векторов
        """
        # Формируем modelUri для модели эмбеддингов
        model_uri = f"emb://{self.folder_id}/{model}"
        
        embeddings = []
        
        # Yandex Cloud API обрабатывает тексты по одному
        for text in texts:
            payload = {
                "modelUri": model_uri,
                "text": text
            }
            
            try:
                response = requests.post(
                    self.embedding_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                embedding = result.get("embedding", [])
                if not embedding:
                    raise Exception(f"Пустой эмбеддинг в ответе API: {result}")
                
                embeddings.append(embedding)
                
            except Exception as e:
                raise Exception(f"Ошибка создания эмбеддинга: {e}")
        
        return embeddings


if __name__ == "__main__":
    # Тестирование клиента
    import sys
    
    try:
        client = YandexGPTClient()
        
        # Тест чата
        print("\n=== Тест чата ===")
        test_message = "Что такое машинное обучение? Ответь кратко."
        print(f"Отправка сообщения: {test_message}")
        response = client.chat_completion(
            messages=[
                {"role": "user", "content": test_message}
            ]
        )
        print(f"Ответ: {response}")
        
        # Тест embeddings
        print("\n=== Тест embeddings ===")
        embeddings = client.get_embeddings(["Тестовый текст"])
        print(f"Размерность вектора: {len(embeddings[0])}")
        
        print("\n[OK] Все тесты пройдены")
        
    except Exception as e:
        print(f"[ERROR] Ошибка: {e}")
        sys.exit(1)

