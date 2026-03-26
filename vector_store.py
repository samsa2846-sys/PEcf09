"""
Модуль работы с векторным хранилищем FAISS.
Обрабатывает загрузку документов, chunking и поиск по векторам с использованием Yandex Cloud API для эмбеддингов.
"""

import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import re
from yandexgpt_client import YandexGPTClient
from dotenv import load_dotenv

# Загрузка переменных окружения
env_path = Path(__file__).resolve().parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


class VectorStore:
    """Векторное хранилище на основе FAISS с Yandex Cloud эмбеддингами."""
    
    def __init__(self, 
                 collection_name: str = "rag_collection", 
                 persist_directory: str = "./faiss_storage",
                 embedding_model: str = "text-search-doc"):
        """
        Инициализация векторного хранилища.
        
        Args:
            collection_name: имя коллекции (используется в имени файлов)
            persist_directory: директория для хранения данных FAISS
            embedding_model: модель Yandex для эмбеддингов
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        
        # Создаем директорию, если её нет
        self.persist_directory.mkdir(exist_ok=True)
        
        # Инициализация YandexGPT клиента для эмбеддингов
        self.embedding_client = YandexGPTClient()
        
        # Пути к файлам FAISS
        self.index_path = self.persist_directory / f"{collection_name}.index"
        self.metadata_path = self.persist_directory / f"{collection_name}_metadata.pkl"
        
        # Загружаем существующий индекс или создаем новый
        if self.index_path.exists() and self.metadata_path.exists():
            print(f"Загрузка существующего FAISS индекса из {self.index_path}...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                metadata_dict = pickle.load(f)
            self.documents = metadata_dict.get('documents', [])
            self.metadatas = metadata_dict.get('metadatas', [])
            print(f"[OK] FAISS индекс загружен. Документов в коллекции: {len(self.documents)}")
        else:
            print("Создание нового FAISS индекса...")
            self.index = None
            self.documents = []
            self.metadatas = []
            print(f"[OK] Новый FAISS индекс готов к использованию")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение текста на чанки с учётом семантики.
        
        Стратегия:
        1. Приоритет абзацам (разделение по \n\n)
        2. Разбиение длинных абзацев по предложениям
        3. Сохранение контекста через overlap
        4. Учёт минимального и максимального размера чанка
        
        Args:
            text: исходный текст
            chunk_size: целевой размер чанка в символах
            overlap: размер перекрытия между чанками
            
        Returns:
            список чанков
        """
        # Разделяем текст на абзацы
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Если абзац помещается в текущий чанк
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            
            # Если текущий чанк не пустой и добавление абзаца превысит размер
            elif current_chunk:
                chunks.append(current_chunk)
                # Добавляем overlap из конца предыдущего чанка
                overlap_text = self._get_overlap_text(current_chunk, overlap)
                current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
            
            # Если абзац слишком большой, разбиваем его на предложения
            else:
                if len(paragraph) > chunk_size:
                    # Разбиваем длинный абзац на предложения
                    sentence_chunks = self._split_long_paragraph(paragraph, chunk_size, overlap)
                    
                    # Добавляем все чанки кроме последнего
                    if sentence_chunks:
                        chunks.extend(sentence_chunks[:-1])
                        current_chunk = sentence_chunks[-1]
                else:
                    current_chunk = paragraph
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        # Пост-обработка: фильтруем слишком короткие чанки
        chunks = [chunk for chunk in chunks if len(chunk) >= 50]
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        Получение текста для overlap из конца предыдущего чанка.
        Пытается взять целые предложения.
        
        Args:
            text: текст для извлечения overlap
            overlap_size: желаемый размер overlap
            
        Returns:
            текст overlap
        """
        if len(text) <= overlap_size:
            return text
        
        # Берём последние overlap_size символов
        overlap_candidate = text[-overlap_size:]
        
        # Ищем начало предложения в overlap
        sentence_starts = ['. ', '! ', '? ', '\n']
        best_start = 0
        
        for delimiter in sentence_starts:
            pos = overlap_candidate.find(delimiter)
            if pos != -1 and pos > best_start:
                best_start = pos + len(delimiter)
        
        if best_start > 0:
            return overlap_candidate[best_start:].strip()
        
        return overlap_candidate.strip()
    
    def _split_long_paragraph(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Разбиение длинного абзаца на чанки по предложениям.
        
        Args:
            paragraph: абзац для разбиения
            chunk_size: целевой размер чанка
            overlap: размер перекрытия
            
        Returns:
            список чанков
        """
        # Разделяем на предложения
        sentences = re.split(r'([.!?]+\s+)', paragraph)
        
        # Собираем предложения обратно с их разделителями
        full_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                full_sentences.append(sentences[i] + sentences[i + 1])
            else:
                full_sentences.append(sentences[i])
        
        # Если осталось что-то в конце без разделителя
        if len(sentences) % 2 == 1:
            full_sentences.append(sentences[-1])
        
        chunks = []
        current_chunk = ""
        
        for sentence in full_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Если предложение помещается в текущий чанк
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                # Сохраняем текущий чанк
                if current_chunk:
                    chunks.append(current_chunk)
                    # Добавляем overlap
                    overlap_text = self._get_overlap_text(current_chunk, overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    # Если одно предложение больше chunk_size, всё равно добавляем его
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def load_documents(self, file_path: str):
        """
        Загрузка документов из файла в векторное хранилище.
        
        Args:
            file_path: путь к файлу с документами
        """
        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        # Чтение файла
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Разбиение на чанки
        chunks = self._chunk_text(text)
        print(f"Текст разбит на {len(chunks)} чанков")
        
        # Проверка, не загружены ли уже документы
        if len(self.documents) > 0:
            print("Документы уже загружены в коллекцию")
            return
        
        # Создание embeddings и добавление в FAISS
        print(f"\nСоздание эмбеддингов для {len(chunks)} чанков через Yandex Cloud API...")
        print(f"(Модель: {self.embedding_model})")
        
        # Обрабатываем батчами для безопасности
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"  Обработка чанков {i+1}-{min(i+batch_size, len(chunks))} из {len(chunks)}...")
            
            batch_embeddings = self.embedding_client.get_embeddings(batch, model=self.embedding_model)
            all_embeddings.extend(batch_embeddings)
        
        # Преобразуем эмбеддинги в numpy массив
        if not all_embeddings:
            print("[WARNING] Нет эмбеддингов для сохранения")
            return
        
        embeddings_array = np.array(all_embeddings).astype('float32')
        dimension = embeddings_array.shape[1]
        
        # Создаем или обновляем FAISS индекс
        if self.index is None:
            print(f"\nСоздание нового FAISS индекса (размерность: {dimension})...")
            self.index = faiss.IndexFlatL2(dimension)
        
        # Добавляем новые эмбеддинги в индекс
        print("Добавление эмбеддингов в FAISS индекс...")
        self.index.add(embeddings_array)
        
        # Обновляем документы и метаданные
        chunk_id = len(self.documents)
        for i, chunk in enumerate(chunks):
            self.documents.append(chunk)
            self.metadatas.append({
                "source": file_path,
                "chunk_id": chunk_id + i,
                "chunk_length": len(chunk)
            })
        
        # Сохраняем индекс и метаданные
        print("Сохранение FAISS индекса и метаданных...")
        faiss.write_index(self.index, str(self.index_path))
        
        metadata_dict = {
            'documents': self.documents,
            'metadatas': self.metadatas
        }
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata_dict, f)
        
        print(f"[OK] Загружено {len(chunks)} чанков. Всего в базе: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Поиск релевантных документов по запросу.
        
        Args:
            query: текст запроса
            top_k: количество документов для возврата
            
        Returns:
            список документов с метаданными
        """
        # Проверяем, есть ли документы
        if not self.index or len(self.documents) == 0:
            print("[WARNING] Предупреждение: индекс пуст, нет документов для поиска")
            return []
        
        # Создание embedding для запроса через Yandex Cloud API
        query_embeddings = self.embedding_client.get_embeddings([query], model="text-search-query")
        query_embedding = query_embeddings[0]
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Выполняем поиск в FAISS
        k = min(top_k, len(self.documents))
        distances, indices = self.index.search(query_vector, k)
        
        # Форматирование результатов
        documents = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Проверяем валидность индекса
                documents.append({
                    'id': f"doc_{idx}",
                    'text': self.documents[idx],
                    'distance': float(distances[0][i]),
                    'source': self.metadatas[idx].get('source', 'unknown')
                })
        
        return documents
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Получение статистики коллекции.
        
        Returns:
            словарь со статистикой
        """
        return {
            'name': self.collection_name,
            'count': len(self.documents),
            'persist_directory': str(self.persist_directory)
        }


if __name__ == "__main__":
    # Тестирование векторного хранилища
    import sys
    
    if not os.getenv("YANDEX_API_KEY") or not os.getenv("YANDEX_FOLDER_ID"):
        print("Ошибка: установите переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID")
        sys.exit(1)
    
    vector_store = VectorStore(collection_name="test_collection")
    
    # Загрузка документов
    if os.path.exists("data/docs.txt"):
        vector_store.load_documents("data/docs.txt")
    
    # Поиск
    results = vector_store.search("Что такое машинное обучение?", top_k=3)
    print("\nРезультаты поиска:")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc['text'][:200]}...")
        print(f"   Distance: {doc['distance']}")
    
    # Статистика
    stats = vector_store.get_collection_stats()
    print(f"\nСтатистика: {stats}")

