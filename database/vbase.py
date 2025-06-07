import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import TypedDict
import pickle
from tqdm import tqdm

from database.psql_base import PostgresBase


class ConstitutionArticle(TypedDict):
    article_id: int
    chapter_id: int
    num: str
    text: str
    score: float

class CodeArticle(TypedDict):
    article_id: int
    codes_id: int
    chapter_num: str
    chapter_title: str
    article_num: str
    article_title: str
    text: str
    comments: str
    additional: str
    court_links: list
    score: float

class LawArticle(TypedDict):
    article_id: int
    law_id: int
    chapter_num: str
    chapter_title: str
    article_num: str
    article_title: str
    text: str
    comments: str
    additional: str
    court_links: list
    score: float


class SearchResult(TypedDict):
    """Класс для результата поиска"""
    chunk_text: str
    full_article: Dict[str, Any]
    score: float


class QdrantLegalRAG:
    def __init__(self, host: str = "localhost", port: int = 6333, collection_prefix: str = "legal", timeout=1000.0):
        """
        Инициализация клиента Qdrant и настройка коллекций
        
        Args:
            host: хост Qdrant сервера
            port: порт Qdrant сервера  
            collection_prefix: префикс для названий коллекций
        """
        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.encoder = SentenceTransformer('BAAI/bge-m3', device='cpu')
        
        # Коллекции
        self.constitution_collection = f"{collection_prefix}_constitution"
        self.codes_collection = f"{collection_prefix}_codes"
        self.laws_collection = f"{collection_prefix}_laws"

        self.postgres_base = PostgresBase()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # Размерность векторов модели
        self.vector_dimension = 1024
        
        # Инициализируем коллекции
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Создание коллекций в Qdrant"""
        collections = [
            self.constitution_collection,
            self.codes_collection, 
            self.laws_collection
        ]
        
        for collection_name in collections:
            try:
                # Проверяем, существует ли коллекция
                self.client.get_collection(collection_name)
                print(f"Коллекция {collection_name} уже существует")
            except:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                print(f"Создана коллекция {collection_name}")
    
    def _create_embeddings(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Создание эмбеддингов для текстов
        
        Args:
            texts: список текстов для векторизации
            
        Returns:
            массив векторов
        """
        if is_query:
            return self.encoder.encode(f'query: {text}')
        return self.encoder.encode(f'passage: {text}')
    
    def _split_article_to_chunks(self, article_text: str) -> List[str]:
        """
        Разбиение статьи на чанки
        
        Args:
            article_text: текст статьи
            
        Returns:
            список чанков
        """
        return self.text_splitter.split_text(article_text)
    
    def add_constitution_articles(self, articles: List[Dict[str, Any]]):
        """
        Добавление статей конституции в коллекцию
        
        Args:
            articles: список статей конституции
        """
        
        for i, article in tqdm(enumerate(articles), total=len(articles)):
            points = []
            # Разбиваем на чанки
            chunks = self._split_article_to_chunks(article['text'])
            
            # Создаем эмбеддинги
            embeddings = [self._create_embeddings(chunk) for chunk in chunks]
            
            # Создаем точки
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                
                payload = {
                    'chunk_text': chunk,
                    'article_id': i,
                    'chapter_id': article['chapter_id'],
                    'article_num': article['num'],
                    'full_text': article['text'],
                    'article_type': 'constitution'
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
        
            # Загружаем точки
            self.client.upsert(
                collection_name=self.constitution_collection,
                points=points
            )

            # Загружаем статью в psql
            # self.postgres_base.add_constitution_atricle([article])


        # print(f"Добавлено {len(points)} чанков статей конституции")
    
    def add_codes_articles(self, articles: List[Dict[str, Any]]):
        """
        Добавление статей кодексов в коллекцию
        
        Args:
            articles: список статей кодексов
        """
        
        for article_id, article in tqdm(enumerate(articles), total=len(articles)):
            points = []
            # Разбиваем статью на чанки
            if article['text'] == '':
                continue
            chunks = self._split_article_to_chunks(article['text'])
            
            # Создаем эмбеддинги для чанков
            embeddings = [self._create_embeddings(chunk) for chunk in chunks]
            
            # Создаем точки для каждого чанка
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                
                payload = {
                    'chunk_text': chunk,
                    'article_id': article_id,
                    'codes_id': article['codes_id'],
                    'chapter_num': article['chapter_num'],
                    'chapter_title': article['chapter_title'],
                    'article_num': article['article_num'],
                    'title': article['title'],
                    'article_type': 'codes'
                }
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
        
            # Загружаем точки в коллекцию
            self.client.upsert(
                collection_name=self.codes_collection,
                points=points
            )

            # Загружаем статью в psql
            self.postgres_base.add_code_atricle(article, article_id)
        # print(f"Добавлено {len(points)} чанков статей кодексов")
    
    def add_laws_articles(self, articles: List[Dict[str, Any]]):
        """
        Добавление статей ФЗ в коллекцию
        
        Args:
            articles: список статей ФЗ
        """
        
        for article_id, article in tqdm(enumerate(articles), total=len(articles)):
            points = []
            if article['text'] == '':
                continue
            # Разбиваем статью на чанки
            chunks = self._split_article_to_chunks(article['text'])
            
            # Создаем эмбеддинги для чанков
            embeddings = [self._create_embeddings(chunk) for chunk in chunks]
            
            # Создаем точки для каждого чанка
            for chunk, embedding in zip(chunks, embeddings):
                point_id = str(uuid.uuid4())
                
                payload = {
                    'chunk_text': chunk,
                    'article_id': article_id,
                    'law_id': article['law_id'],
                    'chapter_num': article['chapter_num'] if article.get('chapter_num') is not None else '',
                    'chapter_title': article['chapter_title'] if article.get('chapter_title') is not None else '',
                    'article_num': article['article_num'],
                    'title': article['title'],
                    'article_type': 'laws'
                }
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
                points.append(point)
        
            # Загружаем точки в коллекцию
            self.client.upsert(
                collection_name=self.laws_collection,
                points=points
            )

            # Загружаем статью в psql
            self.postgres_base.add_law_atricle(article, article_id)
        # print(f"Добавлено {len(points)} чанков статей ФЗ")
    
    def search_constitution_by_chapter(self, query: str, chapter_id: int, limit: int = 3) -> List[SearchResult]:
        """
        Поиск в статьях конституции по главе
        
        Args:
            query: поисковый запрос
            chapter_id: ID главы конституции
            limit: количество результатов
            
        Returns:
            список результатов поиска
        """
        # Создаем эмбеддинг для запроса
        query_embedding = self._create_embeddings(query, is_query=True)
        
        search_params = {
            "collection_name": self.constitution_collection,
            "query_vector": query_embedding.tolist(),
            "limit": limit
        }

        if chapter_id is not None:
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chapter_id",
                        match=models.MatchValue(value=chapter_id)
                    )
                ]
            )
        
        # Выполняем поиск с фильтром 
        search_result = self.client.search(**search_params)
        
        # Преобразуем результаты
        results = []
        article_ids = set()
        for hit in search_result:
            payload = hit.payload
            if payload['article_id'] in article_ids:
                continue

            result = ConstitutionArticle(
                article_id = payload['article_id'],
                chapter_id = payload['chapter_id'],
                num = payload['article_num'],
                text = payload['full_text'],
                score = hit.score,
            )
            results.append(result)
            article_ids.add(payload['article_id'])
        
        return results
    
    def search_codes_by_id(self, query: str, codes_id: int, limit: int = 3) -> List[SearchResult]:
        """
        Поиск в статьях кодексов по ID кодекса
        
        Args:
            query: поисковый запрос
            codes_id: ID кодекса
            limit: количество результатов
            
        Returns:
            список результатов поиска
        """
        # Создаем эмбеддинг для запроса
        query_embedding = self._create_embeddings(query, is_query=True)

        search_params = {
            "collection_name": self.codes_collection,
            "query_vector": query_embedding.tolist(),
            "limit": limit
        }

        if codes_id is not None:
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="codes_id",
                        match=models.MatchValue(value=codes_id)
                    )
                ]
            )
        
        # Выполняем поиск с фильтром по кодексу
        search_result = self.client.search(**search_params)
        
        # Преобразуем результаты
        results = []
        article_ids = set()
        for hit in search_result:
            payload = hit.payload
            article_id = payload['article_id']

            if article_id in article_ids:
                continue

            article = self.postgres_base.get_code_article(article_id)

            result = CodeArticle(
                article_id = article_id,
                codes_id = article['codes_id'],
                chapter_num = article['chapter_num'],
                chapter_title = article['chapter_title'],
                article_num = article['article_num'],
                article_title = article['title'],
                text = article['text'],
                comments = article['comments'],
                additional = article['additional'],
                court_links = article['court_links'],
                score = hit.score
            )
            results.append(result)
            article_ids.add(article_id)
        
        return results
    
    def search_laws_by_id(self, query: str, law_id: int, limit: int = 3) -> List[SearchResult]:
        """
        Поиск в статьях ФЗ по ID закона
        
        Args:
            query: поисковый запрос
            law_id: ID ФЗ
            limit: количество результатов
            
        Returns:
            список результатов поиска
        """
        # Создаем эмбеддинг для запроса
        query_embedding = self._create_embeddings(query, is_query=True)

        search_params = {
            "collection_name": self.laws_collection,
            "query_vector": query_embedding.tolist(),
            "limit": limit
        }

        if law_id is not None:
            search_params["query_filter"] = models.Filter(
                must=[
                    models.FieldCondition(
                        key="law_id",
                        match=models.MatchValue(value=law_id)
                    )
                ]
            )
        
        # Выполняем поиск с фильтром 
        search_result = self.client.search(**search_params)
        
        # Преобразуем результаты
        results = []
        article_ids = set()
        for hit in search_result:
            payload = hit.payload
            article_id = payload['article_id']

            if article_id in article_ids:
                continue

            article = self.postgres_base.get_law_article(article_id)

            result = LawArticle(
                article_id = article_id,
                law_id = article['law_id'],
                chapter_num = article['chapter_num'],
                chapter_title = article['chapter_title'],
                article_num = article['article_num'],
                article_title = article['title'],
                text = article['text'],
                comments = article['comments'],
                additional = article['additional'],
                court_links = article['court_links'],
                score = hit.score
            )
            results.append(result)
            article_ids.add(article_id)

        return results
    
    def get_collection_info(self):
        """Получение информации о коллекциях"""
        collections = [
            self.constitution_collection,
            self.codes_collection,
            self.laws_collection
        ]
        
        shapes = []

        for collection_name in collections:
            try:
                info = self.client.get_collection(collection_name)
                shapes.append(info.points_count)
                print(f"Коллекция {collection_name}: {info.points_count} точек")
            except Exception as e:
                print(f"Ошибка получения информации о коллекции {collection_name}: {e}")
        return shapes


# Добавление данных в бд
if __name__ == "__main__":
    rag_system = QdrantLegalRAG()
    
    shapes = rag_system.get_collection_info()

    if shapes[0] == 0:
        base_path = '/home/roman/jupyter_projects/ai_lawyer/parsing/data/'

        with open(base_path + 'contitution_articles.pickle', 'rb') as f:
            constitution_articles = pickle.load(f)

        with open(base_path + 'codes_articles.pickle', 'rb') as f:
            codes_articles = pickle.load(f)
        with open(base_path + 'codes_articles_2.pickle', 'rb') as f:
            codes_articles += pickle.load(f)

        with open(base_path + 'law_articles.pickle', 'rb') as f:
            laws_articles = pickle.load(f)
        with open(base_path + 'law_articles_2.pickle', 'rb') as f:
            laws_articles += pickle.load(f)
        
        # Добавление данных
        rag_system.add_constitution_articles(constitution_articles)
        rag_system.add_codes_articles(codes_articles)
        rag_system.add_laws_articles(laws_articles)
    
        # Получение информации о коллекциях
        rag_system.get_collection_info()
    
    # Пример поиска
    query = "государственная власть"
    results = rag_system.search_constitution_by_chapter(query, chapter_id=0, limit=3)
    
    print(f"Результаты поиска по запросу '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Релевантность: {result['score']:.3f}")
        print(f"   Чанк: {result['text']}")
        print(f"   Статья: {result['num']}")
        print()

    query = "государственная власть"
    results = rag_system.search_codes_by_id(query, codes_id=0, limit=3)
    
    print(f"Результаты поиска по запросу '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Релевантность: {result['score']:.3f}")
        print(f"   Чанк: {result['text']}")
        print(f"   Статья: {result['article_num']}")
        print()
    
    
    query = "государственная власть"
    results = rag_system.search_laws_by_id(query, law_id=0, limit=3)
    
    print(f"Результаты поиска по запросу '{query}':")
    for i, result in enumerate(results, 1):
        print(f"{i}. Релевантность: {result['score']:.3f}")
        print(f"   Чанк: {result['text']}")
        print(f"   Статья: {result['article_num']}")
        print()
