from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import sqlite3
from scipy.spatial.distance import cosine
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.prompts import EmbEstimatorPrompt
from database.psql_base import PostgresBase


llm_model = ChatOpenAI(
    base_url="http://192.168.1.93:2727/v1",
    model='Qwen2.5-7B-AWQ',
    openai_api_key='password',
    temperature=0.7,
    top_p=0.9,
    stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
    max_tokens=20_000
)

def questions_generate(total_questions):
    total_code_articles = 6345
    total_law_articles = 2720
    current_total_questions = 0

    postgres_base = PostgresBase()
    prompt = EmbEstimatorPrompt().prompt

    code_questions = []
    law_questions = []

    while current_total_questions < total_questions:
        print(current_total_questions)

        code_article_id = np.random.randint(low=0, high=total_code_articles)
        law_article_id = np.random.randint(low=0, high=total_law_articles)        

        code_article_text = postgres_base.get_code_article(code_article_id)
        law_article_text = postgres_base.get_law_article(law_article_id)

        if code_article_text is None or law_article_text is None:
            continue
        
        code_article_text = code_article_text['text']
        law_article_text = law_article_text['text']

        try:
            generated_code_question = llm_model.invoke(prompt(code_article_text)).content
            generated_law_question = llm_model.invoke(prompt(law_article_text)).content

        except Exception as e:
            continue

        code_questions.append((generated_code_question, code_article_id))
        law_questions.append((generated_law_question, law_article_id))

        current_total_questions += 1

    with open('code_questions.pickle', 'wb') as f:
        pickle.dump(code_questions, f)

    with open('law_questions.pickle', 'wb') as f:
        pickle.dump(law_questions, f)


class VectorDatabase:
    def __init__(self, db_path: str, model_name: str = 'intfloat/multilingual-e5-small'):
        """
        Инициализация векторной базы данных SQLite

        :param db_path: Путь к файлу базы данных SQLite
        :param model_name: Название модели для генерации эмбеддингов
        """
        self.db_path = db_path
        self.model = SentenceTransformer(model_name, device='cpu')

        self.postgres_base = PostgresBase()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self._create_table()

    def _create_table(self):
        """
        Создание таблицы для хранения векторов и текста
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    article_id INTEGER,
                    is_codes INTEGER
                )
            ''')
            conn.commit()

    @staticmethod
    def _convert_embedding_to_blob(embedding: np.ndarray) -> bytes:
        """
        Преобразование numpy-массива эмбеддингов в blob

        :param embedding: Numpy-массив эмбеддингов
        :return: Blob-представление эмбеддингов
        """
        return embedding.tobytes()

    @staticmethod
    def _blob_to_embedding(blob_data):
        """
        Преобразование blob-данных обратно в numpy-массив эмбеддингов

        :param blob_ Blob-данные эмбеддинга
        :return: Numpy-массив эмбеддингов
        """
        import numpy as np

        # Определение типа данных и формы эмбеддинга
        embedding = np.frombuffer(blob_data, dtype=np.float32)

        return embedding

    def insert_chunk(self, text: str, article_id: int, is_codes: bool):
        """
        Вставка чанка текста с его эмбеддингом

        :param text: Текст чанка
        :param filename: Имя файла источника
        """

        # Генерация эмбеддинга с добавлением префикса для инструкции
        embedding = self.model.encode(text)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO vectors (text, embedding, article_id, is_codes) VALUES (?, ?, ?, ?)',
                (text, self._convert_embedding_to_blob(embedding), article_id, int(is_codes))
            )
            conn.commit()

    def fill_database(self):
        
        codes_articles = self.postgres_base.select_code_articles_id_text()
        laws_articles = self.postgres_base.select_law_articles_id_text()

        for article in tqdm(codes_articles):
            if article['text'] == '':
                continue
            
            chunks = self.text_splitter.split_text(article['text'])

            for chunk in chunks:
                self.insert_chunk(chunk, article['id'], is_codes=True)

        for article in tqdm(laws_articles):
            if article['text'] == '':
                continue
            
            chunks = self.text_splitter.split_text(article['text'])

            for chunk in chunks:
                self.insert_chunk(chunk, article['id'], is_codes=False)
    
    def cosine_similarity_search(
            self,
            query: str,
            is_code: bool,
            top_k: int = 5,
            similarity_threshold: float = 0.5) -> (list, list):
        """
        Поиск наиболее похожих чанков с использованием косинусного расстояния

        :param query: Текстовый запрос
        :param top_k: Максимальное количество возвращаемых результатов
        :param similarity_threshold: Порог схожести
        :return: Список найденных чанков
        """

        # query_embedding = self.model.encode(f'query: {query}')
        query_embedding = self.model.encode(query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if is_code:
            cursor.execute('SELECT * FROM vectors WHERE is_codes == 1') 
        else:
            cursor.execute('SELECT * FROM vectors WHERE is_codes == 0') 

        rows = cursor.fetchall()

        texts = []
        embeddings = []
        ids = []

        for record in rows:
            # Преобразование blob обратно в numpy-массив
            embedding_array = np.frombuffer(record[2], dtype=np.float32)

            texts.append(record[1])
            embeddings.append(embedding_array)
            ids.append(record[3])

        cosine_similarity = [cosine(embedding, query_embedding) for embedding in embeddings]

        sort_texts = [
            (find_text, filename) for find_text, _, filename in sorted(
                zip(texts, cosine_similarity, ids),
                key=lambda x: x[1]
            )]
        return [find_text for find_text, _ in sort_texts], [filename for _, filename in sort_texts]

if __name__ == '__main__':
    base = VectorDatabase(
        db_path='metrics/articles.db',
        model_name="Qwen/Qwen3-Embedding-0.6B"
        )
    base.fill_database()

    with open('embedder_compare/code_questions.pickle', 'rb') as f:
        code_questions = pickle.load(f)
    with open('embedder_compare/law_questions.pickle', 'rb') as f:
        law_questions = pickle.load(f)

    code_idx = []
    for question, article_id in tqdm(code_questions):
        _, ids = base.cosine_similarity_search(question, is_code=True)

        code_idx.append(ids.index(article_id))
    
    law_idx = []
    for question, article_id in tqdm(law_questions):
        _, ids = base.cosine_similarity_search(question, is_code=False)

        law_idx.append(ids.index(article_id))
    
    print(f'codes: {np.mean(code_idx)}, {np.median(code_idx)}')
    print(f'laws: {np.mean(law_idx)}, {np.median(law_idx)}')
