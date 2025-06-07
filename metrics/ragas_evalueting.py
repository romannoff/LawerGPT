import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    AnswerAccuracy,
)
from typing import List, Dict, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from ragas.embeddings import LangchainEmbeddingsWrapper

from src.agent import Agent, AgentAnswer
from src.rag_agent import SimpleRagAgent

class RagasEval:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.llm_model = ChatOpenAI(
            base_url="http://192.168.1.93:2727/v1",
            model='Qwen2.5-7B-AWQ',
            openai_api_key='password',
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
            max_tokens=10_000,
            request_timeout=200, 
            max_retries=3
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            model_kwargs={"device": "cpu"},  # Используйте "cuda", если есть GPU
            encode_kwargs={"normalize_embeddings": True}  # Нормализация для лучшего качества
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embedding_model)

        self.answer_accuracy = AnswerAccuracy()

    @staticmethod
    def parse_qa_file(file_path: str, limit_per_file: int = None) -> List[Tuple[str, str]]:
        """
        Парсинг файла с вопросами и ответами
        
        Args:
            file_path: путь к файлу с Q&A
            
        Returns:
            Список кортежей (question, answer)
        """
        q, a = 0, 0
        qa_pairs = []
        is_question = False

        current_question = ''
        current_answer = ''
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('Q'):
                    is_question = True

                    if current_question != '':
                        qa_pairs.append((current_question, current_answer))

                    if limit_per_file is not None and q == limit_per_file:
                        break

                    current_question = ''
                    line = line.replace('Q: ', '')
                    q += 1

                elif line.startswith('A'):
                    a += 1
                    is_question = False
                    current_answer = ''
                    line = line.replace('A: ', '')
                    
                
                if is_question:
                    current_question += line
                else:
                    current_answer += line
        
        return qa_pairs
    
    def load_all_qa_data(self, qa_directory: str, limit_per_file: int = None) -> List[Tuple[str, str]]:
        """
        Загрузка всех Q&A данных из директории
        
        Args:
            qa_directory: путь к директории с файлами Q&A
            limit_per_file: ограничение количества вопросов на файл
            
        Returns:
            Список всех пар (question, answer)
        """

        all_qa_pairs = []
        
        for filename in os.listdir(qa_directory):
            if filename.endswith('.txt'):
                file_path = os.path.join(qa_directory, filename)
                qa_pairs = self.parse_qa_file(file_path, limit_per_file)
                
                if limit_per_file:
                    qa_pairs = qa_pairs[:limit_per_file]
                
                all_qa_pairs.extend(qa_pairs)
                print(f"Загружено {len(qa_pairs)} Q&A пар из {filename}")
        
        return all_qa_pairs
    
    def prepare_ragas_dataset(self, questions: List[str], ground_truths: List[str], rag_responses: List[Dict]) -> Dataset:
        """
        Подготовка датасета в формате RAGAS
        
        Args:
            questions: список вопросов
            ground_truths: список эталонных ответов
            rag_responses: список ответов RAG системы с контекстом
            
        Returns:
            Dataset для оценки RAGAS
        """
        data = {
            'question': questions,
            'answer': [resp['answer'] for resp in rag_responses],
            'contexts': [resp['contexts'] for resp in rag_responses],
            'ground_truth': ground_truths
        }
        
        return Dataset.from_dict(data)

    def generate_rag_responses(self, questions: List[str]) -> List[Dict]:
        """
        Генерация ответов через RAG агента с извлечением контекста
        
        Args:
            agent: RAG агент для генерации ответов
            questions: список вопросов
            
        Returns:
            Список словарей с ответами и контекстом
        """
        responses = []
        
        for question in tqdm(questions, desc="Генерация ответов"):
            response = self.agent(question)
            
            contexts = response['context']
            
            responses.append({
                'answer': response['answer'],
                'contexts': contexts
            })
        
        return responses

    def eval(self, limit_per_file: int = 15):
        qa_pairs = self.load_all_qa_data('qa_tests/', limit_per_file)

        questions = [pair[0] for pair in qa_pairs]
        ground_truths = [pair[1] for pair in qa_pairs]

        rag_responses = self.generate_rag_responses(questions)

        dataset = self.prepare_ragas_dataset(questions, ground_truths, rag_responses)

        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                self.answer_accuracy,
            ],
            llm=self.llm_model,
            embeddings=self.ragas_embeddings
        )
        print(result)

if __name__ == '__main__':
    agent = SimpleRagAgent()

    ragas_eval = RagasEval(agent)
    ragas_eval.eval()