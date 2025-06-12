from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.schema import HumanMessage
import json

from database.vbase import QdrantLegalRAG
from database.history_base import HistoryBase
from src.agent import Agent, AgentAnswer
from database.id_to_str import codes_id_to_str, laws_id_to_str
from src.prompts import RagPrompts, SummarizePrompt, RouterPrompt
from src.logging_conf import logger
from src.config import Config


settings = Config.from_yaml("config.yaml")


class RagAgent(Agent):
    def __init__(self, 
                answer_temperature: float=0.7,
                router_temperature: float=0.2,
                reranker_temperature: float=0.7, 
                chunk_count: int=10,
                reranker_enable: bool=True,
                router_enable: bool=True,
                db_enable: bool=True,
                is_test: bool=False,
                chain_of_thoughts: bool=False,
                chat_id: str='0',
                mode: str='standard'
                ):
        self.database = QdrantLegalRAG()
        self.history_base = HistoryBase()

        self.answer_temperature = answer_temperature
        self.router_temperature = router_temperature
        self.reranker_temperature = reranker_temperature
        self.reranker_enable = reranker_enable
        self.router_enable = router_enable
        self.db_enable = db_enable
        self.chunk_count = chunk_count
        self.is_test = is_test
        self.chain_of_thoughts = chain_of_thoughts
        self.chat_id = chat_id

        self.llm_model = ChatOpenAI(
            base_url=settings.base_url,
            model=settings.model,
            openai_api_key=settings.password,
            temperature=self.answer_temperature,
            stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
        )

        self.so_llm_model = OpenAI(
            base_url=settings.base_url,
            api_key=settings.password,
        )

        self.final_prompt = RagPrompts().simple_prompt
        self.summaryze_prompts = SummarizePrompt()
        self.summarize_prompt = self.summaryze_prompts.prompt
        self.reranker_prompt = self.summaryze_prompts.reranker_prompt
        self.router_prompt = RouterPrompt().prompt
        self.summarize_articles = True

    def set_settings(self, **kwargs):
        self.answer_temperature = kwargs.get('answer_temperature', self.answer_temperature)
        self.router_temperature = kwargs.get('router_temperature', self.router_temperature)
        self.reranker_temperature = kwargs.get('reranker_temperature', self.reranker_temperature)
        self.reranker_enable = kwargs.get('reranker_enable', self.reranker_enable)
        self.router_enable = kwargs.get('router_enable', self.router_enable)
        self.db_enable = kwargs.get('db_enable', self.db_enable)
        self.chunk_count = kwargs.get('chunk_count', self.chunk_count)
        self.is_test = kwargs.get('is_test', self.is_test)
        self.chain_of_thoughts = kwargs.get('chain_of_thoughts', self.chain_of_thoughts)
        self.chat_id = kwargs.get('chat_id', self.chat_id)

    def get_article_text(self, articles: list[dict], article_type: str):
        """
        Преобразует полученные чанки в текст для промпта
        """
        if not articles:
            return ''
        result = 'Конституция РФ:\n\n' if article_type == 'constitution' else ''
        for article in articles:
            if article.get('title') is not None:
                result += article['title']
            else:
                result += self.get_title(article, article_type)
            result += article['text'] + '\n\n'
        return result

    @staticmethod
    def get_title(article, article_type):
        """
        Получает точное название статьи
        """
        if article_type == 'constitution':
            return article['num'] + '\n'
        if article_type == 'code':
            result = codes_id_to_str(article['codes_id']) + '\n'
        elif article_type == 'law':
            result = laws_id_to_str(article['law_id']) + '\n'
        result += article['chapter_num'] + ' ' + article['article_title'] + '. ' + article['article_num'] + '\n'
        return result

    def summarize(self, query, articles, article_type):
        """
        Суммаризирует текст статьи, выделяя только нужные фрагменты
        """
        for article in articles:
            text = self.get_title(article, article_type=article_type) + article['text']
            prompt = self.summarize_prompt(query, text)
            summarize_text = self.llm_model.invoke(prompt).content
            if summarize_text.startswith('В данной статье нет информации по запросу'):
                article['text'] = ''
            else:
                article['text'] = summarize_text

    def reranker(self, query, articles, article_type):
        """
        Суммаризирует текст списка статей, выбирая только нужную информацию.
        Помимо этого присуждает каждой статье оценку:
        0 - статья не подходит под вопрос,
        2 - статья частично подходит под вопрос,
        4 - статья полностью подходит под вопрос 
        """
        if not articles:
            return []

        filtered_articles = []
        for article in articles:
            title = self.get_title(article, article_type=article_type)
            text = title + article['text']
            prompt, extra_body = self.reranker_prompt(query, text)
            completion = self.so_llm_model.chat.completions.create(
                model=settings.model,
                messages=prompt,
                extra_body=extra_body,
                temperature=self.reranker_temperature,
                max_tokens=10_000,
            )
            res = json.loads(completion.choices[0].message.content)

            logger.info('RERANKER: %s\nSCORE: %s\nTEXT:%s\n\n' % (title, res['estimation'], res['summarize']))
            
            if not self.reranker_enable:
                filtered_articles.append({
                    'title': title,
                    'text': res['summarize'],
                    'relevance_score': 0
                })
            
            elif int(res['estimation']) > 0:
                filtered_articles.append({
                    'title': title,
                    'text': res['summarize'],
                    'relevance_score': int(res['estimation'])
                })
        return sorted(filtered_articles, key=lambda x: x['relevance_score'], reverse=True)

    def router(self, query):
        """
        Роутер, переформулирующий вопрос, исходя из истории, если вопрос качается юридической темы,
        иначе даёт пользователю ответ средствами LLM.
        """
        if not self.router_enable:
            return 'legal', query
        if not self.is_test:
            messages = self.history_base.get_messages(chat_id=self.chat_id, limit=4)
        else:
            messages = []

        prompt, extra_body = self.router_prompt(query, messages)
        completion = self.so_llm_model.chat.completions.create(
            model=settings.model,
            messages=prompt,
            extra_body=extra_body,
            temperature=self.answer_temperature,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content  

        try:
            res = json.loads(answer)
        except json.decoder.JSONDecodeError:
            logger.warning('ROUTER: JSONDecodeError\nRESULT: %s\n\n' % (answer, ))
            if 'non_legal' in answer:
                return 'non_legal', 'Я готов ответить на ваши юридические вопросы'
            return 'legal', query
        
        logger.info('ROUTER:\nTYPE: %s\nNEW QUESTION: %s\n\n' % (res['msg_type'], res['response']))
        
        return res['msg_type'], res['response']

    def __call__(self, query):
        """
        Получение ответа от RAG на запрос пользователя
        """

        msg_type, response = self.router(query)
        if not self.is_test:
            self.history_base.add_message(role='user', text=query, chat_id=self.chat_id)
        if msg_type == 'non_legal' and not self.is_test:
            self.history_base.add_message(role='assistant', text=response, chat_id=self.chat_id)
            return {'answer': response, 'contexts': []}

        query = response
        if self.db_enable:
            limit = self.chunk_count
            constitutions = self.database.search_constitution_by_chapter(query, chapter_id=None, limit=limit)
            codes = self.database.search_codes_by_id(query, codes_id=None, limit=limit)
            laws = self.database.search_laws_by_id(query, law_id=None, limit=limit)
        else:
            constitutions = []
            codes = []
            laws = []

        constitutions = self.reranker(query, constitutions, article_type='constitution')
        codes = self.reranker(query, codes, article_type='code')
        laws = self.reranker(query, laws, article_type='law')
        
        contexts = constitutions + codes + laws

        constitution_str = self.get_article_text(constitutions, article_type='constitution')
        codes_str = self.get_article_text(codes, article_type='code')
        law_str = self.get_article_text(laws, article_type='law')

        prompt = self.final_prompt(query, constitution_str, codes_str, law_str)
        # print(prompt)
        answer = self.llm_model.invoke(prompt).content

        if not self.is_test:
            self.history_base.add_message(role='assistant', text=answer, chat_id=self.chat_id)
        print(contexts)
        return {'answer': '\n' + answer, 'contexts': contexts}
    

if __name__ == '__main__':
    s = RagAgent()
    print(s('Истрин отказался подписывать протокол об административном правонарушении, не согласившись с его содержанием, и потребовал выдать копию под расписку. Начальник погранзаставы отказал, сославшись на наличие формулировки «С протоколом ознакомлен, согласен». Правомерны ли действия начальника? Какие права имеет лицо, в отношении которого составлен протокол?')['answer'])

    print('_'*30)

    print(s('Привет, какая погода сегодня в Москве?')['answer'])

    print('_'*30)

    print(s('Что будет, если я травмировал в драке человека?')['answer'])

    print('_'*30)

    print(s('А если я оборонялся?')['answer'])

    print('_'*30)

    print(s('А если я был в состоянии аффекта?')['answer'])
    