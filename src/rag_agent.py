from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.schema import HumanMessage
import json

from database.vbase import QdrantLegalRAG
from database.history_base import HistoryBase
from src.agent import Agent, AgentAnswer
from database.id_to_str import codes_id_to_str, laws_id_to_str
from src.prompts import RagPrompts, SummarizePrompt, RouterPrompt

class RagAgent(Agent):
    def __init__(self):
        self.database = QdrantLegalRAG()
        self.history_base = HistoryBase()

        self.llm_model = ChatOpenAI(
            base_url="http://192.168.1.93:2727/v1",
            model='Qwen2.5-7B-AWQ',
            openai_api_key='password',
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
        )

        self.so_llm_model = OpenAI(
            base_url="http://192.168.1.93:2727/v1", 
            api_key="password",
            )

        self.final_prompt = RagPrompts().simple_prompt

        self.summaryze_prompts = SummarizePrompt()
        self.summarize_prompt = self.summaryze_prompts.prompt
        self.reranker_prompt = self.summaryze_prompts.reranker_prompt
        
        self.router_prompt = RouterPrompt().prompt
        
        self.summarize_articles = True

    def get_article_text(self, articles: list[dict], article_type: str):
        if not articles:
            return ''
        result = 'Конституция РФ:\n\n' if article_type == 'constitution' else ''
        for article in articles:
            # Заголовок
            if article.get('title') is not None:
                result += article['title']
            else:
                result += self.get_title(article, article_type)

            # Текст
            if article.get('text') is not None:
                result += article['text'] + '\n\n'
            elif article.get('summarize') is not None:
                result += article['summarize'] + '\n\n'
        return result


    @staticmethod
    def get_title(article, article_type):
        if article_type == 'constitution':
            return article['num'] + '\n'
        
        if article_type == 'code':
            result = codes_id_to_str(article['codes_id']) + '\n'
            
        elif article_type == 'law':
            result = laws_id_to_str(article['law_id']) + '\n'

        result += article['chapter_num'] + ' ' + article['article_title'] + '. ' + article['article_num'] + '\n'
        return result

    def summarize(self, query, articles, article_type):

        for article in articles:
            text = self.get_title(article, article_type=article_type) + article['text']

            prompt = self.summarize_prompt(query, text)

            summarize_text = self.llm_model.invoke(prompt).content
            if summarize_text.startswith('В данной статье нет информации по запросу'):
                article['text'] = ''
            else:
                article['text'] = summarize_text

    def reranker(self, query, articles, article_type):
        filtered_articles = []
        for article in articles:
            text = self.get_title(article, article_type=article_type) + article['text']

            prompt, extra_body = self.reranker_prompt(query, text)

            completion = self.so_llm_model.chat.completions.create(
                model='Qwen2.5-7B-AWQ',
                messages=prompt,
                extra_body=extra_body,
                temperature=0.7,
                max_tokens=10_000,
            )
            res = json.loads(completion.choices[0].message.content)
            if int(res['estimation']) > 0:
                res['title'] = self.get_title(article, article_type)
                filtered_articles.append(res)
        
        return sorted(filtered_articles, key=lambda x: int(x['estimation']), reverse=True)

    def router(self, query):
        # messages = self.history_base.get_messages(chat_id=0, limit=8)
        messages = None

        prompt, extra_body = self.router_prompt(query, messages)
        completion = self.so_llm_model.chat.completions.create(
                model='Qwen2.5-7B-AWQ',
                messages=prompt,
                extra_body=extra_body,
                temperature=0.2,
                max_tokens=500,
        )
        # print(completion.choices[0].message.content)
        answer = completion.choices[0].message.content
        try:
            res = json.loads(answer)
        except json.decoder.JSONDecodeError:
            if 'non_legal' in answer:
                return 'non_legal', 'Я готов ответить на ваши юридические вопросы'
            return 'legal', query

        return res['msg_type'], res['response']

    def __call__(self, query):
        msg_type, response = self.router(query)

        self.history_base.add_message(role='user', text=query, chat_id=0)

        if msg_type == 'non_legal':
            self.history_base.add_message(role='assistant', text=response, chat_id=0)
            return AgentAnswer(
                    query=query,
                    context=[],
                    answer=response,
                )

        query = response

        # Получаем чанки из баз
        constitutions = self.database.search_constitution_by_chapter(query, chapter_id=None, limit=10)
        codes = self.database.search_codes_by_id(query, codes_id=None, limit=10)
        laws = self.database.search_laws_by_id(query, law_id=None, limit=10)

        constitutions = self.reranker(query, constitutions, article_type='constitution')
        codes = self.reranker(query, codes, article_type='code')
        laws = self.reranker(query, laws, article_type='law')

        # self.summarize(query, constitutions, article_type='constitution')
        # self.summarize(query, codes, article_type='code')
        # self.summarize(query, laws, article_type='law')

        constitution_str = self.get_article_text(constitutions, article_type='constitution')
        codes_str = self.get_article_text(codes, article_type='code')
        law_str = self.get_article_text(laws, article_type='law')

        prompt = self.final_prompt(
            query, 
            constitution_str,
            codes_str,
            law_str,
            )

        answer = self.llm_model.invoke(prompt).content

        self.history_base.add_message(role='assistant', text=answer, chat_id=0)
        
        return AgentAnswer(
            query=query,
            context=[constitution_str, codes_str, law_str],
            answer=answer,
        )

        
        
        

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



