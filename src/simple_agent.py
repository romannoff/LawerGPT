from langchain_openai import ChatOpenAI
import os
from src.agent import Agent, AgentAnswer
from src.prompts import SimplePrompts

class SimpleAgent(Agent):
    def __init__(self):
        self.llm_model = ChatOpenAI(
            base_url="http://192.168.1.93:2727/v1",
            model='Qwen2.5-7B-AWQ',
            openai_api_key='password',
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["<|im_end|>", "<|im_start|>", "<|eot_id|>"],
            max_tokens=10000,
        )
        # self.prompt = SimplePrompts().simple_answer
        self.prompt = SimplePrompts().prompt_upgrade

    def __call__(self, query):
        prompt = self.prompt(query)
        answer = self.llm_model.invoke(prompt).content
        return AgentAnswer(
            query=query,
            context=[],
            answer=answer,
        )
        

if __name__ == '__main__':
    s = SimpleAgent()
    print(s('Истрин отказался подписывать протокол об административном правонарушении, не согласившись с его содержанием, и потребовал выдать копию под расписку. Начальник погранзаставы отказал, сославшись на наличие формулировки «С протоколом ознакомлен, согласен». Правомерны ли действия начальника? Какие права имеет лицо, в отношении которого составлен протокол?').content)