from rouge import Rouge
from bert_score import score
import numpy as np
from typing import TypedDict
from tqdm import tqdm
import os
from openai import OpenAI
import json

from src.agent import Agent, AgentAnswer
from src.simple_agent import SimpleAgent
from src.rag_agent import RagAgent
from src.prompts import EvaluetePromptWithTrueAnswer

class CorrectnessAgent:
    def __init__(self, question_per_file=15):
        self.so_llm_model = OpenAI(
            base_url="http://192.168.1.93:2727/v1", 
            api_key="password",
            )
        self.question_per_file = question_per_file
        self.prompt = EvaluetePromptWithTrueAnswer().prompt
    
    def __call__(self, agent: Agent):
        files = os.listdir('qa_tests/')

        for file in files:
            print(file)
            if file in ['QA_pravo.txt', 'QA_trud_pravo.txt', 'QA_gr_pravo.txt', 'QA_UKRF.txt']:
                continue
            q, a = 0, 0

            file_results = []

            with open('qa_tests/' + file, 'r', encoding='utf-8') as f:
                is_question = False

                current_question = ''
                current_answer = ''

                for line in f:
                    if line.startswith('Q'):
                        is_question = True

                        if current_question != '':
                            print(f'{q}/{self.question_per_file}')
                            generated_answer = agent(current_question)['answer']

                            with open('logs/last_llm_answer.txt', 'w') as f:
                                f.write(generated_answer)
                            
                            prompt, extra_body = self.prompt(current_question, current_answer, generated_answer)

                            completion = self.so_llm_model.chat.completions.create(
                                model='Qwen2.5-7B-AWQ',
                                messages=prompt,
                                extra_body=extra_body,
                                temperature=0,
                                max_tokens=5_000,
                            )
                            res = json.loads(completion.choices[0].message.content)
                            file_results.append(int(res['score']))
                            print(int(res['score']))
                                
                        if q == self.question_per_file:
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

            print('SCORE')
            print(f'{sum(file_results)}/{4 * len(file_results)}')
            print(len([1 for res in file_results if res > 0])) 


if __name__ == '__main__':
    # agent = SimpleAgent()
    agent = RagAgent()
    e = CorrectnessAgent(question_per_file=10)
    e(agent)
    