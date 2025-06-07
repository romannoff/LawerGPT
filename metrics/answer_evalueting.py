from rouge import Rouge
from bert_score import score
import numpy as np
from typing import TypedDict
from tqdm import tqdm
import os

from src.agent import Agent, AgentAnswer
from src.simple_agent import SimpleAgent
from src.rag_agent import SimpleRagAgent

class Score(TypedDict):
    rouge_1: float
    rouge_2: float
    rouge_l: float


    P: float
    R: float
    F1: float


class EvaluateAgent:
    def __init__(self, question_per_file=15):
        self.rouge = Rouge()
        self.question_per_file = question_per_file
        
    def rouge_score(self, generated_answer, reference_answer):
        return self.rouge.get_scores(generated_answer, reference_answer)
    
    def bert_score(self, generated_answer, reference_answer):
        return score([generated_answer], [reference_answer], lang="ru", device="cpu")
    
    def __call__(self, agent: Agent):
        files = os.listdir('qa_tests/')

        results = []

        for file in files:
            print(file)
            q, a = 0, 0

            rouge_1_scores = []
            rouge_2_scores = []
            rouge_l_scores = []


            Ps = []
            Rs = []
            F1s = []

            with open('qa_tests/' + file, 'r', encoding='utf-8') as f:
                is_question = False

                current_question = ''
                current_answer = ''

                for line in f:
                    if line.startswith('Q'):
                        is_question = True

                        if current_question != '':
                            print(f'{q}/{self.question_per_file}')
                            # print('-'*100)
                            # print(current_question)
                            # print(current_answer)
                            generated_answer = agent(current_question)['answer']

                            with open('logs/last_llm_answer.txt', 'w') as f:
                                f.write(generated_answer)
                            try:
                                rouge_score = self.rouge_score(generated_answer, current_answer)[0]
                                P, R, F1 = self.bert_score(generated_answer, current_answer)

                                rouge_1_scores.append(rouge_score['rouge-1']['f'])
                                rouge_2_scores.append(rouge_score['rouge-2']['f'])
                                rouge_l_scores.append(rouge_score['rouge-l']['f'])

                                Ps.append(float(P[0]))
                                Rs.append(float(R[0]))
                                F1s.append(float(F1[0]))

                            except RecursionError:
                                print('Зациклились')
                                rouge_1_scores.append(0)
                                rouge_2_scores.append(0)
                                rouge_l_scores.append(0)

                                Ps.append(0)
                                Rs.append(0)
                                F1s.append(0)
                                
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

            result = Score(
                rouge_1=np.mean(rouge_1_scores),
                rouge_2=np.mean(rouge_2_scores),
                rouge_l=np.mean(rouge_l_scores),
                P=np.mean(Ps),
                R=np.mean(Rs),
                F1=np.mean(F1s)
            )
            print('='*20, 'Scores', '='*20)
            print('rouge-1:', result['rouge_1'])
            print('rouge-2:', result['rouge_2'])
            print('rouge-l:', result['rouge_l'])
            print('-'*20, 'Bert Score', '-'*20)
            print('P:', result['P'])
            print('R:', result['R'])
            print('F1:', result['F1'])



            results.append(result)


        return results

if __name__ == '__main__':
    # agent = SimpleAgent()
    agent = SimpleRagAgent()

    e = EvaluateAgent()
    e(agent)