from langchain.schema import SystemMessage, HumanMessage
from transformers import AutoTokenizer
from pydantic import BaseModel
from enum import Enum

from src.logging_conf import logger


class SimplePrompts:
    @staticmethod
    def user_question(question):
        return f"\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}"

    def prompt_upgrade(self, question):
        system = "Ты - виртуальный юрист-консультант, специализирующийся на российском законодательстве. Твоя задача - предоставить точный и обоснованный юридический анализ на основе предоставленных нормативных документов.\n"
        istructions = """
ИНСТРУКЦИИ:
1. Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста. Не используй информацию, которой нет в контексте. НЕ ИСПОЛЬЗУЙ В ОТВЕТЕ СТАТЬИ, КОТОРЫЕ НЕ ОТНОСЯТСЯ К ВОПРОСУ ПОЛЬЗОВАТЕЛЯ.

2. Структурируй ответ следующим образом:
- Краткий ответ на вопрос (2-3 предложения)
- Правовое обоснование со ссылками на конкретные статьи и пункты
- Важные нюансы и исключения (если применимо)
- Практические рекомендации

3. При цитировании ОБЯЗАТЕЛЬНО указывай:
- Точный номер статьи/пункта/части
- Название нормативного акта
- Дословную цитату в кавычках (для ключевых положений)

4. Если в контексте недостаточно информации для полного ответа:
- Четко укажи, какой информации не хватает
- Дай частичный ответ на основе имеющихся данных
- Предупреди о необходимости дополнительной консультации

5. Используй понятный язык:
- Разъясняй юридические термины простыми словами
- Приводи примеры для сложных концепций
- Избегай излишне формального стиля

6. Учитывай иерархию нормативных актов:
- Конституция имеет высшую юридическую силу
- При противоречиях указывай на приоритет норм

7. НЕ ДЕЛАЙ следующего:
- Не придумывай статьи или их содержание
- Не давай советов, выходящих за рамки правовой информации
- Не делай категоричных выводов при неоднозначности норм
- Не заменяй профессиональную юридическую консультацию

8 Отвечай ТОЛЬКО на РУССКОМ языке
"""
        answer_format = """
ФОРМАТ ОТВЕТА:
**Краткий ответ:**
[Основной вывод]

**Правовое обоснование:**
[Анализ с цитатами и ссылками]

**Важные моменты:**
[Нюансы, исключения, особенности]

**Рекомендации:**
[Практические шаги, если применимо]

**Ограничения ответа:**
[Указание на недостающую информацию, если есть]
""" 
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=self.user_question(question) + istructions + answer_format)
        ]
        return messages


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ", trust_remote_code=True)
max_model_len=50_000
max_gen_len=10_000

def is_prompt_too_long(prompt):
    n_prompt_tokens = len(tokenizer.encode(prompt[0].content)) + len(tokenizer.encode(prompt[1].content))
    return n_prompt_tokens + max_gen_len > max_model_len


class RagPrompts:
    system = "Ты - виртуальный юрист-консультант, специализирующийся на российском законодательстве. Твоя задача - предоставить точный и обоснованный юридический анализ на основе предоставленных нормативных документов.\n"

    # 4. Если в контексте недостаточно информации для полного ответа:
    # - Четко укажи, какой информации не хватает
    # - Дай частичный ответ на основе имеющихся данных
    # - Предупреди о необходимости дополнительной консультации

    istructions = """
ИНСТРУКЦИИ:
1. Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста. Не используй информацию, которой нет в контексте. 

2. Структурируй ответ следующим образом:
- Краткий ответ на вопрос (2-3 предложения)
- Правовое обоснование со ссылками на конкретные статьи и пункты

3. При цитировании ОБЯЗАТЕЛЬНО указывай:
- Точный номер статьи/пункта/части
- Название нормативного акта
- Дословную цитату в кавычках (для ключевых положений)

7. НЕ ДЕЛАЙ следующего:
- Не придумывай статьи или их содержание.
- НЕ ИСПОЛЬЗУЙ В ОТВЕТЕ СТАТЬИ, КОТОРЫЕ НЕ ОТНОСЯТСЯ К ВОПРОСУ ПОЛЬЗОВАТЕЛЯ.

8. Отвечай ТОЛЬКО на РУССКОМ языке
"""
    answer_format = """
ФОРМАТ ОТВЕТА:
**Краткий ответ:**
[Основной вывод]

**Правовое обоснование:**
[Анализ с цитатами и ссылками]

**Важные моменты:**
[Нюансы, исключения, особенности]
""" 

    @staticmethod
    def user_question(question):
        return f"\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}"
    
    @staticmethod
    def context(constitution_context: str, code_context: str, law_context: str):
        return f"\n\nКОНТЕКСТ:\n\n{constitution_context}\n\n{code_context}\n\n{law_context}"
    
    def simple_prompt(self, question, constitution_context: str, code_context: str, law_context: str):
        messages = [
            SystemMessage(content=self.system),
            HumanMessage(content=self.istructions + self.answer_format + self.context(constitution_context, code_context, law_context) + self.user_question(question))
        ]

        while is_prompt_too_long(messages):
            messages[1] = HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 1.5)])
        
        return messages
    

class EmbEstimatorPrompt:
    system = f"Проанализируй данный тебе текст юридичекой статьи и задай по данной статье любой один вопрос, ответ на который будет находится в предоставленной статье. " \
             f"В ответе укажи только один вопрос.\n"
    def prompt(self, text):
        messages = [
            SystemMessage(content=self.system),
            HumanMessage(content=f"\nТекст статьи: {text}\nТвой вопрос:")
        ]
        return  messages


class SummarizePrompt:
    system = """Ты - юридический аналитик, специализирующийся на извлечении релевантной информации из правовых документов.
"""

    task = """ЗАДАЧА:
Проанализируй предоставленную юридическую статью и извлеки ТОЛЬКО ту информацию, которая непосредственно относится к запросу пользователя.
"""
    reranker_task = """ЗАДАЧИ:
1. Извлеки ТОЛЬКО ту информацию, которая непосредственно относится к запросу пользователя.
2. Проанализируй предоставленную юридическую статью и оцени, насколько юридическая статья соответствует запросу пользователя

"""

    scale = """ШКАЛА ОЦЕНКИ:
- 0: Статья НЕ относится к запросу (другая тема, другая отрасль права)
- 2: Статья ЧАСТИЧНО относится к запросу (касается темы косвенно, содержит общие положения, но не дает прямого ответа)
- 4: Статья отвечает на запрос или ЧАСТЬ ЗАПРОСА (содержит конкретный ответ, регулирует именно запрашиваемую ситуацию)
"""

    instructions = """ИНСТРУКЦИИ ПО СУММАРИЗАЦИИ:
1. Внимательно прочитай запрос пользователя и определи ключевые юридические аспекты вопроса
2. Изучи полный текст статьи
3. Извлеки только те части статьи, которые:
- Прямо отвечают на вопрос пользователя
- Содержат необходимые для ответа определения, условия или исключения
- Устанавливают права, обязанности или ответственность по теме запроса
4. Сохрани точные формулировки и юридическую терминологию
5. Если оценка 0: в поле "summary" укажи "Статья не содержит информации по запросу"
6. Если статья не содержит в себе ответа на вопрос пользователя, то укажи "Статья не содержит информации по запросу"
7. Отвечай ТОЛЬКО на РУССКОМ языке
"""

    answer_format = """ФОРМАТ ОТВЕТА:
- Используй нумерацию пунктов/подпунктов из оригинальной статьи
- Сохраняй структуру статьи (пункты, подпункты)
- НЕ добавляй собственные комментарии или интерпретации
- НЕ включай информацию, не относящуюся к запросу
"""

    reranker_answer_format = """
-Верни ТОЛЬКО валидный JSON без дополнительного текста:
{
"summary": "<релевантный текст из статьи или сообщение об отсутствии информации>"
"score": <число 0, 2 или 4>,
}

"""

    @staticmethod
    def user_question(question):
        return f"\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}"
    
    @staticmethod
    def article_text(text):
        return f"\n\nТЕКСТ СТАТЬИ:\n{text}\n\nРЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ИЗ СТАТЬИ:\n"

    def prompt(self, question, text):
        messages = [
            SystemMessage(content=self.system + self.task + self.instructions + self.answer_format),
            HumanMessage(content=self.user_question(question) + self.article_text(text))
        ]
        return  messages
    
    def reranker_prompt(self, question, text):
        # messages = [
            # {"role": "system", "content": self.system + self.reranker_task + self.scale + self.instructions 
                        #   + self.answer_format + self.reranker_answer_format},
            # {"role": "user", "content": self.user_question(question) + self.article_text(text)},
        # ]

        messages = [
            SystemMessage(content=self.system + self.reranker_task + self.scale + self.instructions 
                          + self.answer_format + self.reranker_answer_format),
            HumanMessage(content=self.user_question(question) + self.article_text(text))
        ]

        while is_prompt_too_long(messages):
            messages[1] = HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 2)])
            # {"role": "user", "content": messages[1]['content'][:int(len(messages[1]['content']) / 1.5)]},
            # HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 1.5)])

        messages = [
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content}
        ]    

        json_schema = TextAnalyze.model_json_schema()
        extra_body = {"guided_json": json_schema}
        return  messages, extra_body


class Estimation(str, Enum):
    bad = 0
    normal = 2
    good = 4

class TextAnalyze(BaseModel):
    summarize: str
    estimation: Estimation


class RouterPrompt:
    system = "Ты - роутер в специализированной юридической системе. Твоя задача - анализировать запросы пользователей и принимать решения о дальнейшей обработке. Отвечай ТОЛЬКО на РУССКОМ языке"

    system2 = """Твоя задача - проанализировать вопрос пользователя, его историю диалога и сделать следующее:
1. Определить, к какой категории относится вопрос.
2. Сформулировать поисковый запрос, который наилучшим образом описывает суть вопроса пользователя.

Категории запроса:
1) non_legal - НЕРЕЛЕВАНТНЫЙ ЗАПРОС - если запрос:
Приветствие или общение ("Привет", "Как дела?", "Спасибо")
Вопрос НЕ из юридической сферы (программирование, кулинария, медицина и т.д.)
Просьба о помощи в неюридических вопросах

2) legel - ЮРИДИЧЕСКИЙ ЗАПРОС - если запрос касается:
Права и обязанностей граждан/организаций
Правовых последствий действий
Судебных процедур
Административных вопросов
Договорного права
Любых других правовых вопросов

Для НЕРЕЛЕВАНТНЫХ ЗАПРОСОВ:
Верни JSON:

{
  "msg_type": "non_legal",
  "response": "[Вежливый ответ + напоминание о специализации]"
}

Для ЮРИДИЧЕСКИХ ЗАПРОСОВ:
Верни JSON:
{
  "msg_type": "legal",
  "response": "[Переформулированный вопрос пользователя]",
}

Помни, что если запрос юридический, то нужно обязательно вернуть переформулированный вопрос, а не ответ на него!

ПРИМЕРЫ:

"""

    tasks = """

Твои задачи:
1. КЛАССИФИКАЦИЯ ЗАПРОСА
Определи тип запроса пользователя:

А) НЕРЕЛЕВАНТНЫЙ ЗАПРОС - если запрос:
Приветствие или общение ("Привет", "Как дела?", "Спасибо")
Вопрос НЕ из юридической сферы (программирование, кулинария, медицина и т.д.)
Просьба о помощи в неюридических вопросах

B) ЮРИДИЧЕСКИЙ ЗАПРОС - если запрос касается:
Права и обязанностей граждан/организаций
Правовых последствий действий
Судебных процедур
Административных вопросов
Договорного права
Любых других правовых вопросов

2. ОБРАБОТКА ПО ТИПУ ЗАПРОСА
Для НЕРЕЛЕВАНТНЫХ ЗАПРОСОВ:
Верни JSON:

{
  "msg_type": "non_legal",
  "response": "[Вежливый ответ + напоминание о специализации]"
}

Примеры ответов:
Приветствие: "Здравствуйте! Я специализированный юридический помощник. Готов ответить на ваши правовые вопросы."
Неюридический вопрос: "Спасибо за вопрос, но я специализируюсь исключительно на юридических консультациях. Если у вас есть правовые вопросы, буду рад помочь!"

Для ЮРИДИЧЕСКИХ ЗАПРОСОВ:
Проанализируй историю диалога
Если текущий вопрос связан с предыдущими сообщениями, переформулируй его с учетом контекста
Верни JSON:
{
  "msg_type": "legal",
  "response": "[Переформулированный запрос с контекстом]",
}

"""

    rules = """Правила переформулирования:
Добавляй контекст из истории, если новый вопрос неполный:
История диалога:
user: "Я врезался в чужой автомобиль. Что мне за это будет?"
assistant: [Даёт ответ]
user: "А если я был нетрезвым?" 
Ты должен переформулировать вопрос так: "Какие правовые последствия ДТП с ущербом чужому имуществу в состоянии алкогольного опьянения?"

Конкретизируй неясные формулировки:
"Что мне за это будет?" -> "Какие правовые последствия [конкретного действия из контекста]?"

Сохраняй юридическую терминологию и точность
Не добавляй лишнего - только необходимый контекст

Формат ответа:
Всегда возвращай ТОЛЬКО валидный JSON без дополнительных комментариев.

"""

    examples = """Примеры работы:
Пример 1:
История: ["Я врезался в чужую машину. Что мне будет?"]
Новый запрос: "А если я был нетрезвым?"

{
  "msg_type": "legal",
  "response": "Какие правовые последствия ДТП с повреждением чужого автомобиля в состоянии алкогольного опьянения?",
}

Пример 2:
Запрос: "Привет! Как приготовить борщ?"

{
  "msg_type": "non_legal",
  "response": "Здравствуйте! Я специализируюсь на юридических консультациях и не могу помочь с кулинарными вопросами. Если у вас есть правовые вопросы, буду рад помочь!"
}

Пример 3:
История: []
Новый запрос: "Можно ли расторгнуть договор купли-продажи?"

{
  "msg_type": "legal",
  "response": "Основания и порядок расторжения договора купли-продажи",
}
Анализируй внимательно и принимай решения на основе приведенных правил.

Пример 4:
История: []
Новый запрос: "Может ли военнослужащий быть предпринимателем?"

{
  "msg_type": "legal",
  "response": "Может ли военнослужащий быть предпринимателем?",
}

"""

    @staticmethod
    def user_question(question):
        return f"\n\nВОПРОС ПОЛЬЗОВАТЕЛЯ:\n{question}"
    
    @staticmethod
    def prev_messages(prev_messages):
        if prev_messages is None:
            return ''

        messages_str = ''
        for message in prev_messages:
            messages_str += message['role'] + ': ' + message['text'] + '\n'

        res = f"\n\nПРЕДЫДУЩИЕ СООБЩЕНИЯ:\n{messages_str}"
        logger.info("MESSAGES:\n%s", (res, ))
        return res
        

    def prompt(self, question, prev_messages):
        messages = [
            # SystemMessage(content=self.system + self.tasks + self.rules + self.examples + self.prev_messages(prev_messages)),
            SystemMessage(content=self.system2 + self.examples + self.prev_messages(prev_messages)),
            HumanMessage(content=self.user_question(question))
        ]

        while is_prompt_too_long(messages):
            messages[1] = HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 1.5)])
            {"role": "user", "content": messages[1]['content'][:int(len(messages[1]['content']) / 1.5)]},
            HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 1.5)])

        messages = [
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content}
        ]    

        json_schema = RouterAnswer.model_json_schema()
        extra_body = {"guided_json": json_schema}
        return  messages, extra_body
    
class QuestionType(str, Enum):
    non_legal = 'non_legal'
    legal = 'legal'

class RouterAnswer(BaseModel):
    msg_type: QuestionType
    response: str


class EvaluetePromptWithTrueAnswer:
    system = "Ты - эксперт по оценке качества юридических ответов. Твоя задача - сравнить ответ с эталонным правильным ответом и выставить оценку. Отвечай ТОЛЬКО на РУССКОМ языке"

    scale = """ШКАЛА ОЦЕНКИ:
0 - Ответ неверный (противоречит эталону, содержит ошибки, вводит в заблуждение)
2 - Ответ частично верный (частично соответствует эталону, но есть пропуски или неточности)
4 - Ответ полностью верен (полностью соответствует эталону по содержанию и ссылкам)

КРИТЕРИИ ОЦЕНКИ (по приоритету):
1. СООТВЕТСТВИЕ ПРАВОВЫХ НОРМ (КРИТИЧЕСКИЙ)
Статьи и нормы: ответ должен упоминать ТЕ ЖЕ статьи, что и в эталоне
Точность ссылок: номера статей, частей, пунктов должны совпадать
2. ФАКТИЧЕСКАЯ ПРАВИЛЬНОСТЬ (КРИТИЧЕСКИЙ)
Правовые последствия описаны корректно
Процедуры и сроки указаны верно
Нет противоречий с действующим законодательством
3. ПОЛНОТА ОТВЕТА (ВАЖНЫЙ)
Освещены все аспекты из эталонного ответа
Указаны все релевантные правовые нормы
4. СТРУКТУРА И ПОНЯТНОСТЬ (ДОПОЛНИТЕЛЬНЫЙ)
Логичность изложения
Понятность для неюриста
Структурированность информации

"""

    alghoritm = """АЛГОРИТМ ОЦЕНКИ:
ШАГ 1: Анализ правовых норм
Извлеки все статьи/нормы из эталонного ответа
Извлеки все статьи/нормы из оцениваемого ответа
Сравни списки:
- Все ли ключевые статьи упомянуты?
- Нет ли ошибок в номерах статей?
- Нет ли лишних/неуместных ссылок?
ШАГ 2: Проверка фактов
Сравни ключевые утверждения
Проверь правовые последствия
ШАГ 3: Оценка полноты
Все ли аспекты вопроса раскрыты?
Нет ли критических пропусков?
Достаточно ли детализирован ответ?
ШАГ 4: Рассуждения
Опиши все свои рассуждения о том, какую оценку правильно было бы поставить
ШАГ 5: Финальная оценка
Если есть ошибки в статьях или фактах → максимум 0-2 балла
Если статьи верны, но есть пропуски → 2 балла
Если все критерии соблюдены → 4 балла

"""

    examples = """ПРИМЕРЫ ОЦЕНКИ:
Пример 1 (Оценка: 4)
Вопрос: "Какой штраф за превышение скорости на 30 км/ч?"
Эталон: "За превышение на 20-40 км/ч штраф 500 рублей (ст. 12.9 ч.2 КоАП РФ)"
Ответ: "Согласно ст. 12.9 ч.2 КоАП РФ, штраф составляет 500 рублей"
Оценка: 4 - статья верна, сумма верна, ответ полный

Пример 2 (Оценка: 2)
Вопрос: "Можно ли расторгнуть договор купли-продажи?"
Эталон: "Да, по ст. 450 ГК РФ при существенном нарушении, или по ст. 460 при недостатках товара"
Ответ: "Да, можно расторгнуть при существенном нарушении договора"
Оценка: 2 - частично верно, но пропущена ст. 460 и основания по недостаткам

Пример 3 (Оценка: 0)
Вопрос: "Срок исковой давности по трудовым спорам?"
Эталон: "3 месяца с момента нарушения (ст. 392 ТК РФ)"
Ответ: "3 года по общему правилу ст. 196 ГК РФ"
Оценка: 0 - неверная статья и срок, вводит в заблуждение

"""

    answer_format = """Верни JSON:
{
  "reasoning": "[Твои размышления об оценке]",
  "score": "[0/2/4]",
}

"""

    @staticmethod
    def question(question):
        return f"ВОПРОС:\n{question}\n"
    
    @staticmethod
    def true_answer(answer):
        return f"ЭТАЛОННЫЙ ОТВЕТ: \n{answer}\n"
    
    @staticmethod
    def rag_answer(rag_answer):
        return f"ОТВЕТ:\n{rag_answer}\n"
    
    def prompt(self, question, answer, rag_answer):
        messages = [
            SystemMessage(content=self.system + self.scale + self.alghoritm + self.examples + self.answer_format),
            HumanMessage(content=self.question(question) + self.true_answer(answer) + self.rag_answer(rag_answer))
        ]

        while is_prompt_too_long(messages):
            messages[1] = HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 1.5)])
            {"role": "user", "content": messages[1]['content'][:int(len(messages[1]['content']) / 1.5)]},
            HumanMessage(content=messages[1].content[:int(len(messages[1].content) / 1.5)])

        messages = [
            {"role": "system", "content": messages[0].content},
            {"role": "user", "content": messages[1].content}
        ]    

        json_schema = EvalueteAnswer.model_json_schema()
        extra_body = {"guided_json": json_schema}
        return  messages, extra_body


class ScoreType(int, Enum):
    bad = 0
    normal = 2
    good = 4


class EvalueteAnswer(BaseModel):
    reasoning: str
    score: ScoreType