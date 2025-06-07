from abc import ABC, abstractmethod
from typing import TypedDict

class AgentAnswer(TypedDict):
    query: str
    context: list[str]
    answer: str

class Agent(ABC):
    
    @abstractmethod
    def __call__(self, *args, **kwds) -> AgentAnswer: pass

