from sqlalchemy import (
    create_engine, MetaData,
    Table, Column, Integer, String, text
)
from sqlalchemy import delete 

from src.logging_conf import logger


class HistoryBase:
    def __init__(self):
        database_url = "postgresql+psycopg2://legal_user:secret_password@localhost:5432/legal_db"
        self.engine = create_engine(database_url, echo=False)
        self.metadata = MetaData()

        schema = 'rag_history'
        with self.engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {schema};'))

        self.history = Table(
            'history', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('chat_id', String),
            Column('role', String),
            Column('text', String),
            schema=schema
        )

        self.metadata.create_all(self.engine)

    def add_message(self, chat_id: str, role: str, text: str):

        message = {
            'chat_id': chat_id,
            'role': role,
            'text': text
        }

        with self.engine.begin() as conn:
            conn.execute(self.history.insert(), [message])
        if role == 'user':
            logger.info("ADD MESSAGES: chat_id: %s role: %s, text: %s" % (chat_id, role, text))
        else:
            logger.info("ADD MESSAGES: chat_id: %s role: %s" % (chat_id, role))
        
    def get_messages(self, chat_id: str, limit=None):
        logger.info("GET MESSAGES: chat_id: %s limit: %s" % (chat_id, limit))
        query = self.history.select().where(self.history.c.chat_id == str(chat_id))
        if limit is not None:
            query = query.limit(limit)
        with self.engine.connect() as conn:
            rows = conn.execute(query).mappings().all()
            return [dict(row) for row in rows] if rows else None
    
    def delete_messages_by_chat_id(self, chat_id: str):
        logger.info("DELETE MESSAGES: chat_id: %s" % (chat_id))
        stmt = delete(self.history).where(self.history.c.chat_id == chat_id)
        with self.engine.begin() as conn:
            conn.execute(stmt)
    
    def drop_all(self):
        self.metadata.drop_all(self.engine)

if __name__ == '__main__':
    base = HistoryBase()
    base.drop_all()
    print('DROP TABLE')