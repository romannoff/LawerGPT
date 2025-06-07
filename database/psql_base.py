from sqlalchemy import (
    create_engine, MetaData,
    Table, Column, Integer, String, Text, ARRAY, text
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from sqlalchemy.orm import sessionmaker

class PostgresBase:
    def __init__(self):
        database_url = "postgresql+psycopg2://legal_user:secret_password@localhost:5432/legal_db"
        self.engine = create_engine(database_url, echo=False)
        self.metadata = MetaData()

        schemas = ['codes', 'federal_laws']
        with self.engine.begin() as conn:
            for schema in schemas:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {schema};'))

        self.articles_codes = Table(
            'articles', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('codes_id', Integer, nullable=False),
            Column('chapter_num', String, nullable=False),
            Column('chapter_title', String, nullable=True),
            Column('article_num', String, nullable=False),
            Column('title', String, nullable=True),
            Column('text', Text, nullable=False),
            Column('comments', Text, nullable=True),
            Column('additional', Text, nullable=True),
            Column('court_links', PG_ARRAY(String), nullable=True),
            schema='codes'
        )

        self.articles_laws = Table(
            'articles', self.metadata,
            Column('id', Integer, primary_key=True),
            Column('law_id', Integer, nullable=False),
            Column('chapter_num', String, nullable=False),
            Column('chapter_title', String, nullable=True),
            Column('article_num', String, nullable=False),
            Column('title', String, nullable=True),
            Column('text', Text, nullable=False),
            Column('comments', Text, nullable=True),
            Column('additional', Text, nullable=True),
            Column('court_links', PG_ARRAY(String), nullable=True),
            schema='federal_laws'
        )

        self.metadata.create_all(self.engine)

    def add_code_atricle(self, article_code, article_id):
        # Session = sessionmaker(bind=self.engine)
        # session = Session()

        article = {
            'id': article_id,
            'codes_id': article_code['codes_id'],
            'chapter_num': article_code['chapter_num'],
            'chapter_title': article_code['chapter_title'],
            'article_num': article_code['article_num'],
            'title': article_code['title'],
            'text': article_code['text'],
            'comments': article_code['comments'],
            'additional': article_code['additional'] + '\n' + article_code['additional_2'],
            'court_links': article_code['court_links'],
            'article_type': 'codes'
        }

        with self.engine.begin() as conn:
            conn.execute(self.articles_codes.insert(), [article])
        

    def add_law_atricle(self, article_law, article_id):
        # Session = sessionmaker(bind=self.engine)
        # session = Session()

        article = {
            'id': article_id,
            'law_id': article_law['law_id'],
            'chapter_num': article_law['chapter_num'] if article_law.get('chapter_num') is not None else '',
            'chapter_title': article_law['chapter_title'] if article_law.get('chapter_title') is not None else '',
            'article_num': article_law['article_num'],
            'title': article_law['title'],
            'text': article_law['text'],
            'comments': article_law['comments'],
            'additional': article_law['additional'] + '\n' + article_law['additional_2'],
            'court_links': article_law['court_links'],
            'article_type': 'laws'
                }

        with self.engine.begin() as conn:
            conn.execute(self.articles_laws.insert(), [article])

    def get_code_article(self, article_id):
        query = self.articles_codes.select().where(self.articles_codes.c.id == article_id)
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().first()
            if row:
                return dict(row)
        return None
        
    def get_law_article(self, article_id):
        query = self.articles_laws.select().where(self.articles_laws.c.id == article_id)
        with self.engine.connect() as conn:
            row = conn.execute(query).mappings().first()
            if row:
                return dict(row)
        return None
    
    def select_code_articles_id_text(self):
        query = text("SELECT id, text FROM codes.articles")
        with self.engine.connect() as conn:
            result = conn.execute(query).mappings().all()
            return [dict(row) for row in result]

    def select_law_articles_id_text(self):
        query = text("SELECT id, text FROM federal_laws.articles")
        with self.engine.connect() as conn:
            result = conn.execute(query).mappings().all()
            return [dict(row) for row in result]
    
    def drop_all(self):
        self.metadata.drop_all(self.engine)

# if __name__ == '__main__':
    # base = PostgresBase()
    # base.drop_all()
    # print('DROP TABLE')