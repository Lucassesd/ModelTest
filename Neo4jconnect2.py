import os
from typing import List, Tuple
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from models.passage import Passage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
import openai
import os
import numpy as np
from typing import Optional, List
from openai import OpenAI
from templates3 import template_process,template_relation
from models.passage import Passage
from sklearn.cluster import KMeans
from models.ai_answer import AI_answer
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

# 配置环境变量
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "skf707=="
AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

# 连接 Neo4j
graph = Neo4jGraph()

# 配置数据库连接

engine = create_engine("mysql+pymysql://root:Sztu2024!@nj-cdb-ejzzmfxj.sql.tencentcdb.com:63911/crawler", echo=True)
Session = sessionmaker(bind=engine)

# 配置 LLM
model_name = input("Input the model name (GPT3 or GPT4): ").strip()
if model_name == "GPT3":
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
elif model_name == "GPT4":
    llm = ChatOpenAI(model="gpt-4", temperature=0)
else:
    raise ValueError("Unsupported model name. Choose 'GPT3' or 'GPT4'.")
class Process(BaseModel):
    Entity: Optional[str] = Field(default=None, description="The entity in the process")
    Attributes: Optional[str] = Field(default=None, description="The attributes of the entity")
class Data(BaseModel):
    process: List[Process]

class Relation(BaseModel):
    relationship:Optional[str] = Field(default=None,description="The relationship between the entities")
prompt_entity = ChatPromptTemplate.from_messages(template_process)
prompt_relation=ChatPromptTemplate.from_messages(template_relation)

@contextmanager
def scoped_session():
    db_session = Session()
    try:
        yield db_session
    finally:
        db_session.close()

def format_processes(processes: List[Process]) -> List[str]:
    entities = []
    for index, process in enumerate(processes, start=1):
        entity = f"Entity:{index}: {process.Entity}"
        attributes = f"Entity_attributes:{process.Attributes}"
        entities.append((entity, attributes))
    return entities

def extract_entities_from_text(content):
    output_parser = StrOutputParser()
    runnable = create_structured_output_runnable(Data, llm, prompt_entity)
    entity = runnable.invoke({"input": content})
    entities = format_processes(entity.process)
    print(entities)
    return entities

# 从数据库中提取内容
content_ids = input("Input the IDs of the content (separate IDs by space): ").strip().split()
contents = []
with scoped_session() as conn:
    for content_id in content_ids:
        content = conn.query(Passage.content).offset(int(content_id)-1).limit(1).all()
        if content:
            contents.append(content[0][0])

def create_graph_entities(entities):  
    relation=create_structured_output_runnable(Relation, llm, prompt_relation)
    relationship=relation.invoke({"put1":entities})
    print(relationship)
    return relationship

# 提取实体并创建图结构
for content in contents:
    entities = extract_entities_from_text(content)
    create_graph_entities(entities)
