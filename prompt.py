import os
import numpy as np
from typing import Optional, List
from openai import OpenAI
from templates import template_process,Examples_of_entities,template_relation
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

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "root"

graph = Neo4jGraph()

# AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
# AI_URL = "https://api.chatanywhere.tech/v1"

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

engine = create_engine("mysql+pymysql://root:root@localhost:3306/crawler", echo=True)
Session = sessionmaker(bind=engine)

model_name = input("input the model name: ")
if model_name == "mistral":
    llm = ChatMistralAI(model="mistral-chat", temperature=0)
elif model_name == "GPT3":
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
elif model_name == "GPT4":
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
llm_transformer = LLMGraphTransformer(llm=llm)
@contextmanager
def scoped_session():
        db_session = Session()
        try:
            yield db_session
        finally:
            db_session.close()

class Process(BaseModel):
    Entity: Optional[str] = Field(default=None, description="The entity in the process")
    Attributes: Optional[str] = Field(default=None, description="The attributes of the entity")
class Data(BaseModel):
    process: List[Process]

class Relation(BaseModel):
    relationship:Optional[str] = Field(default=None,description="The relationship between the entities")
prompt_entity = ChatPromptTemplate.from_messages(template_process)
prompt_relation=ChatPromptTemplate.from_messages(template_relation)

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
    entity = runnable.invoke({"input": content, "Examples_of_entities": Examples_of_entities})
    entities = format_processes(entity.process)
    return entities

def cluster_entities(contents):
    all_entities = []
    for content in contents:
        entities = extract_entities_from_text(content)
        all_entities.extend(entities)

    embeddings_model = OpenAIEmbeddings()
    embedded_entities = []
    entity_texts = []

    for entity, attributes in all_entities:
        combined_text = f"{entity} {attributes}"
        embedded_vector = embeddings_model.embed_query(combined_text)
        embedded_entities.append(embedded_vector)
        entity_texts.append(combined_text)

    embedded_entities = np.array(embedded_entities)

    # 使用K-Means聚类
    kmeans = KMeans(n_clusters=3, random_state=0).fit(embedded_entities)  

    # 获取每个嵌入向量的聚类标签
    labels = kmeans.labels_

    clusters = {}
    for label, entity in zip(labels, entity_texts):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(entity)

    for cluster, entities in clusters.items():
        print(f"Cluster {cluster}: {entities}\n")
    relation=create_structured_output_runnable(Relation, llm, prompt_relation)
    relationship=relation.invoke({"clusters":clusters})
    print(relationship)

content_ids = input("Input the IDs of the content (separate IDs by space): ").split()
contents = []
with scoped_session() as conn:
    for content_id in content_ids:
        result = conn.query(Passage.content).offset(int(content_id) - 1).limit(1).all()
        content = result[0][0] if result else None
        if content:
            contents.append(content)

cluster_entities(contents)
