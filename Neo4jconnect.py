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
from templates import relationships
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Sztu2024!"

graph = Neo4jGraph()

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

engine = create_engine("mysql+pymysql://root:root@localhost:3306/crawler", echo=True)
Session = sessionmaker(bind=engine)

model_name = input("input the model name: ")
if model_name == "GPT3":
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


content_id = input("Input the IDs of the content (separate IDs by space): ")
with scoped_session() as conn:
    content=conn.query(Passage.content).offset(int(content_id)-1).limit(1).all()
            
llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
    allowed_nodes = ["Characteristics of the Question", "Question", "Analysis of Question", "Solution"],
    allowed_relationships=relationships,
    node_properties=True,
)
documents=[Document(page_content=content[0][0])]
graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents_filtered)
print(f"Nodes:{graph_documents_filtered[0].nodes}")
print(f"Relationships:{graph_documents_filtered[0].relationships}")