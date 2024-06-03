import os
import json
import re
from models.passage import Passage
from config import(
    NEO4J_URI,
    NEO4J_PASSWORD,
    NEO4J_USERNAME,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    DATABASE,
)
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import  RunnableParallel, RunnablePassthrough
from neo4j import GraphDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from models.passage import Passage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from templates2 import template_grade, template_search, template1,template_entity
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import GraphCypherQAChain


api_key=OPENAI_API_KEY
api_url=OPENAI_API_BASE

graph=Neo4jGraph()

engine = create_engine(DATABASE, echo=True)
Session = sessionmaker(bind=engine)

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

def delete_all_nodes():
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        
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
            
# content_ids = input("Input the IDs of the content (separate IDs by space): ").strip().split()
# contents = []
# with scoped_session() as conn:
#     for content_id in content_ids:
#         result = conn.query(Passage.content).offset(int(content_id) - 1).limit(1).all()
#         content = result[0][0] if result else None
#         if content:
#             contents.append(content)
        
# llm_transformer_filtered = LLMGraphTransformer(
#     llm=llm,
# )
# delete_all_nodes()
# text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# documents_text_chunks = []
# for content in contents:
#     documents_text_chunks.extend(text_splitter.split_text(content))

# documents = [Document(page_content=chunk) for chunk in documents_text_chunks]


# graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
# graph.add_graph_documents(graph_documents_filtered, baseEntityLabel=True, include_source=True)


CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Examples: Here are a few examples of generated Cypher statements for particular questions:

# I want to find answers to my question from documents
MATCH (d:Document)
WHERE d.text CONTAINS '{question}'
RETURN d.text AS answer"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

serarch_result=chain.invoke({"query":"如何使用市场中的知识库?"})
print(serarch_result)