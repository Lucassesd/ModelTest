import re
import os
from neo4j import GraphDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from models.passage import Passage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from templates2 import prompt
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship
)
from typing import List, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "Sztu2024!"

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

# 连接 Neo4j
neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)

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

class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )

prompt_graph = ChatPromptTemplate.from_messages(prompt)

@contextmanager
def scoped_session():
    db_session = Session()
    try:
        yield db_session
    finally:
        db_session.close()

def nodes_to_string(nodes: List[Node]) -> str:
    return ', '.join([f"Node(id='{node.id}', type='{node.type}', properties={node.properties})" for node in nodes])

def rels_to_string(rels: List[Relationship]) -> str:
    return ', '.join([
        f"Relationship(source=Node(id='{rel.source.id}', type='{rel.source.type}'), "
        f"target=Node(id='{rel.target.id}', type='{rel.target.type}'), type='{rel.type}', properties={rel.properties})"
        for rel in rels
    ])

from typing import List

def insert_graph_to_neo4j(nodes: List[Node], rels: List[Relationship]):
    with neo4j_driver.session() as session:
        # 插入节点并检查重复
        for node in nodes:
            node_props = {prop.key: prop.value for prop in (node.properties or [])}
            prop_str = ", ".join([f"n.`{k}` = ${k}" for k in node_props.keys()])
            node_type = node.type.replace(" ", "_")  # 用下划线替换空格
            if prop_str:
                query = (
                    f"MERGE (n:{node_type} {{id: $id}}) "
                    f"ON CREATE SET {prop_str} "
                    f"ON MATCH SET {prop_str}"
                )
            else:
                query = (
                    f"MERGE (n:{node_type} {{id: $id}})"
                )
            params = {"id": node.id, **node_props}
            session.run(query, params)

        # 插入关系并检查重复
        for rel in rels:
            source_id = rel.source.id
            target_id = rel.target.id
            rel_props = {prop.key: prop.value for prop in (rel.properties or [])}
            rel_prop_str = ", ".join([f"r.`{k}` = ${k}" for k in rel_props.keys()])
            rel_type = rel.type.replace(" ", "_")  # 用下划线替换空格
            if rel_prop_str:
                query = (
                    "MATCH (a {id: $source_id}), (b {id: $target_id}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    f"ON CREATE SET {rel_prop_str} "
                    f"ON MATCH SET {rel_prop_str}"
                )
            else:
                query = (
                    "MATCH (a {id: $source_id}), (b {id: $target_id}) "
                    f"MERGE (a)-[r:{rel_type}]->(b)"
                )
            params = {"source_id": source_id, "target_id": target_id, **rel_props}
            session.run(query, params)





def split_text_into_chunks(text: str, chunk_size: int) -> List[str]:
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def delete_all_nodes():
    with neo4j_driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def extract_and_store_graph(content,code) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = create_structured_output_runnable(KnowledgeGraph, llm, prompt_graph)
    data = extract_chain.invoke({"input": content, "code": code})

    # Assuming data is an instance of KnowledgeGraph
    if isinstance(data, KnowledgeGraph):
        nodes_str = nodes_to_string(data.nodes)
        rels_str = rels_to_string(data.rels)
        
        print(nodes_str, "\n\n\n", rels_str)
        insert_graph_to_neo4j(data.nodes, data.rels)
    else:
        print("Data extraction failed")
        
def extract_code_blocks(markdown_text):
    code_blocks = re.findall(r'```(.*?)```', markdown_text, re.DOTALL)
    return code_blocks

def format_code_blocks(code_blocks):
    formatted_blocks = []
    for block in code_blocks:
        # Split the block into lines and strip leading/trailing whitespace
        lines = block.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Add back the code block markers and join the lines with newlines
        formatted_block = '```\n' + '\n'.join(lines) + '\n```'
        formatted_blocks.append(formatted_block)
    
    return '\n\n'.join(formatted_blocks)

# 删除所有节点和关系
delete_all_nodes()

# 从数据库中提取内容
content_ids = input("Input the IDs of the content (separate IDs by space): ").strip().split()
contents = []
with scoped_session() as conn:
    for content_id in content_ids:
        result = conn.query(Passage.content).offset(int(content_id) - 1).limit(1).all()
        content = result[0][0] if result else None
        if content:
            contents.append(content)

contents_str="\n".join(contents)
code=extract_code_blocks(contents_str)
formatted_code = format_code_blocks(code)
extract_and_store_graph(contents_str,code)
# # 关闭 Neo4j 驱动
neo4j_driver.close()