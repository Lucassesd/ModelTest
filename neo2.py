import os
import json
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
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.runnables import  RunnableParallel, RunnablePassthrough
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
from templates2 import template_grade, template_search, template1,template_entity
from langchain_openai import OpenAIEmbeddings
from typing import List
from langchain.pydantic_v1 import Field, BaseModel
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter


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
            
content_ids = input("Input the IDs of the content (separate IDs by space): ").strip().split()
contents = []
with scoped_session() as conn:
    for content_id in content_ids:
        result = conn.query(Passage.content).offset(int(content_id) - 1).limit(1).all()
        content = result[0][0] if result else None
        if content:
            contents.append(content)
        
llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
)

delete_all_nodes()
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

documents_text_chunks = []
for content in contents:
    documents_text_chunks.extend(text_splitter.split_text(content))

documents = [Document(page_content=chunk) for chunk in documents_text_chunks]


graph_documents_filtered = llm_transformer_filtered.convert_to_graph_documents(documents)
graph.add_graph_documents(graph_documents_filtered, baseEntityLabel=True, include_source=True)


vector_index = Neo4jVector.from_existing_graph( 
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the verb, noun or business entities that appear in the text",
    )

prompt_entity = ChatPromptTemplate.from_messages(template_entity)

entity_chain = prompt_entity | llm.with_structured_output(Entities)

def remove_lucene_chars(text):
    lucene_special_chars = r'[+\-&|!(){}\[\]^"~*?:\\/]'
    cleaned_text = re.sub(lucene_special_chars, '', text)
    return cleaned_text

graph.query(
    "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text
    search. It processes the input string by splitting it into words and 
    appending a similarity threshold (~2 changed characters) to each
    word, then combines them using the AND operator. Useful for mapping
    entities from user questions to database values, and allows for some 
    misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, 
            {limit:2})  
            YIELD node,score
            CALL {
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS 
              output
              UNION
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS 
              output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(template1)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | ChatOpenAI(temperature=0)
        | StrOutputParser(),
    ),
    RunnableLambda(lambda x : x["question"]),
)
prompt_search = ChatPromptTemplate.from_template(template_search)

chain_search = (
    RunnableParallel(
        {
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | prompt_search
    | llm
    | StrOutputParser()
)
prompt_grade = ChatPromptTemplate.from_template(template_grade)
chain_grade = (
    RunnableParallel(
        {
            "expected_result": RunnablePassthrough(),
            "coll_result": RunnablePassthrough(),
        }
    )
    | prompt_grade
    | llm
    | StrOutputParser()
)


file_path="fibona-drop rag测试集.json"
# try:
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)

#     for item in data:
#         question = item["question"]
#         expected_result = item["expected_result"]
        
#         coll_result = chain_search.invoke({"question": question})
        
#         grade_result = chain_grade.invoke({"expected_result": expected_result, "coll_result": coll_result})
        
#         print(f"Question: {question}")
#         print(f"Collected Result: {coll_result}")
#         print(f"Grade Result: {grade_result}")
#         print("\n" + "-"*50 + "\n")

# except UnicodeDecodeError as e:
#     print(f"Unicode decoding error: {e}")
# except json.JSONDecodeError as e:
#     print(f"JSON decoding error: {e}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
coll_result = chain_search.invoke({"question":"如何使用市场中的知识库?"})
grade_result = chain_grade.invoke({"expected_result":"在知识库聊天界面，点击右上角 **“市场”** 按钮即可进入知识库市场弹窗，在这里你可以对知识库进行点收藏，收藏后使用知识库进行聊天，并且在提示词时可供绑定。", "coll_result": coll_result})
print(coll_result,"\n\n",grade_result)
