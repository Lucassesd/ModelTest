import json
import os
from neo4j import GraphDatabase
from langchain_core.runnables import (
    RunnablePassthrough,
)
from templates2 import  template_search, template_entity,template_grade
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import re
from loguru import logger 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from typing import List
from langchain.pydantic_v1 import Field, BaseModel
from langchain_community.graphs import Neo4jGraph
from langchain_core.retrievers import BaseRetriever
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "skf707=="

graph = Neo4jGraph()

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

engine = create_engine("mysql+pymysql://root:Sztu2024!@nj-cdb-ejzzmfxj.sql.tencentcdb.com:63911/crawler", echo=True)
Session = sessionmaker(bind=engine)

neo4j_driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
)
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    
# logger.add("/Users/skf/ModelTest/work/file_2.log")  
@contextmanager
def scoped_session():
        db_session = Session()
        try:
            yield db_session
        finally:
            db_session.close() 
class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description="All the verb, noun or business entities that appear in the text",
    )


class Retriever(BaseRetriever):
        def structured_retriever(self, question: str) -> str:
            prompt_entity = ChatPromptTemplate.from_messages(template_entity)
            entity_chain = prompt_entity | llm.with_structured_output(Entities)
            result_list = []   
            seen_neighbors1 = set()
            entities = entity_chain.invoke({"question": question})
            logger.debug(entities)
            for entity in entities.names: 
                response1 = graph.query(   
                    """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
                    YIELD node
                    RETURN node""",
                    {"query": entity},  
                )
            for row in response1:
                    node_info = {
                        "节点": row['node'],
                        "关系": [],
                        "邻节点": []
                    }
                    neighbor_response = graph.query(
                        """MATCH (node)-[r]-(neighbor:__Entity__) 
                        WHERE node.id = $node_id
                        RETURN neighbor, r""", 
                        {"node_id": row['node']['id']},
                    )
                    for neighbor_row in neighbor_response:
                        neighbor_id = neighbor_row['neighbor']['id']
                        if neighbor_id not in seen_neighbors1:  
                            node_info["关系"].append(neighbor_row['r'])
                            node_info["邻节点"].append(neighbor_row['neighbor'])
                            seen_neighbors1.add(neighbor_id) 
                    result_list.append(node_info)
                    logger.info(result_list)

            return json.dump(result_list, ensure_ascii=False, indent=2)    
    
    # def structured_retriever(self, question: str) -> str:
    #     prompt_entity = ChatPromptTemplate.from_messages(template_entity)
    #     entity_chain = prompt_entity | llm.with_structured_output(Entities)
    #     result1 = ""
    #     result_list = []  
    #     seen_neighbors = set()  
    #     seen_neighbors1 = set()
    #     entities = entity_chain.invoke({"question": question})
    #     logger.debug(entities)
    #     for entity in entities.names:
    #         response = graph.query(
                
    #             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
    #             YIELD node
    #             RETURN node""",
    #             {"query": entity},  
    #         )
    #     logger.warning(response)
    #     for row in response:
    #             neighbor_response = graph.query(
    #                 """MATCH (node)-[r]-(neighbor:Document) 
    #                 WHERE node.id = $node_id
    #                 RETURN neighbor, r""", 
    #                 {"node_id": row['node']['id']},
    #             )
    #             for neighbor_row in neighbor_response:
    #                 neighbor_id = neighbor_row['neighbor']['id']
    #                 if neighbor_id not in seen_neighbors:  
    #                     result1 += f"coll_back_content:{neighbor_row['neighbor']}"
    #                     seen_neighbors.add(neighbor_id)  
    #             logger.warning(result1) 
    #     for entity in entities.names: 
    #         response1 = graph.query(
                
    #             """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
    #             YIELD node
    #             RETURN node""",
    #             {"query": entity},  
    #         )
    #     for row in response1:
    #             node_info = {
    #                 "节点": row['node'],
    #                 "关系": [],
    #                 "邻节点": []
    #             }
    #             neighbor_response = graph.query(
    #                 """MATCH (node)-[r]-(neighbor:__Entity__) 
    #                 WHERE node.id = $node_id
    #                 RETURN neighbor, r""", 
    #                 {"node_id": row['node']['id']},
    #             )
    #             for neighbor_row in neighbor_response:
    #                 neighbor_id = neighbor_row['neighbor']['id']
    #                 if neighbor_id not in seen_neighbors1:  
    #                     node_info["关系"].append(neighbor_row['r'])
    #                     node_info["邻节点"].append(neighbor_row['neighbor'])
    #                     seen_neighbors1.add(neighbor_id) 
    #             result_list.append(node_info)
    #             logger.info(result_list)
    #     combined_result = {
    #         "result1": result1,
    #         "result_list": result_list
    #     }
    #     return combined_result

        def text_retriever(self, question: str) -> str:
            prompt_entity = ChatPromptTemplate.from_messages(template_entity)
            entity_chain = prompt_entity | llm.with_structured_output(Entities)
            result = ""
            seen_neighbors = set()  
            entities = entity_chain.invoke({"question": question})
            logger.error(entities)
            # for entity in entities.names:
            response = graph.query(
                    """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                    YIELD node
                    RETURN node""",
                    {"query": question}, 
                )
            for row in response:
                    neighbor_response = graph.query(
                        """MATCH (node)-[r]-(neighbor:Document) 
                        WHERE node.id = $node_id
                        RETURN neighbor, r""", 
                        {"node_id": row['node']['id']},
                    )
                    for neighbor_row in neighbor_response:
                        neighbor_id = neighbor_row['neighbor']['id']
                        if neighbor_id not in seen_neighbors:  
                            result += f"coll_back_content:{neighbor_row['neighbor']}"
                            seen_neighbors.add(neighbor_id)  
            logger.warning(result)  
            return result


        def _get_relevant_documents(self, question: str) -> dict:
            structured_data = self.structured_retriever(question)
            return {
                "structured_data": structured_data,
            }
        
        def prompt_grade(self, expected_result, question):
            prompt_grade = ChatPromptTemplate.from_template(template_grade)
            chain_grade = prompt_grade| llm | StrOutputParser()
            i=self.text_retriever(question)
            n=self.structured_retriever(question)
            result=chain_grade.invoke({"expected_result": expected_result, "coll_result": self.text_retriever(question)})
            return result
        
        def test(self):
            all_results = []
            file_path="fibona-drop rag测试集.json"
            # try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            for item in data:
                question = item["question"]
                logger.info("question{}",question)
                expected_result = item["expected_result"]
                logger.info("expected_result{}",expected_result)
                retrieved_result=self.structured_retriever(question)
                combined_item = {
                    "question": question,
                    "expected_result": expected_result,
                    "retrieved_result": retrieved_result
                }
                all_results.append(combined_item)
            with open("/Users/skf/ModelTest/work/all_results.json", 'w', encoding='utf-8') as json_file:
                json.dump(all_results, json_file, ensure_ascii=False, indent=2)   

                
        
instance = Retriever()
# # coll_result = chain_search.invoke({"question":"如何快速搭建知识库并绑定提示词"})
# graph.query(
#     "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
# instance.structured_retriever({"question":"如何快速新建知识库并绑定提示词"})
instance.test()
# # grade_result = chain_grade.invoke({"expected_result":"在知识库聊天界面，点击右上角 **“市场”** 按钮即可进入知识库市场弹窗，在这里你可以对知识库进行点收藏，收藏后使用知识库进行聊天，并且在提示词时可供绑定。", "coll_result": coll_result})
# # print(coll_result,"\n\n",grade_result)