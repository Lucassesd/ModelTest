import os
from typing import Optional, List
from openai import OpenAI
from templates import template_process,permissible_nodes_to_extract
from models.passage import Passage
from models.ai_answer import AI_answer
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_core.runnables import RunnablePassthrough


# AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
# AI_URL = "https://api.chatanywhere.tech/v1"

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

engine = create_engine("mysql+pymysql://root:Sztu2024!@nj-cdb-ejzzmfxj.sql.tencentcdb.com:63911/crawler", echo=True)
Session = sessionmaker(bind=engine)

model_name = input("input the model name: ")
if model_name == "mistral":
    llm = ChatMistralAI(model="mistral-chat", temperature=0)
elif model_name == "GPT3":
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
elif model_name == "GPT4":
    llm = ChatOpenAI(model="gpt-4", temperature=0)
elif model_name == "deepseek":
    llm = ChatOpenAI(model="deepseek-chat", temperature=0)
@contextmanager
def scoped_session():
        db_session = Session()
        try:
            yield db_session
        finally:
            db_session.close()

class Process(BaseModel):
    question: Optional[str] = Field(default=None, description="information about the question.")
    question_detail: Optional[str] = Field(default=None, description="detailed information about the question.")
    keywords: Optional[str] = Field(default=None, description="keywords related to the information.")
    solution: Optional[str] = Field(default=None, description="solution to the problem.")
    relationship: Optional[str] = Field(default=None, description="relationship between the entities.")
    
class Data(BaseModel):
    process: List[Process]
    
prompt = ChatPromptTemplate.from_messages(template_process)

def format_processes(processes: List[Process]) -> str:
    output = ""
    for index, process in enumerate(processes, start=1):
        output += f"question:{index}: {process.question}\n"
        output += f"question_detail:{process.question_detail}\n"
        output += f"keywords:{process.keywords}\n"
        output += f"solution:{process.solution}\n"
        output += f"relationship:{process.relationship}\n\n"
    return output
 

contents_id=int(input("input the id of the content: "))
chain = create_structured_output_runnable(Data, llm, prompt)
with scoped_session() as conn:
    contents=conn.query(AI_answer.content).offset(contents_id-1).limit(1).all()
    content = contents[0][0] if contents else None

if content:
    output_parser = StrOutputParser()
    runnable = prompt | llm.with_structured_output(schema=Data)
    response = runnable.invoke({"input":content,"permissible_nodes":permissible_nodes_to_extract})
    print(response)
else:
    print("No content found for the specified id.")