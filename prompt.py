import os
from typing import Optional, List
from openai import OpenAI
from templates import template_process,Examples_of_entities
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
    Entity: Optional[str] = Field(default=None, description="The entity in the process")
    Attributes: Optional[str] = Field(default=None, description="The attributes of the entity")
class Data(BaseModel):
    process: List[Process]
    
prompt = ChatPromptTemplate.from_messages(template_process)

def format_processes(processes: List[Process]) -> str:
    output = ""
    for index, process in enumerate(processes, start=1):
        output += f"Entity:{index}: {process.Entity}\n"
        output += f"Entity_attributes:{process.Attributes}\n\n"
    return output
 

contents_id=int(input("input the id of the content: "))
chain = create_structured_output_runnable(Data, llm, prompt)
with scoped_session() as conn:
    contents=conn.query(AI_answer.content).offset(contents_id-1).limit(1).all()
    content = contents[0][0] if contents else None

if content:
    output_parser = StrOutputParser()
    runnable = prompt | llm.with_structured_output(schema=Data)
    response = runnable.invoke({"input":content,"Examples_of_entities":Examples_of_entities})
    output= format_processes(response.process)
    print(output)
else:
    print("No content found for the specified id.")