
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import re
from langchain.chains import LLMChain
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import func
from jieba.analyse import extract_tags
from typing import List
import os
from templates import template_test

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

llm=ChatOpenAI(model="gpt-3.5-turbo-16k",temperature=0)

prompt = PromptTemplate(input_variables=["passage"],template=template_test)

chain_test=LLMChain(llm=llm,prompt=prompt)

otput_parser=StrOutputParser()
output=chain_test.invoke("* 使用sync.Pool存放临时变量，做到协程间共享已分配内存 具体参看* []byte复用的一个例子")
output_chain=otput_parser.parse(output)
print(output_chain)
