from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from models.passage import Passage
from models.ai_answer import AI_answer
from templates import (
    template_output,
    example_question,
    example_feature,
)
from langchain_core.prompts import PromptTemplate
import re
from langchain.chains import LLMChain
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from jieba.analyse import extract_tags
from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser

AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL

engine = create_engine("mysql+pymysql://root:root@localhost:3306/crawler", echo=True)
Session = sessionmaker(bind=engine)

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)

@contextmanager
def scoped_session():
        db_session = Session()
        try:
            yield db_session
        finally:
            db_session.close()

class Chainlang():
    def __init__(self,document_index=1):
        self.document_index=document_index

    def _get_document_content(self,document_index):
        with scoped_session() as conn:
            content = (
                conn.query(Passage.content).offset(document_index - 1).limit(1).all()
            )
        return content or ""

    def _get_doucement_catalog(self, document_content):
        pattern = r"^(#{1,6})\s*(.*)$"
        matches = re.findall(pattern, document_content, re.MULTILINE)
        headings = []
        for match in matches:
            level = len(match[0])
            title = match[1].strip()
            indent = " " * (level - 1) * 4
            headings.append(f"{indent}{title}")
        return "\n".join(headings)

    def _summarize_document(self, document_content, document_catalog):
        output = StrOutputParser
        prompt_feature = PromptTemplate(
            input_variables=["example_feature", "example_question", "passage", "catalog"], template=template_output
        )
        chain_feature = LLMChain(output_key="output", prompt=prompt_feature, llm=llm)

        output = chain_feature.invoke(
            {"example_feature": example_feature, "example_question": example_question, "passage": document_content, "catalog": document_catalog}
        )
        return output
    
    def _encode_keywords_to_tfidf(self, summary):
        stop_words = [
        "cs", "line","maybe","make","check","int""while"
        "please", "help", "need", "problem","if","else",
        "thank", "thanks", "issue", "question",
        "solution", "could", "would", "might",
        "may", "using", "work","具体内容","解决方案","问题"
        "getting", "trying", "currently","your","export"
    ]   
        tokens: List[str] = extract_tags(
            summary,
            topK=10,
            withWeight=False,
            allowPOS=("n",
                "a",    
                "ns",
                "nr",
                "nt",
                "nz",
                "i",
                "l",
                "j",
                "t",
                "f",
                "eng",),
            withFlag=False,
        )
        tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in stop_words]
        return tokens
    
    def _calculate_jaccard_similarity(self,set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union if union != 0 else 0
        return similarity   

    def _calculate_similarity_for_all_articles(self):
        with scoped_session() as conn:
            all_keywords = conn.query(AI_answer.keywords).all()
            all_keyword_sets = [set(keyword[0]) for keyword in all_keywords]

            similarities = []
            for i in range(len(all_keyword_sets)):
                for j in range(i + 1, len(all_keyword_sets)):
                    similarity = self._calculate_jaccard_similarity(all_keyword_sets[i], all_keyword_sets[j])
                    similarity_percentage = similarity * 100
                    similarities.append((i + 1, j + 1, similarity_percentage))
        return similarities

    def start(self):
        if self.document_index and self.document_index.strip():
            if int(self.document_index):
                document_contents = self._get_document_content(int(self.document_index))
                if document_contents:
                    parser=SimpleNodeParser()
                    documents = SimpleDirectoryReader("D:\pytest\dro").load_data()
                    document_content = document_contents[0][0]
                    document_catalog = self._get_doucement_catalog(document_content)
                    summary = self._summarize_document(document_content, document_catalog).get("output")
                    keywords = self._encode_keywords_to_tfidf(summary)
                    with scoped_session() as conn:
                        existing_answer = conn.query(AI_answer).filter(AI_answer.content == summary).first()
                        if existing_answer:
                            print(f"Answer already exists for the given document: {existing_answer.id}")
                        else:
                            page_id=conn.query(Passage.page_id).filter(Passage.content==document_content).scalar()
                            process=AI_answer(type="juejin",page_id=page_id,keywords=keywords,content=summary)
                            conn.add(process)
                            conn.commit()
                            print(f"Answer generated for the given document: {process.id}")
                else:
                    print("No document found for the given index.")
            return
        
        similarities = self._calculate_similarity_for_all_articles()
        total_similarity = sum(similarity for _, _, similarity in similarities)
        for i, j, similarity in similarities:
            print(f"Similarity between article {i} and article {j}: {similarity:.2f}%")
        print(f"Total similarity between all articles: {total_similarity:.2f}%")
            
if __name__ == "__main__":
    chain = Chainlang(document_index=input("Enter the document index: "))
    chain.start()


