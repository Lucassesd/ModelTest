from dotenv import load_dotenv
import os

load_dotenv("wc.env")

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
DATABASE=os.getenv("DATABASE")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")