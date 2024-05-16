from langchain_openai import OpenAIEmbeddings
import os


AI_KEY = "sk-Swi6dHHVWDY342vVaCwFLwmguz6YXfVlSXAfNxzukMtsScfP"
AI_URL = "https://api.chatanywhere.tech/v1"

os.environ["OPENAI_API_KEY"] = AI_KEY
os.environ["OPENAI_API_BASE"] = AI_URL
embeddings_model = OpenAIEmbeddings()
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
print(embedded_query[:6])