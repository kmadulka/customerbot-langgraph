import openai
import os
from langchain_openai import ChatOpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Global LLM instance (configure model here)
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)
