import os
from llama_index.llms.gemini import Gemini

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Set GEMINI_API_KEY in .env")

llm = Gemini(model="models/gemini-pro", api_key=api_key)
response = llm.complete("Yo Gemini, can you confirm you're alive?")
print(response.text)
