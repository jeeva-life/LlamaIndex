import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.openai import OpenAI

# Debugging the loaded API key
print(f"OPEN_API_KEY: {os.getenv('OPEN_API_KEY')}")

open_api_key = os.getenv("OPEN_API_KEY")

if not open_api_key:
    raise ValueError("API key not found. Please set the OPEN_API_KEY environment variable.")


response = OpenAI(model = "gpt-3.5-turbo", 
                    openai_api_key = open_api_key).complete("who is bhagath singh")

print(response)
