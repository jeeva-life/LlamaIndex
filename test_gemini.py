# pipenv install llama-index llms-gemini google-generativeai llama-index-embeddings-gemini

import logging
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
import os
from dotenv import load_dotenv
load_dotenv()

from llama_index.llms.gemini import Gemini


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Function to load environment variables safely
def load_env_vars():
    try:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key is None:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            raise ValueError("GOOGLE_API_KEY not found.")
        logger.info("Environment variables loaded successfully.")
        return google_api_key
    except Exception as e:
        logger.exception("Error loading environment variables.")
        raise e

# Set your LLM to Google Gemini Model
def create_llm():
    try:
        google_api_key = load_env_vars()  # Make sure the environment variable is set
        llm = Gemini(model="models/gemini-pro")
        logger.info("Google Gemini model created successfully.")
        return llm
    except Exception as e:
        logger.exception("Error creating Google Gemini model.")
        raise e

# Load documents and create the index
def create_index():
    try:
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        logger.info("Index created successfully.")
        return index
    except Exception as e:
        logger.exception("Error creating index.")
        raise e

# Main function to set up retriever and query engine
def main():
    try:
        # Set up the LLM and index
        llm = create_llm()
        index = create_index()

        # Use VectorIndexRetriever to retrieve documents
        retriever = VectorIndexRetriever(index=index,
        similarity_top_k=10)
        query_engine = RetrieverQueryEngine(retriever)
        logger.info("Retriever and query engine set up successfully.")

        # Example: querying (you can replace this with your actual query logic)
        query = "Sample query text"
        results = query_engine.query(query)
        logger.info(f"Query results: {results}")

    except Exception as e:
        logger.exception("Error in main function.")
        raise e

if __name__ == "__main__":
    main()