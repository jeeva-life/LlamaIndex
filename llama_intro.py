import os
import logging
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers.type import ResponseMode 
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Settings

# ----------------------------
# Logging Configuration
# ----------------------------
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ----------------------------
# Environment Variable Loader
# ----------------------------
def load_api_key():
    """
    Load OpenAI API Key from environment variables.
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPEN_API_KEY")
        if not api_key:
            logger.error("OPEN_API_KEY not found. Please set the API key in environment variables.")
            raise ValueError("OPEN_API_KEY is missing.")
        os.environ["OPEN_API_KEY"] = api_key
        logger.info("Successfully loaded OpenAI API key from environment.")
        return api_key
    except Exception as e:
        logger.exception("Error loading API key.")
        raise e

# ----------------------------
# LLM Configuration
# ----------------------------
def configure_llm(api_key):
    """
    Configure OpenAI LLM settings.
    """
    try:
        Settings.llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")
        logger.info("LLM successfully configured with OpenAI settings.")
    except Exception as e:
        logger.exception("Error configuring LLM.")
        raise e

# ----------------------------
# Document Loader and Indexing
# ----------------------------
def load_documents_and_index(data_dir):
    """
    Load documents from a specified directory and create a VectorStoreIndex.
    """
    try:
        if not os.path.exists(data_dir):
            logger.error(f"Data directory '{data_dir}' does not exist.")
            raise FileNotFoundError(f"Directory '{data_dir}' not found.")
        
        logger.info("Loading documents from directory: %s", data_dir)
        try:
            documents = SimpleDirectoryReader(data_dir).load_data()
        except Exception as reader_error:
            logger.error("Failed to load data using SimpleDirectoryReader.")
            raise RuntimeError(f"Document loading failed: {reader_error}") from reader_error

        if not documents:
            logger.warning("No documents found in the specified directory.")
            raise ValueError("No documents available to load.")
        
        logger.info("Successfully loaded %d documents.", len(documents))
        # Create the VectorStoreIndex
        try:
            index = VectorStoreIndex.from_documents(documents)
        except Exception as index_error:
            logger.error("Failed to create VectorStoreIndex.")
            raise RuntimeError(f"Index creation failed: {index_error}") from index_error

        logger.info("VectorStoreIndex created successfully.")
        return index, len(documents)
    except FileNotFoundError as fnf_error:
        logger.error("FileNotFoundError: %s", fnf_error)
        raise
    except ValueError as ve_error:
        logger.warning("ValueError: %s", ve_error)
        raise
    except RuntimeError as rt_error:
        logger.error("RuntimeError: %s", rt_error)
        raise
    except Exception as e:
        logger.exception("An unexpected error occurred while processing documents or creating the index.")
        raise

# ----------------------------
# Query Engine Setup
# ----------------------------
def setup_query_engine(index):
    """
    Configure and return the retriever query engine.
    """
    try:
        retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.8)
        synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=[postprocessor]
        )
        logger.info("RetrieverQueryEngine successfully configured.")
        return query_engine
    except Exception as e:
        logger.exception("Error setting up query engine.")
        raise e

# ----------------------------
# Query Execution
# ----------------------------
def execute_query(query_engine, query_text):
    """
    Execute a query and return the result.
    """
    try:
        logger.info("Executing query: %s", query_text)
        response = query_engine.query(query_text)
        logger.info("Query executed successfully.")
        return response
    except Exception as e:
        logger.exception("Error executing query.")
        raise e

# ----------------------------
# Main Function
# ----------------------------
def main():
    """
    Main entry point for loading documents, creating index, and querying the system.
    """
    try:
        logger.info("Starting the application.")
        
        # Load API key and configure LLM
        api_key = load_api_key()
        configure_llm(api_key)
        
        # Load documents and create index
        data_directory = "data"
        index, doc_count = load_documents_and_index(data_directory)
        logger.info("Number of documents loaded: %d", doc_count)
        
        # Set up query engine
        query_engine = setup_query_engine(index)
        
        # Execute a sample query
        sample_query = "What is the content of the documents?"
        response = execute_query(query_engine, sample_query)
        
        logger.info("Response: %s", response)
        print(f"Query Response:\n{response}")
    
    except Exception as e:
        logger.critical("Application failed to execute due to an error: %s", str(e))
    finally:
        logger.info("Application execution completed.")

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
