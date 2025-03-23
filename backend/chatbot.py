from datetime import datetime
import pymongo
import chromadb
import dotenv
from textblob import TextBlob
from langchain_huggingface import HuggingFaceEmbeddings
import config.config as CONFIG
from langchain_groq import ChatGroq
from logger import get_logger

dotenv.load_dotenv()
logger = get_logger(__name__)

def connect_mongo():
    """Connect to MongoDB."""
    try:
        client = pymongo.MongoClient(CONFIG.MONGO_URI)
        return client[CONFIG.DB_NAME]
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return None

def initialize_chatgroq():
    """Initialize ChatGroq model."""
    try:
        return ChatGroq(
            temperature=0,
            groq_api_key=CONFIG.GROK_API_KEY,
            model_name=CONFIG.MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        return None

def search_faq(user_input):
    """Search for the most relevant FAQ using ChromaDB."""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection("faq_embeddings")
        
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        user_embedding = embedding_model.embed_query(user_input)

        results = collection.query(query_embeddings=[user_embedding], n_results=1)

        if results and results["ids"] and results["metadatas"][0]:
            return results["metadatas"][0][0].get("answer", "No answer found in FAQ.")
        return None  # Ensure it returns None if no match is found

    except Exception as e:
        logger.error(f"Error searching FAQs: {e}")
        return None

def store_chat_history(user_input, bot_reply):
    """Save user queries and bot responses to MongoDB with timestamps."""
    try:
        db = connect_mongo()
        if not db:
            raise Exception("Database connection failed.")

        chat_collection = db[CONFIG.CHAT_HISTORY_COLLECTION]
        chat_collection.insert_one({
            "user": user_input,
            "bot": bot_reply,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"Chat history stored: User -> {user_input} | Bot -> {bot_reply}")

    except Exception as e:
        logger.error(f"Error storing chat history: {e}")

def correct_spelling(text):
    """Auto-correct spelling mistakes."""
    return str(TextBlob(text).correct())

def get_chatbot_response(user_input):
    """Generate a response from FAQs or ChatGroq."""
    try:
        user_input = correct_spelling(user_input)

        # Step 1: Check ChromaDB for FAQ answer
        faq_answer = search_faq(user_input)
        if faq_answer:
            return f"(From FAQ) {faq_answer}"

        # Step 2: If not found, use ChatGroq
        llm = initialize_chatgroq()
        response = llm.invoke(user_input)
        bot_reply = response.content

        # Step 3: Store chat history
        store_chat_history(user_input, bot_reply)

        return bot_reply

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, an error occurred."

def main():
    """Main function to run chatbot."""
    llm = initialize_chatgroq()
    if not llm:
        print("Failed to initialize AI model. Exiting.")
        return

    while True:
        user_query = input("\nAsk me anything: ")
        response = get_chatbot_response(user_query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
