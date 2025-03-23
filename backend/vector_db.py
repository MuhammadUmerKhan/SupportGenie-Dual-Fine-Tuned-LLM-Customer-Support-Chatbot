from langchain_huggingface import HuggingFaceEmbeddings
import chromadb, pymongo, uuid, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config.config as CONFIG
from logger import get_logger

logger = get_logger(__name__)

def connect_mongo():
    """Connect to MongoDB."""
    try:
        client = pymongo.MongoClient(CONFIG.MONGO_URI)
        db = client[CONFIG.DB_NAME]
        if db is None:
            raise Exception("Database connection failed.")
        return db
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return None

def load_faqs_from_mongo():
    """Retrieve FAQs from MongoDB."""
    try:
        db = connect_mongo()
        if db is None:
            raise Exception("Database connection failed.")
        faqs = list(db[CONFIG.FAQ_COLLECTION].find({}, {"_id": 0, "question": 1, "answer": 1}))
        if not faqs:
            raise Exception("No FAQs found in MongoDB.")
        return faqs
    except Exception as e:
        logger.error(f"Error loading FAQs from MongoDB: {e}")
        return []

def reset_chroma_db():
    """Delete existing ChromaDB collection to reset the database."""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        chroma_client.delete_collection("faq_embeddings")  # ✅ Delete old collection
        logger.info("✅ ChromaDB collection reset successfully.")
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {e}")

def store_faqs_in_chroma():
    """Store FAQ embeddings in ChromaDB after resetting the database."""
    try:
        reset_chroma_db()  # ✅ First, clear the existing database
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection("faq_embeddings")

        # Load FAQ Data
        faqs = load_faqs_from_mongo()
        if not faqs:
            raise Exception("No FAQs found in MongoDB.")

        questions = [faq["question"] for faq in faqs]
        answers = {faq["question"]: faq["answer"] for faq in faqs}

        # ✅ Use Hugging Face Cloud Embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        embeddings = embedding_model.embed_documents(questions)

        # ✅ Store fresh data in ChromaDB
        for question, embedding in zip(questions, embeddings):
            unique_id = str(uuid.uuid4())  # Generate a stable unique ID
            
            collection.add(
                ids=[unique_id],
                embeddings=[embedding],
                metadatas=[{"question": question, "answer": answers[question]}]
            )

        logger.info("✅ FAQs stored in ChromaDB successfully after reset.")
    except Exception as e:
        logger.error(f"Error storing FAQs in ChromaDB: {e}")

if __name__ == "__main__":
    store_faqs_in_chroma()
