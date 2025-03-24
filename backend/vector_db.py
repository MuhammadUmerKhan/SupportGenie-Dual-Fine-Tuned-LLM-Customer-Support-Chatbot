import chromadb, pymongo, uuid, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config.config as CONFIG
import config.utils as utils
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
    """Retrieve FAQs from MongoDB with indexing for speed."""
    try:
        db = connect_mongo()
        if db is None:
            raise Exception("Database connection failed.")

        # ✅ Ensure an index is created on the 'question' field
        db[CONFIG.FAQ_COLLECTION].create_index("question")

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
    """Store ONLY new FAQ embeddings in ChromaDB."""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        collection = chroma_client.get_or_create_collection("faq_embeddings")

        # Load FAQ Data
        faqs = load_faqs_from_mongo()
        if not faqs:
            raise Exception("No FAQs found in MongoDB.")

        # ✅ Check if this question already exists in ChromaDB
        existing_questions = {metadata["question"] for metadata in collection.get()["metadatas"]}

        # ✅ Filter only new questions that are NOT in ChromaDB
        new_faqs = [faq for faq in faqs if faq["question"] not in existing_questions]

        if not new_faqs:
            logger.info("✅ No new FAQs to store. ChromaDB is up to date.")
            return  # ✅ Skip processing if no new FAQs

        # Extract only new questions
        questions = [faq["question"] for faq in new_faqs]
        answers = {faq["question"]: faq["answer"] for faq in new_faqs}

        # ✅ Use Hugging Face Cloud Embeddings
        embedding_model = utils.configure_vector_embeddings()
        embeddings = embedding_model.embed_documents(questions)

        # ✅ Store only new FAQs in ChromaDB
        for question, embedding in zip(questions, embeddings):
            unique_id = str(uuid.uuid4())  # Generate unique ID
            collection.add(
                ids=[unique_id],
                embeddings=[embedding],
                metadatas=[{"question": question, "answer": answers[question]}]
            )

        logger.info(f"✅ {len(new_faqs)} new FAQs stored in ChromaDB successfully.")

    except Exception as e:
        logger.error(f"Error storing FAQs in ChromaDB: {e}")


if __name__ == "__main__":
    store_faqs_in_chroma()
