import pymongo, faiss, uuid, sys, os, numpy as np, json
from langchain_huggingface import HuggingFaceEmbeddings
import scripts.config as CONFIG
import scripts.utils as utils
from scripts.logger import get_logger

logger = get_logger(__name__)

# Define FAISS directory path
FAISS_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_db")
FAISS_INDEX_FILE = os.path.join(FAISS_DB_PATH, "faiss_index.bin")
FAISS_METADATA_FILE = os.path.join(FAISS_DB_PATH, "faiss_metadata.json")

def ensure_faiss_db_directory():
    """Ensure that the FAISS database directory exists."""
    if not os.path.exists(FAISS_DB_PATH):
        os.makedirs(FAISS_DB_PATH)
        logger.info(f"✅ Created FAISS DB directory: {FAISS_DB_PATH}")

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
            raise Exception("❌ Database connection failed.")

        db[CONFIG.FAQ_COLLECTION].create_index("question")  # Ensure index
        faqs = list(db[CONFIG.FAQ_COLLECTION].find({}, {"_id": 0, "question": 1, "answer": 1}))

        if not faqs:
            raise Exception("❌ No FAQs found in MongoDB.")
        
        return faqs
    except Exception as e:
        logger.error(f"❌ Error loading FAQs from MongoDB: {e}")
        return []

def reset_faiss_db():
    """Reset FAISS index and metadata."""
    try:
        index = faiss.IndexFlatL2(768)  # Assuming 768-dimension embeddings
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(FAISS_METADATA_FILE, "w") as f:
            json.dump({}, f)  # Reset metadata file
        logger.info("✅ FAISS index reset successfully.")
    except Exception as e:
        logger.error(f"❌ Error resetting FAISS: {e}")

def store_faqs_in_faiss():
    """Store FAQ embeddings in FAISS."""
    try:
        db = connect_mongo()
        if db is None:
            raise Exception("❌ Database connection failed.")

        # Load FAQ Data
        faqs = load_faqs_from_mongo()
        if not faqs:
            raise Exception("❌ No FAQs found in MongoDB.")

        questions = [faq["question"] for faq in faqs]
        answers = [faq["answer"] for faq in faqs]  # Convert to list instead of dict

        # Load Hugging Face Embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        embeddings = embedding_model.embed_documents(questions)
        embeddings_np = np.array(embeddings).astype('float32')

        # Create FAISS index
        index = faiss.IndexFlatL2(embeddings_np.shape[1])
        index.add(embeddings_np)

        # Store FAISS index and metadata
        faiss.write_index(index, FAISS_INDEX_FILE)
        with open(FAISS_METADATA_FILE, "w") as f:
            json.dump(answers, f)  # Store list instead of mapping

        logger.info(f"✅ {len(faqs)} FAQs stored in FAISS successfully.")

    except Exception as e:
        logger.error(f"❌ Error storing FAQs in FAISS: {e}")

if __name__ == "__main__":
    store_faqs_in_faiss()