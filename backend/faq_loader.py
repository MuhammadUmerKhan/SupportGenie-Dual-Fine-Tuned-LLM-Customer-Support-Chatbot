import csv, pymongo, os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config.config as CONFIG
from logger import get_logger

logger = get_logger(__name__)

def connect_mongo():
    """Connect to MongoDB."""
    try:
        client = pymongo.MongoClient(CONFIG.MONGO_URI)
        return client[CONFIG.DB_NAME]
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return None

def load_faq_to_mongo(faq_path):
    """Load FAQs from a CSV file into MongoDB."""
    try:
        db = connect_mongo()
        if db is None:
            raise Exception("Database connection failed.")

        collection = db[CONFIG.FAQ_COLLECTION]

        # Read CSV file
        faqs = []
        with open(faq_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames  

            # ✅ Print headers only once for debugging
            if headers:
                logger.info(f"CSV Headers Detected: {headers}")  

            for row in reader:
                faqs.append({
                    "question": row.get("Question", "").strip(),
                    "answer": row.get("Answer", "").strip(),
                    "category": row.get("Class", "").strip()
                })

        if not faqs:
            raise Exception("CSV file is empty or not formatted correctly.")

        collection.delete_many({})
        collection.insert_many(faqs)
        print("✅ FAQs loaded into MongoDB successfully.")
        logger.info("✅ FAQs loaded into MongoDB successfully.")

    except Exception as e:
        logger.error(f"Error loading FAQs into MongoDB: {e}")

if __name__ == "__main__":
    # print(connect_mongo())
    load_faq_to_mongo(CONFIG.FAQ_PATH)