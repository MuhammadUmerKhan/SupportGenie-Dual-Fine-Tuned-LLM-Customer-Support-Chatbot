import csv, pymongo, os, sys
import scripts.config as CONFIG
from scripts.logger import get_logger

logger = get_logger(__name__)

def connect_mongo():
    """Connect to MongoDB."""
    try:
        client = pymongo.MongoClient(CONFIG.MONGO_URI)
        print("✅ MongoDB connection established.")
        return client[CONFIG.DB_NAME]
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        # logger.error(f"MongoDB connection failed: {e}")
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
                print(f"CSV Headers Detected: {headers}")
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

        logger.info("✅ FAQs loaded into MongoDB successfully.")

    except Exception as e:
        logger.error(f"Error loading FAQs into MongoDB: {e}")

if __name__ == "__main__":
    # print(connect_mongo())
    load_faq_to_mongo(CONFIG.FAQ_PATH)