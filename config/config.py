import dotenv, os, sys, os
from urllib.parse import quote_plus
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import get_logger

# ✅ Configure Logging
logger = get_logger(__name__)

try:
    # ✅ Load environment variables
    dotenv.load_dotenv()

    # ✅ Load API Key & Model Name
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    if not GROK_API_KEY:
        logger.warning("⚠️ GROK_API_KEY is missing. Some features may not work.")

    MODEL_NAME = "qwen-2.5-32b"

    # ✅ Load FAQ File Path
    FAQ_PATH = "./FAQS/BankFAQs.csv"
    if not os.path.exists(FAQ_PATH):
        logger.warning(f"⚠️ FAQ file not found at {FAQ_PATH}")

    # ✅ Load MongoDB Credentials
    DB_NAME = os.getenv("MONGO_DB", "chatbotDB")  # Default value if not set
    FAQ_COLLECTION = "faqs"
    CHAT_HISTORY_COLLECTION = "chat_history"

    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

    if not MONGO_USER or not MONGO_PASSWORD or not MONGO_CLUSTER:
        raise ValueError("❌ MongoDB credentials are missing! Check .env file.")

    # ✅ Securely encode password
    MONGO_PASSWORD = quote_plus(MONGO_PASSWORD)

    # ✅ Construct MongoDB URI
    MONGO_URI = (
        f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/{DB_NAME}"
        "?retryWrites=true&w=majority&appName=Cluster0"
    )

    logger.info("✅ Configuration loaded successfully.")

except Exception as e:
    logger.error(f"❌ Error loading configuration: {e}")
    raise  # Re-raise error to prevent silent failures
