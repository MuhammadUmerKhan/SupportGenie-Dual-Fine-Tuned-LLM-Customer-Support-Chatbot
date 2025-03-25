import dotenv, os
from urllib.parse import quote_plus

# Load environment variables
dotenv.load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
MODEL_NAME = "qwen-2.5-32b"
FAQ_PATH = "./FAQS/BankFAQs.csv"
DB_NAME = os.getenv("MONGO_DB", "chatbotDB")  # Default to chatbotDB if not set
FAQ_COLLECTION = "faqs"
CHAT_HISTORY_COLLECTION = "chat_history"

# Securely encode MongoDB password
MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = quote_plus(os.getenv("MONGO_PASSWORD"))  # Encode special characters
MONGO_CLUSTER = os.getenv("MONGO_CLUSTER")

# Construct MongoDB URI
MONGO_URI = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_CLUSTER}/{DB_NAME}?retryWrites=true&w=majority&appName=Cluster0"