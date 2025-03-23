import dotenv, os
# Load environment variables
dotenv.load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
MODEL_NAME = "qwen-2.5-32b"
FAQ_PATH = "./FAQS/BankFAQs.csv"
DB_NAME = "chatbotDB"
FAQ_COLLECTION = "faqs"
MONGO_URI = "mongodb://localhost:27017/"
CHAT_HISTORY_COLLECTION = "chat_history"