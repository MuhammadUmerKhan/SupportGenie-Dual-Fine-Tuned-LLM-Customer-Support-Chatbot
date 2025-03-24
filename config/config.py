import dotenv, os
from urllib.parse import quote_plus
# Load environment variables
dotenv.load_dotenv()

GROK_API_KEY = os.getenv("GROK_API_KEY")
MODEL_NAME = "qwen-2.5-32b"
FAQ_PATH = "./FAQS/BankFAQs.csv"
DB_NAME = "chatbotDB"
FAQ_COLLECTION = "faqs"
CHAT_HISTORY_COLLECTION = "chat_history"
encoded_password = quote_plus("MUK546@!")  # Replace this with your actual password
MONGO_URI = f"mongodb+srv://muhammadumerk546:{encoded_password}@cluster0.tgibo.mongodb.net/chatbotDB?retryWrites=true&w=majority&appName=Cluster0"