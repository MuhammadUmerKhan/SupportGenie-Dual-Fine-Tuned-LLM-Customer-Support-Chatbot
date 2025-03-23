from fastapi import FastAPI, Query
from pydantic import BaseModel
from backend.chatbot import get_chatbot_response
from fastapi.middleware.cors import CORSMiddleware
import logging, pymongo, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config.config as CONFIG

# Initialize FastAPI App
app = FastAPI()

# Enable CORS for Frontend Communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Logging
logging.basicConfig(filename="logs/api.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Define Request Model
class ChatRequest(BaseModel):
    query: str

# Define API Route for Chatbot
@app.get("/chat")
def chat(query: str = Query(..., description="User's question to the chatbot")):
    try:
        response = get_chatbot_response(query)
        logging.info(f"User Query: {query} | Bot Response: {response}")
        return {"user": query, "bot": response}
    except Exception as e:
        logging.error(f"Error in chatbot API: {str(e)}")
        return {"error": "Something went wrong!"}
    
def connect_mongo():
    """Connect to MongoDB."""
    try:
        client = pymongo.MongoClient(CONFIG.MONGO_URI)
        db = client[CONFIG.DB_NAME]
        if db is None:
            raise Exception("Database connection failed.")
        return db
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        return None
    
@app.get("/chat-history")
def get_chat_history():
    """Fetch chat history from MongoDB"""
    db = connect_mongo()
    if db is None:
        return {"error": "Database connection failed."}
    chat_collection = db[CONFIG.CHAT_HISTORY_COLLECTION]
    chat_data = list(chat_collection.find({}, {"_id": 0}))  # Remove MongoDB ID field
    return {"data": chat_data}

# Root API Endpoint
@app.get("/")
def root():
    return {"message": "AI Chatbot API is running!"}