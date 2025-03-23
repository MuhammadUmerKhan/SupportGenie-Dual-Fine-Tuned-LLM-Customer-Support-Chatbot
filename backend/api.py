from fastapi import FastAPI, Query
from pydantic import BaseModel
from backend.chatbot import get_chatbot_response
from fastapi.middleware.cors import CORSMiddleware
import logging

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

# Root API Endpoint
@app.get("/")
def root():
    return {"message": "AI Chatbot API is running!"}