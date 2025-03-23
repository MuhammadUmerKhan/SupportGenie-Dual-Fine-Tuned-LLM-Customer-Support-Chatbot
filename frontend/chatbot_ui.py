import streamlit as st
import os, sys, requests

# Ensure Python can find utils and streaming modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import custom utility and streaming functions
import utils
from streaming import StreamHandler

# Streamlit Page Config
st.set_page_config(page_title="AI Customer Support Chatbot", page_icon="💬", layout="wide")

# Backend API URL
API_URL = "http://127.0.0.1:8000/chat"

# Chatbot Header
st.markdown("<h1 style='text-align: center;'>💬 AI Customer Support Chatbot</h1>", unsafe_allow_html=True)
st.write("Ask any question or click on a common FAQ below:")

# Initialize Chat History
utils.enable_chat_history(lambda: None)  # Ensure chat persists across interactions

# Function to Call Backend API
def get_chatbot_response(query):
    try:
        response = requests.get(API_URL, params={"query": query})
        if response.status_code == 200:
            return response.json().get("bot", "Sorry, I couldn't fetch a response.")
        return "Error: Unable to connect to chatbot API."
    except Exception as e:
        return f"Error: {e}"

# Handle User Input
user_input = st.chat_input("Type your question here...")
if user_input:
    utils.display_msg(user_input, "user")  # Display user's message in chat
    bot_response = get_chatbot_response(user_input)

    # Display Bot Response (Avoid Nested st.chat_message)
    assistant_container = st.chat_message("assistant")
    assistant_container.write(bot_response)
    utils.display_msg(bot_response, "assistant")  # Store bot's response in chat history