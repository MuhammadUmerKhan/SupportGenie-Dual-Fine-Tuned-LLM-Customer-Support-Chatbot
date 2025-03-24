import streamlit as st
import os, sys, requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config.utils as utils

# Function to Run Chatbot UI
def chatbot():
    """Displays the AI Chatbot Interface."""
    st.markdown("<h1 style='text-align: center;'>💬 AI Customer Support Chatbot</h1>", unsafe_allow_html=True)
    st.write("Ask any question or click on a common FAQ below:")

    # Initialize Chat History
    utils.enable_chat_history(lambda: None)

    # Backend API URL
    API_URL = "http://127.0.0.1:8000/chat"

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
        utils.display_msg(bot_response, "assistant")  # Store bot's response in chat history
