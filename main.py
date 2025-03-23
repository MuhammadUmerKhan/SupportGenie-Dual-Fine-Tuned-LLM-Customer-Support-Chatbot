import streamlit as st
from frontend import chatbot_ui, analytics_ui 

# Streamlit Page Config
st.set_page_config(page_title="AI Customer Support System", page_icon="🤖", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select a Page:", ["Chatbot", "Analytics Dashboard"])

# Route to Selected Page
if page == "Chatbot":
    chatbot_ui.chatbot()
elif page == "Analytics Dashboard":
    analytics_ui.analytics()
