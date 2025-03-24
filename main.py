import streamlit as st

# Uncomment these lines to use API instead of direct function calls
# from frontend.api import chatbot_ui    
# from frontend.api import analytics_ui  

from frontend.streamlit import chatbot_analytics

# Streamlit Page Config
st.set_page_config(page_title="AI Customer Support System", page_icon="🤖", layout="wide")

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/295/295128.png", width=100)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select a Page:", ["🏠 Home", "📱 Chatbot", "📶 Analytics Dashboard"])

# Home Page
if page == "🏠 Home":
    st.title("🤖 AI Customer Support System")
    st.markdown("""
    Welcome to the **AI Customer Support System**! 🚀 This intelligent chatbot can:
    - 🧠 Answer customer queries using **FAQs & AI responses**.
    - 🎯 Analyze customer **sentiment & trends**.
    - 📊 Provide **insights on customer interactions**.
    
    **How It Works:**
    - 1️⃣ **User asks a question** 💬
    - 2️⃣ The chatbot **retrieves the best FAQ answer** 🔍
    - 3️⃣ If no match is found, **AI generates a response** 🧠
    - 4️⃣ Sentiment & category are **analyzed & stored** 📊
    
    **How to Use It:**
    - Go to the **Chatbot** page and ask questions.
    - View insights in the **Analytics Dashboard**.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3203/3203165.png", width=600)

# Chatbot Page
elif page == "📱 Chatbot":
    # chatbot_ui.chatbot()          # Uncomment these lines to use API instead of direct function calls
    chatbot_analytics.chatbot()

# Analytics Dashboard Page
elif page == "📶 Analytics Dashboard":
    # analytics_ui.analytics()      # Uncomment these lines to use API instead of direct function calls
    chatbot_analytics.analytics()