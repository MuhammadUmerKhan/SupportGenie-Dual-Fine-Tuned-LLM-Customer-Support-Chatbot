import streamlit as st

# Uncomment these lines to use API instead of direct function calls
# from frontend.api import chatbot_ui    
# from frontend.api import analytics_ui  

from frontend.streamlit_files import chatbot_analytics

# Streamlit Page Config
st.set_page_config(page_title="AI Customer Support System", page_icon="🤖", layout="wide")

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/295/295128.png", width=100)
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "💬 Chatbot", "🔧 Fine Tuned Bot", "📶 Analytics Dashboard", "📖 FAQs"])

# Home Page
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>🤖 AI Customer Support System</h1>", unsafe_allow_html=True)
    st.markdown("""
    ## 🌟 Welcome to the AI-Powered Customer Support System!
    This intelligent chatbot system is designed to **enhance customer interactions** by providing instant support, analyzing sentiment, and tracking trends.
    
    ---
    ### 🚀 **Key Features:**
    - **💬 Smart AI Chatbot:** Answers customer queries using a mix of **predefined FAQs & AI-generated responses**.
    - **📊 Analytics Dashboard:** Gain insights into customer interactions, trends, and engagement.
    - **🧠 Sentiment Analysis:** Tracks and categorizes customer emotions (Positive, Negative, Neutral).
    - **📅 Time-Based Engagement Tracking:** Analyze **peak user activity hours** for better customer support.
    - **📉 Trend Analysis:** Discover emerging trends in customer inquiries.
    
    ---
    ### 🔍 **How It Works:**
    - 1️⃣ **User asks a question** 💬
    - 2️⃣ The chatbot **retrieves the best-matching FAQ answer** 🔍
    - 3️⃣ If no match is found, **AI generates a dynamic response** 🧠
    - 4️⃣ The system **analyzes sentiment & classifies the question category** 📊
    - 5️⃣ All interactions are stored for future **trend analysis & reporting** 📈
    
    ---
    ### 🛠 **How to Use It:**
    - **Go to the Chatbot Page** 🗨️ → Ask any question and get real-time responses.
    - **Explore the Analytics Dashboard** 📊 → Visualize customer trends and insights.
    - **Track Sentiment Over Time** 📅 → Understand customer emotions and engagement.
    
    ---
    ### 🏆 **Why This System is Powerful?**
    - ✅ **Faster Response Times:** AI-driven support for instant answers.
    - ✅ **Better Customer Insights:** Learn what customers are talking about.
    - ✅ **Improved Business Decisions:** Make data-driven improvements to services.
    - ✅ **Enhanced User Experience:** Provide **personalized & engaging** interactions.
    
    ---
    **Ready to get started? Head over to the Chatbot & Analytics sections now!** 🚀
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3203/3203165.png", width=600)

# Chatbot Page
elif page == "💬 Chatbot":
    # chatbot_ui.chatbot()          # Uncomment these lines to use API instead of direct function calls
    chatbot_analytics.chatbot()
elif page == "🔧 Fine Tuned Bot":
    chatbot_analytics.finetuned_chatbot()
# Analytics Dashboard Page
elif page == "📶 Analytics Dashboard":
    # analytics_ui.analytics()      # Uncomment these lines to use API instead of direct function calls
    chatbot_analytics.analytics()
elif page == "📖 FAQs":
        chatbot_analytics.faq_page()