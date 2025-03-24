import streamlit as st
import os, sys, pandas as pd, plotly.express as px, plotly.graph_objects as go
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config.utils as utils
import config.config as CONFIG
from backend.chatbot import get_chatbot_response, connect_mongo

# Function to Run Chatbot UI
def get_chat_history():
    """Fetch chat history from MongoDB"""
    db = connect_mongo()
    if db is None:
        return {"error": "Database connection failed."}
    chat_collection = db[CONFIG.CHAT_HISTORY_COLLECTION]
    chat_history = list(chat_collection.find({}, {"_id": 0}))  # Remove MongoDB ID field
    return {"data": chat_history}
def chatbot():
    """Displays the AI Chatbot Interface without API Calls."""
    st.markdown("<h1 style='text-align: center;'>💬 AI Customer Support Chatbot</h1>", unsafe_allow_html=True)
    st.write("Ask any question or click on a common FAQ below:")

    # Initialize Chat History
    utils.enable_chat_history(lambda: None)

    # Handle User Input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        utils.display_msg(user_input, "user")  # Display user's message in chat
        bot_response = get_chatbot_response(user_input)  # Call function directly
        utils.display_msg(bot_response, "assistant")  # Store bot's response in chat history

# Function to Run Analytics Dashboard

def analytics():
    """Displays the AI Analytics Dashboard without API Calls."""
    st.title("📊 AI Customer Support - Analytics Dashboard")

    # Fetch Data Directly from MongoDB
    chat_data = get_chat_history()
    if "error" in chat_data:
        st.error(chat_data["error"])
        st.stop()

    df = pd.DataFrame(chat_data["data"])

    # Convert Timestamp Column
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 📌 Sentiment Distribution - 3D Pie Chart
    st.subheader("🧠 Sentiment Distribution")
    sentiment = df[df['sentiment'] != "Not a Review"]
    sentiment_counts = sentiment["sentiment"].value_counts()
    fig_sentiment_pie = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Customer Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={"Positive": "#28a745", "Negative": "#dc3545"},
        hole=0.3
    )
    fig_sentiment_pie.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0.05])
    with st.expander("", expanded=True):
        st.plotly_chart(fig_sentiment_pie, use_container_width=True)

    st.subheader("📈 Sentiment Trends Over Time")

    df_sentiment = df[df['sentiment'] != "Not a Review"]
    df_sentiment["timestamp"] = pd.to_datetime(df_sentiment["timestamp"])
    df_sentiment["year_month"] = df_sentiment["timestamp"].dt.to_period("M")
    df_sentiment_time = df_sentiment.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)

    for sentiment in ["Positive", "Negative", "Neutral"]:
        if sentiment not in df_sentiment_time.columns:
            df_sentiment_time[sentiment] = 0

    # ✅ Create Plotly Figure
    fig_sentiment_trend = go.Figure()
    colors = {"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#FFA500"}  # Added Neutral color

    for sentiment in ["Positive", "Negative", "Neutral"]:
        fig_sentiment_trend.add_trace(go.Scatter(
            x=df_sentiment_time.index.astype(str),  # Convert Period Index to String
            y=df_sentiment_time[sentiment],
            mode='lines+markers',
            stackgroup='one',
            name=sentiment,
            line=dict(color=colors[sentiment])
        ))

    # ✅ Update Layout for Readability
    fig_sentiment_trend.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Year-Month",
        yaxis_title="Sentiment Count",
        template="plotly_dark",
        hovermode="x"
    )

    # ✅ Display Chart
    with st.expander("", expanded=False):
        st.plotly_chart(fig_sentiment_trend, use_container_width=True)


    st.subheader("📈 Most Frequently Asked Question Categories")
    # 📌 Most Asked FAQs - Advanced Horizontal Bar Chart
    category_counts = df["category"].value_counts()
    fig_category = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={"x": "Category", "y": "Number of Questions"},
        title="Most Frequently Asked Question Categories",
        color=category_counts.values,
        color_continuous_scale="blues"
    )

    fig_category.update_traces(marker=dict(line=dict(width=2, color="black")))
    with st.expander("", expanded=False):
        st.plotly_chart(fig_category, use_container_width=True)
    
    # ====== 📌 User Engagement Heatmap (Hourly) ======
    st.subheader("🔥 User Engagement by Time of Day")
    df["hour"] = df["timestamp"].dt.hour
    hourly_counts = df["hour"].value_counts().sort_index()

    fig_heatmap = go.Figure(go.Heatmap(
        z=hourly_counts.values.reshape(1, -1),
        x=hourly_counts.index,
        colorscale="reds"
    ))
    fig_heatmap.update_layout(
        title="User Engagement Across Different Hours",
        xaxis_title="Hour of Day",
        yaxis_title="Engagement Level",
        template="plotly_dark"
    )
    with st.expander("", expanded=False):
        st.plotly_chart(fig_heatmap, use_container_width=True)