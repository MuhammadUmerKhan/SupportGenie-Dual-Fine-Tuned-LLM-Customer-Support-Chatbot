import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# Function to Run Analytics Dashboard
def analytics():
    """Displays the AI Analytics Dashboard."""
    st.title("📊 AI Customer Support - Analytics Dashboard")

    # API Endpoint
    API_URL = "http://localhost:8000/chat-history"

    # Fetch Data
    response = requests.get(API_URL)
    if response.status_code == 200:
        chat_data = response.json()["data"]
        df = pd.DataFrame(chat_data)
    else:
        st.error("Failed to load chat data.")
        st.stop()

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

    # 📌 Sentiment Trends Over Time - Enhanced Area Chart
    st.subheader("📈 Sentiment Trends Over Time")
    df_sentiment = df[df['sentiment'] != "Not a Review"]
    df_sentiment_time = df_sentiment.groupby(df_sentiment["timestamp"].dt.date)["sentiment"].value_counts().unstack().fillna(0)
    for sentiment in ["Positive", "Negative"]:
        if sentiment not in df_sentiment_time.columns:
            df_sentiment_time[sentiment] = 0

    fig_sentiment_trend = go.Figure()
    for sentiment, color in zip(["Positive", "Negative"], ["#28a745", "#dc3545"]):
        fig_sentiment_trend.add_trace(go.Scatter(
            x=df_sentiment_time.index, y=df_sentiment_time[sentiment], mode='lines+markers',
            stackgroup='one', name=sentiment, line=dict(color=color)
        ))

    fig_sentiment_trend.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Date", yaxis_title="Count",
        template="plotly_dark",
        hovermode="x"
    )
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
