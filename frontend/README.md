# Frontend Documentation

The **frontend** folder contains all the UI components built using **Streamlit**. This section provides a brief yet comprehensive explanation of each file and its functionality.

## 📂 frontend/ (Main UI Components)
This folder houses the **primary UI components** of the AI-Powered Customer Support System.

### 1️⃣ **chatbot_ui.py** (💬 Chatbot Interface)
**Purpose:**
- This file contains the **main user interface** for interacting with the chatbot.
- Provides an **interactive chat window** where users can enter queries.
- Displays both **user input** and chatbot-generated responses dynamically.

**Key Features:**
- **Chat History Management:** Displays past messages in the chat.
- **Real-time Response Handling:** Calls backend functions directly to generate responses.
- **Multilingual Support 🌍:** Automatically detects and translates input/output languages.

---

### 2️⃣ **chatbot_analytics.py** (📊 Analytics Dashboard)
**Purpose:**
- Provides a **dashboard** to analyze customer queries, chatbot interactions, and trends.
- Uses **Plotly** to generate interactive and visually appealing charts.

**Key Features:**
- **Sentiment Distribution:** Displays the breakdown of customer sentiments (Positive, Negative, Neutral).
- **Sentiment Trends Over Time:** Shows how customer sentiment changes month-to-month.
- **FAQ Category Analysis:** Highlights the most frequently asked question categories.
- **User Engagement Heatmap:** Displays peak hours of chatbot usage.
- **Yearly Sentiment Trends:** Users can select a year and analyze monthly trends.

---

## 📂 frontend/streamlit/ (Additional Streamlit Components)
This folder contains additional **Streamlit-based UI enhancements**.

### 3️⃣ **chatbot_analytics.py** (📊 Extended Analytics UI)
**Purpose:**
- An extended analytics dashboard version with **more in-depth data visualizations**.
- Provides a **drill-down analysis** of chatbot performance and customer interactions.

**Key Features:**
- **Advanced Sentiment Charts:** Includes both **monthly and yearly** customer sentiment analysis.
- **Enhanced Data Visualization:** Utilizes **animated charts, tooltips, and interactive filtering.**
- **Comparison Metrics:** Compare chatbot engagement levels **over different time periods.**

---

## 🌟 Summary
The **frontend** folder serves as the **user-facing interface** for interacting with the chatbot and viewing analytics. It consists of:
- **Chatbot UI (`chatbot_ui.py`)** → Provides a conversational interface.
- **Analytics Dashboard (`chatbot_analytics.py`)** → Displays interactive customer insights.
- **Extended Analytics (`streamlit/chatbot_analytics.py`)** → Offers more in-depth data visualization.

These components together ensure a **seamless user experience**, allowing users to interact with the chatbot efficiently and monitor customer engagement trends effectively. 🚀

