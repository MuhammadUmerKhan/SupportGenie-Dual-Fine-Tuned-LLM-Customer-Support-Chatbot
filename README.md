# 🤖 **AI-Powered Customer Support System**

## 🚀 **Project Overview**
The **AI-Powered Customer Support System** is an intelligent chatbot designed to enhance customer service by **answering queries**, **analyzing user sentiment**, and **providing insightful analytics**. The system integrates **FAQs, LLM-based responses, sentiment analysis, and feedback collection** to continuously improve customer interactions.

💡 **Why This Project?**  
- ✅ **AI-Driven Chatbot** → Uses **FAQ matching** and **LLM** for dynamic responses.
- ✅ **Multilingual Support** → Detects input language & translates it for better comprehension.
- ✅ **Sentiment Analysis & Trends** → Tracks **customer emotions** over time.
- ✅ **Real-Time Analytics Dashboard** → Displays customer insights using **interactive charts**.
- ✅ **MongoDB & FAISS Integration** → Efficient storage & retrieval of FAQs and chat history.
- ✅ **Streamlit UI for Chatbot & Analytics** → Easy-to-use customer support platform.

---

## 📁 **Table of Contents**
- [📌 Features](#-features)
- [🛠️ Tech Stack](#-tech-stack)
- [📂 Project Structure](#-project-structure)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🚀 Running the Application](#-running-the-application)
- [📊 AI-Powered Analytics Dashboard](#-ai-powered-analytics-dashboard)
- [🌐 Deployment Guide](#-deployment-guide)
- [📌 Future Improvements](#-future-improvements)

---

## 📌 **Features**
🔹 **AI Chatbot with FAQ Matching** → Finds the best **FAQ-based response** using **FAISS indexing**.  
🔹 **Multilingual Support** → Detects input language & translates queries before processing.  
🔹 **Real-Time Sentiment Analysis** → Categorizes user interactions as **Positive, Negative, or Neutral**.  
🔹 **Feedback System (👍👎)** → Users can rate responses, helping to improve chatbot accuracy.  
🔹 **Analytics Dashboard** → Tracks **sentiment trends, category insights, and user activity**.  
🔹 **MongoDB for Chat History** → Stores user queries and bot responses for future reference.  
🔹 **Fast Search with FAISS** → Uses **vector search** for **efficient FAQ retrieval**.  
🔹 **Interactive UI (Streamlit)** → Clean and user-friendly interface for chatbot & analytics.  

---

## 🛠️ **Tech Stack**
| Technology | Usage |
|------------|-------|
| **Python** | Backend API, Chatbot, Data Processing |
| **Streamlit** | Frontend UI for chatbot and analytics dashboard |
| **MongoDB Atlas** | Stores chat history and FAQs |
| **FAISS** | Efficient **vector search** for FAQ retrieval |
| **Hugging Face Transformers** | Embedding model for vector similarity |
| **TextBlob** | Sentiment Analysis |
| **Plotly** | Visualization in **analytics dashboard** |
| **FastAPI (Optional)** | API layer for chatbot (if deployed separately) |

---

## 📂 **Project Structure**
```
AI-Powered-Customer-Support-System/
│
├── backend/               # Backend Logic & Core Processing
│   ├── chatbot.py         # Core Chatbot Logic (LLM, FAQ, Sentiment, Category)
│   ├── faq_loader.py      # Loads FAQ Data into MongoDB
│   ├── vector_db.py       # FAISS for FAQ Embeddings
│
├── frontend/              # Streamlit UI Components
│   ├── chatbot_ui.py      # Chatbot Interface
│   ├── chatbot_analytics.py # Analytics Dashboard UI
│
├── faiss_db/              # FAISS Storage for Vector Search
│   ├── faiss_index.bin    # FAISS Index
│   ├── faiss_metadata.json # FAQ Metadata for indexing
│
├── FAQs/                  # FAQ Dataset
│   ├── BankFAQs.csv       # Raw FAQ Dataset
│   ├── processed_faqs.json # Preprocessed FAQ Data
│
├── logs/                  # Logging & Monitoring
│   ├── chatbot.log        # Logs for Chatbot Responses
│
├── config/                # Configuration & Utility Functions
│   ├── config.py          # Global Configuration
│   ├── utils.py           # Helper Functions
│
├── requirements.txt       # Python Dependencies
├── main.py                # Entry Point for Streamlit App
├── Dockerfile             # Docker Deployment Configuration
└── README.md              # Project Documentation
```

---
## 📊 **Dashboard Analytics**


| Visualization Type      | Distribution/Insights |
|------------------------|----------------------|
| **Most Frequently Asked Questions** | ![Feature Importance](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/most_fre_ques.png) |
| **Sentiments Over Time**   | ![Confusion Matrix](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sent_ovr_time.png) |
| **Sentiment Trend** | ![Churn Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sent_trend.png) |
| **Sentiment Distribution**    | ![Customer Tenure Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sentiment_distribution.png) |

---

## ⚙️ **Setup and Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/AI-Powered-Customer-Support-System.git
cd AI-Powered-Customer-Support-System
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up MongoDB Atlas**
- Create a **MongoDB Atlas** cluster.
- Add your **connection string** to `.env` file:
```ini
MONGO_URI=mongodb+srv://your_user:your_password@cluster.mongodb.net/chatbotDB
```

### **4️⃣ Run the Streamlit App**
```bash
streamlit run main.py
```

---

## 📊 **AI-Powered Analytics Dashboard**
The **Analytics Dashboard** provides insights into chatbot interactions.
- 📈 **Sentiment Trends** → Tracks how users feel about responses.
- 🔥 **Most Asked Questions** → Identifies common customer concerns.
- 🕒 **User Engagement Heatmap** → Shows peak chat hours.
- ✅ **Feedback Ratings** → Measures helpful vs. unhelpful responses.

---

## 🌐 **Deployment Guide**
### **1️⃣ Deploy on Streamlit Cloud**
- Push your repository to **GitHub**.
- Go to **[Streamlit Cloud](https://streamlit.io/cloud)** and deploy your repo.

### **2️⃣ Deploy using Docker**
```bash
# Build Docker Image
docker build -t ai-customer-support .

# Run Container
docker run -p 8501:8501 ai-customer-support
```

---

## 📌 **Future Improvements**
🔹 **Voice-Enabled Chatbot** – Integrate **speech recognition** for voice queries.  
🔹 **WhatsApp & Telegram Integration** – Expand support to messaging apps.  
🔹 **Advanced Sentiment Analysis** – Use transformer models for better predictions.  
🔹 **Proactive Support Suggestions** – Predict user needs based on chat history.  

---

## 📌 **Conclusion**
The **AI-Powered Customer Support System** provides **seamless, intelligent customer interactions** through **FAQ retrieval, sentiment analysis, and analytics**. With **scalable deployment** and **real-time insights**, this project can revolutionize **customer engagement** across multiple industries. 🚀

🔹 **Want to contribute?** Fork the repo and submit a PR! 🎉

