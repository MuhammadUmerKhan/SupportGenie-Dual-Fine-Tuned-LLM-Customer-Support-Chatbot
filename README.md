# 📌 **SupportGenie: AI-Powered Customer Assistance & Insights**

![ai_chatbot.png](https://www.addevice.io/storage/ckeditor/uploads/images/64d0d72b8dcde_the.role.of.chatbots.and.humans.in.customer.support.1.png)

## 🚀 **Project Overview**
In today's digital world, businesses need **efficient and scalable** customer support solutions. This project leverages **AI-powered chatbots, FAQ retrieval, sentiment analysis, and analytics dashboards** to enhance customer experience.

💡 **What makes this project unique?**
- 👉 **AI-Powered Chatbot** → Retrieves responses from **FAQs** and generates answers using **LLMs**.
- 👉 **Multilingual Support** → Automatically detects **input language**, translates it into **English**, processes it, and responds in the original language.
- 👉 **Sentiment Analysis** → Understands customer emotions to classify interactions as **positive, negative, or neutral**.
- 👉 **FAISS Vector Search** → Stores and retrieves **FAQ embeddings** for **fast and accurate** responses.
- 👉 **MongoDB Integration** → Stores **chat history, feedback, and analytics**.
- 👉 **Interactive Analytics Dashboard** → Provides **data insights** on chatbot interactions and sentiment trends.
- 👉 **Streamlit UI** → Web-based **interactive chatbot and analytics dashboard**.

---

## **📁 Table of Contents**
- [📌 Problem Statement](#-problem-statement)
- [🛠️ Solution Approach](#-solution-approach)
- [🔥 Project Features](#-project-features)
- [📊 AI-Powered Chatbot](#-ai-powered-chatbot)
- [📈 Analytics Dashboard](#-analytics-dashboard)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🚀 Running the Chatbot & Analytics](#-running-the-chatbot--analytics)
- [🖥️ Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
- [🛠️ Future Improvements](#-future-improvements)
- [📌 Conclusion](#-conclusion)

---

## 📌 **Problem Statement**
Customer support teams face **high workloads and delays**, leading to **poor user experience**. The challenge is:
**"Can we automate responses to common queries while understanding customer sentiment and improving support?"**

To solve this, we need:
- 👉 A **fast & accurate chatbot** to **handle FAQs** automatically.
- 👉 **Sentiment analysis** to categorize **customer feedback**.
- 👉 **Real-time analytics** to monitor trends and **optimize responses**.

---

## 🛠️ **Solution Approach**
Our solution uses **AI chatbots, NLP, and analytics** to **automate and improve customer interactions**.

### **1️⃣ FAQ-Based Chatbot**
- 🚀 **Retrieves relevant answers** from a pre-defined **FAQ dataset**.
- 📡 **Uses FAISS for vector search** to fetch the most relevant FAQ.
- 🤖 **Generates responses** via an LLM when no FAQ matches the query.

### **2️⃣ Sentiment Analysis & Feedback Collection**
- 🧠 **Detects user sentiment** (Positive, Negative, Neutral).
- 📊 **Stores insights in MongoDB for continuous learning**.

### **3️⃣ Real-Time Analytics Dashboard**
- 📈 **Tracks chatbot usage & sentiment trends over time**.
- 🎨 **Provides interactive charts & insights**.
- 🔄 **Helps optimize responses and improve user experience**.

---


## 🔥 **Project Features**
-  **AI-Powered Chatbot** for **instant support**.
-  **Multilingual Support** – Detects and responds in **any language**.
-  **Sentiment Analysis & Feedback**.
-  **FAISS Vector Search** for **fast FAQ retrieval**.
-  **MongoDB Integration** for **chat storage**.
-  **Interactive Streamlit UI** for **chatbot & analytics**.

---

## 📊 **AI-Powered Chatbot**
- ✅ Uses **FAISS for fast FAQ retrieval**.
- ✅ **Supports multiple languages**.
- ✅ **Sentiment detection** on responses.
- ✅ **Interactive UI with feedback buttons**.

---

## 📈 **Analytics Dashboard**
- ✅ **Sentiment Distribution** (Positive, Negative, Neutral).
- ✅ **Trends Over Time** – Tracks **chatbot usage patterns**.
- ✅ **Engagement Heatmap** – Shows peak chatbot usage hours.
- ✅ **Top FAQs** – Identifies **most asked questions**.

---

## 📦 **Database & API Integration**

✅ **MongoDB** – Stores chat history & user feedback  
✅ **FAISS** – Fast FAQ search & retrieval  

---

## 🔁 **LLM Functionality**

### **1️⃣ Understanding User Input**
🔹 Detects **language** and **query intent**

### **2️⃣ Classifying Questions**
🔹 Determines **category (e.g., Loans, Security, Payments, etc.)**

### **3️⃣ Retrieving Answers**
🔹 Searches **FAISS database** for relevant FAQ answers
🔹 If no match is found, generates a response **using LLM**

### **4️⃣ Sentiment Analysis & Storage**
🔹 Predicts **user sentiment** (Positive, Negative, Neutral)
🔹 Stores **chat history & feedback** in MongoDB

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
📂 AI-Powered-Customer-Support-System/
│
├── 📂 backend/               # Backend Logic & Core Processing
│   ├── chatbot.py           # Core Chatbot Logic (LLM, FAQ, Sentiment, Category)
│   ├── faq_loader.py        # Loads FAQ Data into MongoDB
│   ├── vector_db.py         # ChromaDB for FAQ Embeddings
│
├── 📂 frontend/              # Streamlit UI Components
│   ├── chatbot_ui.py        # Chatbot Interface
│   ├── chatbot_analytics.py # Analytics Dashboard UI
│
├── 📂 api/          # API Services
│   ├── api.py    
│
├── 📂 frontend/streamlit/    # Streamlit UI Components
│   ├── chatbot_analytics.py # Analytics Dashboard UI
│
├── 📂 sample/                # Sample Data & Scripts
│   ├── db_populate.py       # Script to populate MongoDB           
│
├── 📂 FAQs/                  # FAQ Dataset
│   ├── BankFAQs.csv         # FAQ Dataset (Raw)
│   ├── processed_faqs.json  # Preprocessed FAQ Data (Optional)
│
├── 📂 chroma_db/             # Persistent ChromaDB Storage
│   ├── chromadb_index/      # Vector Store for FAQ Retrieval
│
├── 📂 logs/                  # Logging & Monitoring
│   ├── chatbot.log          # Logs for Chatbot Responses
│
├── 📂 config/                # Configuration Files
│   ├── config.py            # Global Configuration
│   ├── streaming.py         # Streamlit Configuration
│   ├── utils.py             # Utility Functions
│
├── 📂 deployment/             # Deployment Configurations
│   ├── Dockerfile            # Docker Configuration

├── .env                      # Environment Variables (API Keys, DB Config)
├── requirements.txt          # Python Dependencies for Streamlit & Backend
├── main.py                   # Entry Point for Streamlit App
├── README.md                 # Project Overview & Instructions
├── vercel.json               # Deployment Config for Vercel (Optional)
├── Dockerfile                # Docker Deployment Config (Optional)
└── .gitignore                # Ignore Unnecessary Files
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

## **🖥️ Running the FastAPI Server**
Once the model is trained and registered, run **FastAPI** to serve real-time predictions:

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```
This starts the FastAPI server on **http://127.0.0.1:8000**.

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

## 🛠️ **Future Improvements**
- 🔹 **User Sessions** → Recognize returning users.
- 🔹 **Advanced LLM Fine-Tuning** → Improve chatbot responses.
- 🔹 **Voice Interaction** → Convert text-based chatbot into a **voice assistant**.
- 🔹 **Voice-Enabled Chatbot** – Integrate **speech recognition** for voice queries.  
- 🔹 **WhatsApp & Telegram Integration** – Expand support to messaging apps.  
- 🔹 **Advanced Sentiment Analysis** – Use transformer models for better predictions.  
- 🔹 **Proactive Support Suggestions** – Predict user needs based on chat history.  


---

## 📌 **Conclusion**
The **AI-Powered Customer Support System** provides **seamless, intelligent customer interactions** through **FAQ retrieval, sentiment analysis, and analytics**. With **scalable deployment** and **real-time insights**, this project can revolutionize **customer engagement** across multiple industries. 🚀

🔹 **Want to contribute?** Fork the repo and submit a PR! 🎉