# 📌 **AI-Powered Customer Support System**

## 🚀 **Project Overview**
The **AI-Powered Customer Support System** is designed to provide **automated customer assistance** by leveraging **Large Language Models (LLMs), sentiment analysis, and an FAQ-based knowledge base**. This system aims to improve customer interactions by providing **quick, accurate, and relevant responses** while also offering **analytical insights** into user behavior and common queries.

### 💡 **Why This Project?**
- ✅ **AI-Powered FAQ Retrieval**: Uses **ChromaDB** to fetch answers from FAQs.
- ✅ **Sentiment Analysis**: Classifies customer feedback as **Positive, Negative, or Neutral**.
- ✅ **Multilingual Support**: Detects and translates queries into English before processing.
- ✅ **User Interaction Insights**: Tracks **most asked questions, sentiment trends, and user behavior**.
- ✅ **Feedback Collection**: Allows users to rate responses as **Helpful 👍 or Not Helpful 👎**.

---

## 📁 **Project Directory Structure**
```
AI-Powered-Customer-Support-System/
│
├── 📂 backend/               # Backend Logic & Core Processing
│   ├── chatbot.py           # Core Chatbot Logic (LLM, FAQ Retrieval, Sentiment Analysis)
│   ├── faq_loader.py        # Loads FAQ Data into MongoDB
│   ├── vector_db.py         # ChromaDB for FAQ Embeddings
│
├── 📂 frontend/              # Streamlit UI Components
│   ├── chatbot_ui.py        # Chatbot Interface
│   ├── chatbot_analytics.py # Analytics Dashboard UI
│   ├── faqs_ui.py           # FAQ Display Page
│
├── 📂 FAQs/                  # FAQ Dataset
│   ├── BankFAQs.csv         # Raw FAQ Dataset
│   ├── processed_faqs.json  # Preprocessed FAQ Data
│
├── 📂 chroma_db/             # ChromaDB Storage for FAQ Retrieval
│
├── 📂 api/                   # API Services
│   ├── api.py               # FastAPI Backend (Optional)
│
├── 📂 logs/                  # Logging & Monitoring
│   ├── chatbot.log          # Logs for Chatbot Responses
│
├── 📂 config/                # Configuration & Utilities
│   ├── config.py            # Global Configurations (API Keys, DB Configs)
│   ├── streaming.py         # Streamlit UI Configuration
│   ├── utils.py             # Utility Functions (Chat History, Translations, etc.)
│
├── .env                      # Environment Variables (MongoDB URI, API Keys)
├── requirements.txt          # Python Dependencies
├── main.py                   # Entry Point for Streamlit App
├── README.md                 # Project Documentation
└── .gitignore                # Ignore Unnecessary Files
```

---

## 🛠️ **Setup & Installation**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/AI-Powered-Customer-Support-System.git
cd AI-Powered-Customer-Support-System
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Setup Environment Variables**
- Create a **.env** file in the root directory.
- Add the following details:
```ini
MONGO_URI=your_mongodb_connection_string
GROK_API_KEY=your_groq_api_key
MODEL_NAME=qwen-2.5-32b
FAQ_PATH=./FAQs/BankFAQs.csv
```

### **4️⃣ Populate the Database**
Run the following command to insert the FAQs into MongoDB:
```bash
python backend/faq_loader.py
```

### **5️⃣ Run the Chatbot Application**
```bash
streamlit run main.py
```

---

## 📊 **Features & Functionality**

### **1️⃣ AI Chatbot 💬**
- Retrieves answers from FAQs using **ChromaDB**.
- If no match is found, generates AI responses via **LLM**.
- Detects and translates non-English queries before processing.
- **Collects user feedback** for response improvement.

### **2️⃣ Analytics Dashboard 📊**
- **Sentiment Analysis**: Displays trends in positive/negative interactions.
- **Most Asked Questions**: Shows common user queries.
- **User Engagement**: Heatmaps show peak activity times.
- **Sentiment by Year**: Tracks sentiment shifts over months.

### **3️⃣ FAQs Page 📄**
- Displays a list of **all available FAQs** from the dataset.
- Allows **searching and filtering FAQs** for easier access.

---

## 🚀 **Deployment**

### **1️⃣ Deploy on Streamlit Cloud**
- Upload the repository to **GitHub**.
- Go to **Streamlit Cloud**.
- Connect your repository and deploy the `main.py` file.

### **2️⃣ Docker Deployment (Optional)**
- Build the Docker image:
```bash
docker build -t ai-customer-support .
```
- Run the container:
```bash
docker run -p 8501:8501 ai-customer-support
```

---

## 📌 **Future Enhancements**

- ✅ **User Sessions** – Recognize returning users for a personalized experience.
- ✅ **Voice Input** – Enable users to **speak queries** instead of typing.
- ✅ **Smart Suggestions** – Predict user queries before they finish typing.
- ✅ **API Integration** – Connect chatbot with customer support ticketing systems.

---

## 🏆 **Final Thoughts**
This project demonstrates how **AI can enhance customer support** by **automating responses, tracking sentiment trends, and improving over time**. By integrating **FAQ retrieval, LLM-generated responses, and analytics**, this system provides **a complete AI-powered support solution**. 🚀