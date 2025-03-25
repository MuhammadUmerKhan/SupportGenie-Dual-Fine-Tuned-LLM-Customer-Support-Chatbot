# Backend Documentation

The **backend** folder contains all the core processing logic for the AI-Powered Customer Support System. This section provides a detailed explanation of each file and its role in the system.

## 📂 backend/ (Core Processing & AI Logic)
This folder manages the **chatbot's logic, data storage, and retrieval mechanisms**.

### 1️⃣ **chatbot.py** (🧠 Core Chatbot Logic)
**Purpose:**
- The main processing engine for handling user queries.
- Uses **LLMs (Large Language Models)** to generate responses when no FAQ match is found.
- Implements **sentiment analysis** and **question categorization**.

**Key Features:**
- **FAQ-Based Answer Retrieval:** Queries FAISS DB for relevant FAQs.
- **LLM-Generated Responses:** Uses AI to generate answers when an FAQ match is unavailable.
- **Sentiment Analysis:** Classifies user feedback as Positive, Negative, or Neutral.
- **Category Classification:** Assigns each question to predefined categories (e.g., security, loans, accounts).
- **Multilingual Support 🌍:** Detects the input language and translates responses accordingly.
- **Chat History Storage:** Stores chat interactions in MongoDB for analytics.

---

### 2️⃣ **faq_loader.py** (📂 FAQ Data Management)
**Purpose:**
- Loads FAQ data into MongoDB for easy retrieval.
- Preprocesses the FAQ dataset to ensure optimized search queries.

**Key Features:**
- **Data Preprocessing:** Cleans and structures the FAQ dataset before insertion.
- **MongoDB Storage:** Inserts processed FAQs into the database.
- **Duplicate Handling:** Ensures no duplicate FAQ entries exist in the database.
- **Logging & Error Handling:** Provides logs for debugging and efficient data management.

---

### 3️⃣ **vector_db.py** (🔍 FAISS DB for FAQ Retrieval)
**Purpose:**
- Stores and retrieves **vector embeddings** of FAQ questions for fast and accurate matching.
- Uses **FAISS DB** as a **vector database** for semantic search.

**Key Features:**
- **Embeddings Generation:** Converts FAQ questions into numerical representations using **Hugging Face Embeddings**.
- **Efficient Search Mechanism:** Retrieves the most relevant FAQ based on similarity scores.
- **FAISS DB Integration:** Provides a **persistent** and **scalable** storage solution for embeddings.
- **Automatic Updates:** Ensures the database stays updated when new FAQs are added.

---

## 🌟 Summary
The **backend** folder powers the AI chatbot, ensuring **fast and accurate responses** using **FAQs, AI models, and database storage**.
- **`chatbot.py` →** Manages AI logic, sentiment analysis, and response generation.
- **`faq_loader.py` →** Loads and processes FAQ data into MongoDB.
- **`vector_db.py` →** Handles FAQ embeddings and semantic search using FAISS DB.

Together, these components create a **scalable and intelligent** AI customer support system. 🚀

