# Configuration Documentation

The **config** folder contains essential configuration and utility files that streamline the **AI-Powered Customer Support System**. These files handle **global settings, utility functions, and Streamlit configurations**.

## 📂 config/ (Project Configuration & Utility Functions)
This folder ensures smooth project execution by managing **API keys, database connections, and common utility functions**.

### 1️⃣ **config.py** (⚙️ Global Configuration)
**Purpose:**
- Stores global **configuration settings** for the entire project.
- Manages environment variables and API keys securely.

**Key Features:**
- **Database Configuration:** Stores **MongoDB connection URI**.
- **AI Model Settings:** Defines the **LLM model name and API key**.
- **File Paths:** Manages paths for FAQ datasets and logs.
- **Security Best Practices:** Loads credentials securely using **dotenv**.

---

### 2️⃣ **streaming.py** (📺 Streamlit Configuration)
**Purpose:**
- Handles **Streamlit settings and optimizations** for a smooth UI experience.

**Key Features:**
- **Session State Management:** Ensures session persistence across UI interactions.
- **Caching Optimization:** Uses `st.cache_resource` to optimize chatbot performance.
- **UI Customization:** Manages **default colors, font sizes, and layout settings**.
- **Error Handling:** Prevents crashes with well-defined exceptions and logs.

---

### 3️⃣ **utils.py** (🔧 Utility Functions)
**Purpose:**
- Provides **helper functions** that are used across multiple modules to improve efficiency.

**Key Features:**
- **Chat History Management:** Loads and stores previous chatbot interactions.
- **Language Processing:** Detects user input language and translates queries (for multilingual support 🌍).
- **Vector Embedding Handling:** Helps in processing **FAQ embeddings** for ChromaDB.
- **Logging & Debugging Tools:** Simplifies tracking errors and debugging processes.

---

## 📂 FAQs/ (FAQ Dataset Management)
This folder contains **FAQ datasets** used for chatbot responses.

### 1️⃣ **BankFAQs.csv** (📄 Raw FAQ Dataset)
**Purpose:**
- Contains **raw FAQ data** in CSV format.
- Used as the primary source for chatbot responses.

**Key Features:**
- **Structured Data:** Includes **questions and answers**.
- **Easy Import:** Readable and importable for processing.

---

### 2️⃣ **processed_faqs.json** (📂 Preprocessed FAQ Data)
**Purpose:**
- Stores **preprocessed FAQs** optimized for quick retrieval.
- Used for **faster access** compared to raw CSV data.

**Key Features:**
- **Optimized Structure:** Data is stored in **JSON format**.
- **Reduced Processing Time:** Eliminates redundant preprocessing.

---

## 🌟 Summary

The **FAQs** folder manages **customer queries and chatbot knowledge base**.
- **`BankFAQs.csv` →** Stores raw FAQ data.
- **`processed_faqs.json` →** Stores optimized FAQ responses for faster chatbot processing.

These components **enhance the efficiency, security, and maintainability** of the AI-powered chatbot system. 🚀

