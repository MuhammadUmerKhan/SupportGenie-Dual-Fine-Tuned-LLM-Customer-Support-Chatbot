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

## 🌟 Summary
The **config** folder acts as the **backbone of the project**, handling **global settings, UI configurations, and utility functions**.
- **`config.py` →** Stores global settings like API keys and database URIs.
- **`streaming.py` →** Manages Streamlit-specific UI settings and caching.
- **`utils.py` →** Provides reusable utility functions for chat history, translations, and debugging.

These components **enhance the efficiency, security, and maintainability** of the AI-powered chatbot system. 🚀