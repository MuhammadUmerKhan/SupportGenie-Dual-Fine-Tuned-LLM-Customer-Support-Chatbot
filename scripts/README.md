# AI-Powered Customer Support Chatbot 🚀

Welcome to the **AI-Powered Customer Support Chatbot** project! This system leverages **MongoDB**, **FAISS**, and **large language models (LLM)** to provide efficient and friendly customer support. It loads FAQs from a CSV file, stores them in a MongoDB database, creates vector embeddings for fast FAQ retrieval, and delivers user-friendly responses via a Streamlit interface. Below is a brief overview of each script and its role in the system. 🌟

## 📂 Project Structure and Scripts

### 1️⃣ `faq_loader.py` 📄
**Purpose**: Loads FAQs from a CSV file into a MongoDB database for efficient storage and retrieval.  
**Key Functions**:
- **Connect to MongoDB** (`connect_mongo`): Establishes a connection to MongoDB Atlas using credentials from `configs.py`. 🔗
- **Load FAQs** (`load_faq_to_mongo`): Reads a CSV file (`BankFAQs.csv`) containing FAQs, processes it (removing `<think>` tags if present), and stores the data in MongoDB. 📚
- **Features**:
  - Reads CSV with columns: `Question`, `Answer`, and `Class` (category).
  - Clears existing FAQs in the database before loading new ones.
  - Logs success or errors using `loggers.py`. ✅❌
- **Usage**: Run `python faq_loader.py` to populate the MongoDB `faqs` collection.

### 2️⃣ `configs.py` ⚙️
**Purpose**: Centralizes configuration settings for the project, including API keys, database credentials, and file paths.  
**Key Functions**:
- **Load Environment Variables**: Uses `dotenv` to securely load settings from a `.env` file. 🔒
- **Key Variables**:
  - `GROK_API_KEY`: API key for the Grok LLM.
  - `MODEL_NAME`: Specifies the LLM model (`qwen-2.5-32b`).
  - `FAQ_PATH`: Path to the `BankFAQs.csv` file.
  - `MONGO_URI`: MongoDB connection string with encoded credentials.
  - `DB_NAME`, `FAQ_COLLECTION`, `CHAT_HISTORY_COLLECTION`: MongoDB database and collection names.
- **Features**:
  - Validates environment variables and logs warnings if missing. ⚠️
  - URL-encodes MongoDB password for secure connection. 🔐
- **Usage**: Imported by other scripts to access global settings.

### 3️⃣ `chatbot.py` 🤖
**Purpose**: Core script for processing user queries, retrieving FAQ answers, and generating LLM responses.  
**Key Functions**:
- **Connect to MongoDB** (`connect_mongo`): Establishes MongoDB connection for chat history storage. 🔗
- **Initialize LLM** (`initialize_chatgroq`): Sets up the Grok LLM via `utilss.py`. 🧠
- **Search FAQs** (`search_faq`): Uses FAISS to find the most relevant FAQ based on user input embeddings. 🔍
- **Sentiment Analysis** (`analyze_sentiment`): Analyzes user input sentiment using `TextBlob` (Positive, Negative, Neutral). 😊😔
- **Classify Input** (`is_customer_review`, `classify_question_category`): Determines if input is a review or inquiry and categorizes it (e.g., `security`, `loans`). 🏷️
- **Store Chat History** (`store_chat_history`): Saves user queries and bot responses in MongoDB with metadata (sentiment, category, timestamp). 📝
- **Process Queries** (`get_chatbot_response`): Matches user input to FAQs via FAISS; if no match, uses LLM to generate a response. Rephrases FAQ answers in a friendly, bullet-pointed format with emojis. 😊
- **Main Loop** (`main`): Runs an interactive chatbot loop for user input. 💬
- **Features**:
  - Handles negative feedback (e.g., "service is bad") with sentiment analysis.
  - Stores interactions in MongoDB for analytics.
  - Logs errors and interactions for debugging.
- **Usage**: Run `python chatbot.py` to test with inputs like `"service is bad my issue didn't solve"`.

### 4️⃣ `vector_db.py` 📊
**Purpose**: Manages the FAISS vector database for fast FAQ retrieval using embeddings.  
**Key Functions**:
- **Ensure Directory** (`ensure_faiss_db_directory`): Creates the FAISS database directory if it doesn’t exist. 📁
- **Connect to MongoDB** (`connect_mongo`): Connects to MongoDB to retrieve FAQs. 🔗
- **Load FAQs** (`load_faqs_from_mongo`): Fetches FAQs from the MongoDB `faqs` collection. 📚
- **Reset FAISS** (`reset_faiss_db`): Clears the FAISS index and metadata for a fresh start. 🧹
- **Store FAQs in FAISS** (`store_faqs_in_faiss`): Generates embeddings for FAQ questions using HuggingFace’s `all-mpnet-base-v2` model and stores them in FAISS. 🧮
- **Features**:
  - Uses 768-dimensional embeddings for efficient FAQ matching.
  - Saves FAISS index (`faiss_index.bin`) and metadata (`faiss_metadata.json`).
  - Logs success or errors for debugging. ✅❌
- **Usage**: Run `python vector_db.py` to populate the FAISS database after loading FAQs into MongoDB.

### 5️⃣ `streaming.py` 📺
**Purpose**: Provides real-time streaming of LLM responses in the Streamlit UI.  
**Key Functions**:
- **StreamHandler Class**:
  - Initializes a Streamlit container for displaying text. 🖥️
  - Updates the UI in real-time as new LLM tokens are generated (`on_llm_new_token`). 📈
- **Features**:
  - Enhances user experience with live response updates.
  - Integrates with LangChain for streaming LLM output.
- **Usage**: Used by the Streamlit app to display chatbot responses dynamically.

### 6️⃣ `utilss.py` 🔧
**Purpose**: Contains utility functions for chat history, LLM responses, and embeddings.  
**Key Functions**:
- **Chat History Decorator** (`enable_chat_history`): Persists chat messages across Streamlit sessions and displays them in the UI. 💬
- **Remove Think Tags** (`remove_think_tags`): Strips `<think>` tags from LLM responses using regex. ✂️
- **Cached LLM Response** (`get_cached_llm_response`): Caches Grok LLM responses to reduce API calls, using the `qwen/qwen3-32b` model. 🧠
- **Display Messages** (`display_msg`): Appends and displays chat messages in the Streamlit UI. 📝
- **Log Q&A** (`print_qa`): Logs user questions and bot answers for debugging. 📋
- **Configure Embeddings** (`configure_vector_embeddings`): Loads HuggingFace embeddings for FAQ matching. 🧮
- **Sync Session State** (`sync_st_session`): Synchronizes Streamlit session state for consistency. 🔄
- **Features**:
  - Optimizes performance with caching (`st.cache_resource`).
  - Supports Streamlit’s interactive UI.
  - Logs interactions for debugging.
- **Usage**: Imported by `chatbot.py` and `vector_db.py` for shared functionality.

## 🌟 Project Workflow
1. **Load FAQs** (`faq_loader.py`): Reads `BankFAQs.csv` and stores FAQs in MongoDB. 📄➡️📚
2. **Build FAISS Index** (`vector_db.py`): Generates embeddings for FAQs and stores them in FAISS for fast retrieval. 🧮
3. **Process Queries** (`chatbot.py`): Matches user input to FAQs using FAISS; if no match, generates an LLM response in a friendly, bullet-pointed format. 🤖😊
4. **Store Interactions** (`chatbot.py`): Saves user queries and responses in MongoDB with sentiment and category metadata. 📝
5. **Stream Responses** (`streaming.py`): Displays LLM responses in real-time in the Streamlit UI. 📺
6. **Utilities** (`utilss.py`): Provides helper functions for embeddings, chat history, and response processing. 🔧

## 📋 Requirements
- **Python Libraries**: `pymongo`, `python-dotenv`, `numpy`, `faiss-cpu`, `textblob`, `langchain-huggingface`, `langchain-groq`, `streamlit`
- **Environment Variables** (in `.env`):
  - `GROK_API_KEY`: For Grok LLM access.
  - `MONGO_USER`, `MONGO_PASSWORD`, `MONGO_CLUSTER`: For MongoDB Atlas.
  - `MONGO_DB`, `FAQ_PATH`: Database and CSV file settings.
- **MongoDB Atlas**: A configured cluster with `faqs` and `chat_history` collections.
- **FAISS Files**: `faiss_index.bin` and `faiss_metadata.json` for FAQ retrieval.

## 🚀 Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Set up `.env` with MongoDB and Grok API credentials.
3. Load FAQs: `python faq_loader.py`
4. Build FAISS index: `python vector_db.py`
5. Run the chatbot: `python chatbot.py` or launch the Streamlit app: `streamlit run app.py`
6. Test with queries like `"service is bad my issue didn't solve"`. 😊

## 🔍 Debugging Tips
- Check logs in `./logs/chatbot.log` for errors. 📋
- Verify MongoDB connection and FAISS file paths. 🔗
- Ensure `BankFAQs.csv` exists and is formatted correctly (`Question`, `Answer`, `Class`). 📄
- If SSL issues occur, install `certifi` and update `connect_mongo` with `tls=True` and `tlsCAFile=certifi.where()`. 🔐

## 📈 Future Enhancements
- Add multilingual support for non-English queries. 🌍
- Enhance sentiment analysis with advanced NLP models. 😊
- Integrate more LLM models for flexibility. 🧠

This project is a robust foundation for an AI-powered customer support system, combining efficient FAQ retrieval with friendly, dynamic responses. Let’s make customer support awesome! 🎉