from datetime import datetime
import pymongo, dotenv, os, sys, numpy as np, json, faiss
from textblob import TextBlob
import scripts.config as CONFIG
import scripts.utils as utils
from scripts.logger import get_logger

dotenv.load_dotenv()
logger = get_logger(__name__)

CATEGORIES = ['security', 'loans', 'accounts', 'insurance', 'investments', 'fundstransfer', 'cards']

# Define FAISS storage paths
FAISS_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "faiss_db"))
FAISS_INDEX_FILE = os.path.join(FAISS_DB_PATH, "faiss_index.bin")
FAISS_METADATA_FILE = os.path.join(FAISS_DB_PATH, "faiss_metadata.json")

def connect_mongo():
    """Connect to MongoDB and verify connection."""
    try:
        client = pymongo.MongoClient(CONFIG.MONGO_URI)
        db = client[CONFIG.DB_NAME]

        # ✅ Debugging Step: Check Connection
        if db is None:
            raise Exception("Database object is None.")
        
        # ✅ Check if the collection exists
        if CONFIG.CHAT_HISTORY_COLLECTION not in db.list_collection_names():
            print(f"❌ Collection '{CONFIG.CHAT_HISTORY_COLLECTION}' does not exist in MongoDB!")
        
        logger.info("✅ MongoDB connection established successfully.")
        return db
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return None

def initialize_chatgroq():
    """Initialize ChatGroq model."""
    try:
        logger.info("✅ ChatGroq initialized successfully.")
        return utils.get_cached_llm_response
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        return None

def search_faq(user_input):
    """Search for the most relevant FAQ using FAISS."""
    try:
        # Ensure FAISS index and metadata exist before loading
        if not os.path.exists(FAISS_INDEX_FILE) or not os.path.exists(FAISS_METADATA_FILE):
            logger.error("FAISS index or metadata file not found.")
            return None

        # Load FAISS index
        index = faiss.read_index(FAISS_INDEX_FILE)
        
        # Load metadata
        with open(FAISS_METADATA_FILE, "r") as f:
            answers = json.load(f)  # Ensure this is a list of FAQs

        # Convert metadata keys to a list (for indexing)
        faq_questions = list(answers.keys()) if isinstance(answers, dict) else list(answers)

        # Generate embedding for user query
        embedding_model = utils.configure_vector_embeddings()
        user_embedding = embedding_model.embed_query(user_input)
        user_embedding = np.array(user_embedding).astype('float32').reshape(1, -1)

        # Perform FAISS search
        distances, indices = index.search(user_embedding, 1)

        # Handle case where no valid match is found
        if indices[0][0] == -1 or indices[0][0] >= len(faq_questions):
            logger.info("No relevant FAQ match found.")
            return None

        # Retrieve matched question and its answer
        question_matched = faq_questions[indices[0][0]]
        answer = answers.get(question_matched, "No answer found.") if isinstance(answers, dict) else answers[indices[0][0]]
        
        logger.info("✅ Answer Loaded Successfully")
        return answer

    except Exception as e:
        logger.error(f"❌ Error searching FAQs: {e}")
        return None

def analyze_sentiment(text):
    """Perform sentiment analysis."""
    score = TextBlob(text).sentiment.polarity
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"

def is_customer_review(user_input):
    """Use LLM to determine if user input is a review or general inquiry."""
    prompt = f"""
    Classify the following user input as either a 'Review' or 'General Inquiry'.
    A review is an opinion, complaint, or feedback about service quality.
    A general inquiry is a request for information without an opinion.

    User Input: "{user_input}"

    Respond with ONLY 'Review' or 'General Inquiry'.
    """
    
    response = utils.get_cached_llm_response(prompt)
    return response == "Review"

def classify_question_category(user_input):
    """Use LLM to classify the user question into a predefined category."""
    prompt = f"""
    Classify the following user question into one of these categories:
    {CATEGORIES}

    If the question does not fit into any category, classify it as 'other'.

    User Question: "{user_input}"

    Respond with ONLY the category name.
    """
    
    response = utils.get_cached_llm_response(prompt)
    
    return response if response in CATEGORIES else "other"

def store_chat_history(user_input, bot_reply):
    """Store chat history in MongoDB with explicit None check."""
    try:
        db = connect_mongo()
        if db is None:  # ✅ Explicitly check if db is None
            raise Exception("❌ Database connection failed.")

        chat_collection = db[CONFIG.CHAT_HISTORY_COLLECTION]

        # ✅ Ensure Collection is Retrieved
        if chat_collection is None:
            raise Exception(f"❌ Collection '{CONFIG.CHAT_HISTORY_COLLECTION}' not found!")

        # ✅ Use LLM to determine if this is a review
        is_review = is_customer_review(user_input)
        sentiment = analyze_sentiment(user_input) if is_review else "Not a Review"
        category = classify_question_category(user_input)
        chat_data = {
            "user": user_input,
            "category": category,
            "bot": bot_reply,
            "sentiment": sentiment,
            "timestamp": datetime.utcnow()
        }

        # ✅ Log chat data before insertion
        print(f"🔄 Storing chat: {chat_data}")

        # ✅ Insert data
        chat_collection.insert_one(chat_data)
        logger.info(f"✅ Chat stored: {chat_data}")

    except Exception as e:
        logger.error(f"❌ Error storing chat history: {e}")
        print(f"❌ Error storing chat history: {e}")

def get_chatbot_response(user_input):
    """Process user input in English."""
    try:
        print(f"🔍 Processing User Input: {user_input}")

        # Step 1: Check FAISS for FAQ answer
        faq_answer = search_faq(user_input)

        # Step 2: Use Cached LLM Response
        llm = initialize_chatgroq()
        if llm is None:
            print("❌ ERROR: LLM initialization failed!")
            return "Sorry, the AI model is unavailable."

        if faq_answer:
            # 🔹 Use Cached LLM to rephrase & enhance the FAQ response
            prompt = f"""
            You are a helpful AI assistant providing customer support. 😊
            A user asked: "{user_input}"
            We found the following FAQ answer:
            "{faq_answer}"
            
            Rephrase this answer in a friendly, engaging, and natural way. Simplify the response into bullet points and use emojis to make it welcoming. If additional relevant information can be inferred, include it. Avoid adding extra text outside the bullet points.
            """
            bot_reply = utils.get_cached_llm_response(prompt)  # ✅ Use cached response
        else:
            # 🔹 If no FAQ match, use LLM to generate an answer
            bot_reply = utils.get_cached_llm_response(user_input)  # ✅ Use cached response

        print(f"📝 AI Response: {bot_reply}")  # Debugging log

        # Step 3: Store chat history
        store_chat_history(user_input, bot_reply)

        return bot_reply

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        print(f"❌ ERROR: {e}")
        return "Sorry, an error occurred."

def main():
    """Main function to run chatbot."""
    llm = initialize_chatgroq()
    if not llm:
        print("Failed to initialize AI model. Exiting.")
        return

    while True:
        user_query = input("\nAsk me anything: ")
        response = get_chatbot_response(user_query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    print(get_chatbot_response("service is bad my issue did'nt solve"))