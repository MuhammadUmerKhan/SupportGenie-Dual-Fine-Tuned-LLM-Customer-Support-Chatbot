from datetime import datetime
import pymongo, chromadb, dotenv, os, sys
from textblob import TextBlob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config.config as CONFIG
import config.utils as utils
from logger import get_logger

dotenv.load_dotenv()
logger = get_logger(__name__)

CATEGORIES = ['security', 'loans', 'accounts', 'insurance', 'investments', 'fundstransfer', 'cards']

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
        
        return db
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return None

def initialize_chatgroq():
    """Initialize ChatGroq model."""
    try:
        return utils.get_cached_llm_response
        # return utils.configure_llm()
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        return None

def search_faq(user_input):
    """Search for the most relevant FAQ using ChromaDB."""
    try:
        chroma_client = chromadb.PersistentClient(path="./chroma_db", settings={"chroma_db_impl": "duckdb"})
        collection = chroma_client.get_or_create_collection("faq_embeddings")
        
        embedding_model = utils.configure_vector_embeddings()
        user_embedding = embedding_model.embed_query(user_input)

        # ✅ Return only 1 result for faster search
        results = collection.query(query_embeddings=[user_embedding], n_results=1)

        if results["ids"] and results["metadatas"][0]:
            return results["metadatas"][0][0]["answer"]
        return None

    except Exception as e:
        logger.error(f"Error searching FAQs: {e}")
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

# Predefined categories from the dataset

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

def detect_language_and_translate(user_input):
    """Detect the input language and translate it into English if necessary."""
    prompt = f"""
    You are a language assistant. Identify the language of the given text and translate it into English if needed.

    User Input: "{user_input}"

    Respond in this format:
    Language: <Detected Language>
    Translated: <English Translation>
    """

    response = utils.get_cached_llm_response(prompt)

    # Extract language & translation
    lines = response.split("\n")
    detected_language = lines[0].replace("Language:", "").strip()
    translated_text = lines[1].replace("Translated:", "").strip()

    return detected_language, translated_text

def translate_back_to_original(text, original_language="English"):
    """Translate chatbot response back to the original language."""
    if original_language.lower() == "english":
        return text  # No translation needed

    prompt = f"""
    Translate the following English text into {original_language}.

    English: "{text}"

    Translated:
    """

    return utils.get_cached_llm_response(prompt)

def get_chatbot_response(user_input):
    """Process user input with multilingual support."""
    try:
        print(f"🔍 Processing User Input: {user_input}")

        # Detect language & translate if needed
        original_language, translated_input = detect_language_and_translate(user_input)

        # Step 1: Check ChromaDB for FAQ answer
        faq_answer = search_faq(translated_input)

        # Step 2: Use Cached LLM Response
        llm = initialize_chatgroq()
        if llm is None:
            print("❌ ERROR: LLM initialization failed!")
            return "Sorry, the AI model is unavailable."

        if faq_answer:
            # 🔹 Use Cached LLM to **rephrase & enhance** the FAQ response
            prompt = f"""
            You are an AI assistant providing customer support.
            A user asked: "{translated_input}"
            We found the following FAQ answer:
            "{faq_answer}"
            
            Rephrase this answer in a more natural, engaging, and helpful way without including extra text. If additional relevant information can be inferred, include it.
            """
            bot_reply = utils.get_cached_llm_response(prompt)  # ✅ Use cached response

        else:
            # 🔹 If no FAQ match, use LLM to generate an answer
            bot_reply = utils.get_cached_llm_response(translated_input)  # ✅ Use cached response

        # Translate response back to original language
        final_response = translate_back_to_original(bot_reply, original_language)

        print(f"📝 AI Response: {final_response}")  # Debugging log

        # Step 3: Store chat history
        store_chat_history(user_input, bot_reply)

        return final_response

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
    main()
    # print(get_chatbot_response("service is bad my issue did'nt solve"))
    # print(get_chatbot_response("le service est très mauvais, mon problème n'a pas été résolu")) # service is very bad, my issue did'nt resolve
    # print(store_chat_history("Thanks my issue reolved", "great"))
    # print(initialize_chatgroq())
    # print(detect_language_and_translate("comment vas-tu?"))
    # print(translate_back_to_original("comment vas-tu?"))