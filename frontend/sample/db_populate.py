import pymongo, os, random, sys
from dotenv import load_dotenv
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config.utils as utils
import config.config as CONFIG

MONGO_URI = CONFIG.MONGO_URI

# Connect to MongoDB Atlas
client = pymongo.MongoClient(MONGO_URI)
db = client[CONFIG.DB_NAME]  # Ensure this matches your database name
collection = db[CONFIG.CHAT_HISTORY_COLLECTION]  # Ensure this matches your collection name

# Sample Categories and Sentiments
CATEGORIES = ['security', 'loans', 'accounts', 'insurance', 'investments', 'fundstransfer', 'cards']
sentiments = ["Positive", "Negative", "Neutral", "Not a Review"]
users = [
    "hi", "hello", "how can I reset my password?", "I need a loan", "worst service ever", "thank you for your help",
    "How do I transfer funds?", "Can I apply for a credit card?", "I lost my card, what should I do?",
    "The service is amazing!", "Where can I check my account balance?", "My payment got stuck, help!", 
    "Tell me about investment plans", "I'm not happy with the customer support", "Great assistance!",
    "What is my account limit?", "How do I change my registered phone number?", "Explain the insurance policies",
    "Where can I report fraud?", "Why was my transaction declined?", "Help me with my EMI payment",
    "I want to update my address", "Is my account safe?", "How do I activate my debit card?",
    "My OTP is not working", "I got charged twice for a transaction", "Can I cancel my loan?",
    "Is there a fee for fund transfers?", "How do I check my credit score?", "Can I open a joint account?",
    "Can I use my card abroad?", "What are the terms for overdraft?", "My balance is incorrect",
    "How do I reset my PIN?", "I need to dispute a transaction", "Why was my check rejected?",
    "Can I increase my credit limit?", "The app is not working", "How do I block my card?",
    "What is the minimum balance requirement?", "How do I download my statement?", "How do I close my account?",
    "I love the services!", "I need help with my UPI payment", "My refund is taking too long",
    "Are there any hidden charges?", "How do I change my email ID?", "How do I get a home loan?",
    "I forgot my net banking password", "I am unable to make payments", "Your customer service is bad"
]

# Generate 50 records with random years (2020 - 2025)
chat_data = []

for _ in range(50):
    user_input = random.choice(users)
    category = random.choice(CATEGORIES)
    bot_reply = f"Response to '{user_input}'"
    sentiment = random.choice(sentiments)
    
    # ✅ Randomize Year, Month, and Day
    random_year = random.randint(2020, 2025)
    random_month = random.randint(1, 12)
    random_day = random.randint(1, 28)  # Keep within 28 to avoid invalid dates

    timestamp = datetime(random_year, random_month, random_day, random.randint(0, 23), random.randint(0, 59))

    chat_data.append({
        "user": user_input,
        "category": category,
        "bot": bot_reply,
        "sentiment": sentiment,
        "timestamp": timestamp
    })

# Insert Data into MongoDB
collection.insert_many(chat_data)

print("✅ 50 Chat Records Inserted Successfully from 2020 - 2025!")


# --------------------------------------------------------------------------------------------------------------------------------

# Load environment variables
# MONGO_URI = CONFIG.MONGO_URI

# # Connect to MongoDB Atlas
# client = pymongo.MongoClient(MONGO_URI)
# db = client[CONFIG.DB_NAME]  # Ensure this matches your database name
# collection = db[CONFIG.CHAT_HISTORY_COLLECTION]  # Ensure this matches your collection name

# # Delete All Documents from the Collection
# delete_result = collection.delete_many({})

# print(f"✅ Deleted {delete_result.deleted_count} documents from chat_history!")
