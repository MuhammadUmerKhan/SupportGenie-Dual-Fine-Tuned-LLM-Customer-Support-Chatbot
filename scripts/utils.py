# Import required libraries
import re, streamlit as st, torch  # Streamlit for building UI
from streamlit.logger import get_logger  # Streamlit's built-in logger
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain_groq import ChatGroq
import scripts.config as CONFIG
load_dotenv()  # ✅ Load environment variables from .env


# Initialize logger for tracking interactions and errors
logger = get_logger("LangChain-Chatbot")

# ✅ Decorator to enable chat history
def enable_chat_history(func):
    """
    Decorator to handle chat history and UI interactions.
    Ensures chat messages persist across interactions.
    """
    current_page = func.__qualname__  # Get function name to track current chatbot session

    # Clear session state if model/chatbot is switched
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = current_page  # Store the current chatbot session
    if st.session_state["current_page"] != current_page:
        try:
            st.cache_resource.clear()  # Clear cached resources
            del st.session_state["current_page"]
            del st.session_state["messages"]
        except Exception:
            pass  # Ignore errors if session state keys do not exist

    # Initialize chat history if not already present
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display chat history in the UI
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)  # Execute the decorated function

    return execute

def remove_think_tags(text):
    """Remove text between <think> and </think> tags using regex."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

@st.cache_resource
def get_cached_llm_response(user_input):
    """Cache LLM responses to avoid redundant API calls."""
    llm = ChatGroq(
        temperature=0,
        groq_api_key=CONFIG.GROK_API_KEY,
        model_name="qwen/qwen3-32b"
    )
    response = llm.invoke(user_input)
    return remove_think_tags(response.content.strip())

def display_msg(msg, author):
    """
    Displays a chat message in the UI and appends it to session history.

    Args:
        msg (str): The message content to display.
        author (str): The author of the message ("user" or "assistant").
    """
    st.session_state.messages.append({"role": author, "content": msg})  # Store message in session
    st.chat_message(author).write(msg)  # Display message in Streamlit UI

def print_qa(cls, question, answer):
    """
    Logs the Q&A interaction for debugging and tracking.

    Args:
        cls (class): The calling class.
        question (str): User question.
        answer (str): Model response.
    """
    log_str = f"\nUsecase: {cls.__name__}\nQuestion: {question}\nAnswer: {answer}\n" + "-" * 50
    logger.info(log_str)  # Log the interaction using Streamlit's logger

@st.cache_resource
def configure_vector_embeddings():
    """
    Configures and caches the vector embeddings for Groq API.

    Returns:
        vector_embeddings (HuggingFaceEmbeddings): The loaded vector embeddings.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Load and return the vector embeddings

# ✅ Load Fine-Tuned Mistral LLM from Hugging Face
@st.cache_resource
def load_finetuned_mistral():
    # Use BitsAndBytes for quantized loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned")

    # Load model in 4-bit to fit in Colab GPU
    model = AutoModelForCausalLM.from_pretrained(
        "Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Create inference pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512, device_map="auto")
    
    return pipe

def get_chatbot_response(user_input):
    try:
        model_pipe = load_finetuned_mistral()
        prompt = f"<s>[INST] {user_input} [/INST]"
        result = model_pipe(prompt)[0]['generated_text']
        return result.replace(prompt, '').strip()
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        return "Sorry, I encountered an issue while generating the response."

def sync_st_session():
    """
    Ensures Streamlit session state values are properly synchronized.
    """
    for k, v in st.session_state.items():
        st.session_state[k] = v  # Sync all session state values
