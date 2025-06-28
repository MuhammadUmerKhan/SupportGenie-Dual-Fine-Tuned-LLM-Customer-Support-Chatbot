import pandas as pd, plotly.express as px, plotly.graph_objects as go, streamlit as st
import scripts.utils as utils
import scripts.config as CONFIG
from scripts.chatbot import get_chatbot_response, connect_mongo

# Function to Run Chatbot UI
def get_chat_history():
    """Fetch chat history from MongoDB"""
    db = connect_mongo()
    if db is None:
        return {"error": "Database connection failed."}
    chat_collection = db[CONFIG.CHAT_HISTORY_COLLECTION]
    chat_history = list(chat_collection.find({}, {"_id": 0}))  # Remove MongoDB ID field
    return {"data": chat_history}

def get_faqs():
    """Fetch FAQs from MongoDB"""
    db = connect_mongo()
    if db is None:
        return {"error": "Database connection failed."}
    faq_collection = db[CONFIG.FAQ_COLLECTION]
    faqs = list(faq_collection.find({}, {"_id": 0}))  # Remove MongoDB ID field
    return {"data": faqs}

def chatbot():
    """Displays the AI Chatbot Interface without API Calls."""
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>💬 AI Customer Support Chatbot</h1>", unsafe_allow_html=True)
    st.write("Ask any question or click on a common FAQ below:")

    # Initialize Chat History
    utils.enable_chat_history(lambda: None)

    # Handle User Input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        utils.display_msg(user_input, "user")  # Display user's message in chat
        bot_response = get_chatbot_response(user_input)  # Call function directly
        utils.display_msg(bot_response, "assistant")  # Store bot's response in chat history

def finetuned_chatbot():
    """Displays the Fine-Tuned LLM Chatbot Interface."""
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>💬 Fine-Tuned AI Customer Support Chatbot</h1>", unsafe_allow_html=True)
    st.write("Chat with the fine-tuned Mistral LLM. Ask any question below:")

    # Initialize Chat History for Fine-Tuned LLM
    utils.enable_chat_history(lambda: None)

    # Handle User Input
    user_input = st.chat_input("Type your question here...")
    if user_input:
        utils.display_msg(user_input, "user")  # Display user's message in chat
        bot_response = utils.get_fined_tuned_chatbot_response(user_input)  # Use the fine-tuned LLM response function
        utils.display_msg(bot_response, "assistant")  # Store bot's response in chat history

def show_finetuned_llm_details():
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>🧠 Fine-Tuned Mistral-7B: Intelligent FAQ Assistant</h1>", unsafe_allow_html=True)

    st.warning("""
    ❗ Due to high GPU memory requirements (~15GB VRAM) and limited storage on free hosting platforms (e.g., Gradio, Streamlit, Hugging Face Spaces), this fine-tuned model **cannot be deployed directly online**.  
    👉 Run it on Google Colab with 4-bit quantization for smooth performance.
    """)

    st.markdown("""
    This project showcases a **fine-tuned Mistral-7B-Instruct-v0.3** model, tailored to serve as an **FAQ assistant** for customer support in domains like banking. The model was enhanced using **QLoRA** (4-bit quantization with LoRA adapters) on Google Colab’s T4 GPU, enabling efficient training on limited resources.

    ---
    ### 📌 Model Overview
    - **Base Model**: [`mistralai/Mistral-7B-Instruct-v0.3`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
    - **Fine-Tuned Model**: [`Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned`](https://huggingface.co/Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned)
    - **Architecture**: Decoder-only transformer (Mistral)
    - **Fine-Tuning Method**: QLoRA (4-bit quantization + LoRA adapters)
    - **Quantization**: 4-bit `nf4` using `bitsandbytes`
    - **Libraries**: `transformers`, `peft`, `trl`, `datasets`, `accelerate`, `huggingface_hub`
    - **Training Environment**: Google Colab with T4 GPU (~15GB VRAM)
    """)

    st.divider()

    st.markdown("### 🧾 Dataset Used")
    st.markdown("""
    A custom dataset of **1,764 question-answer pairs** was created from banking FAQs, covering topics like security, loans, and fund transfers. It was reformatted into the **Mistral instruction format** for fine-tuning.
    - **Original Dataset**: [`Muhammad-Umer-Khan/FAQ_Dataset`](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQ_Dataset) (raw CSV)
    - **Reformatted Dataset**: [`Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k`](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k) (JSONL with Mistral format)
    - **Format Example**:
    """)
    st.code("""
[
  {
    "text": "<s>[INST] Do I need to enter ‘#’ after keying in my Card number/ Card expiry date/ CVV number [/INST] Please listen to the recorded message and follow the instructions while entering your card details. </s>"
  },
  {
    "text": "<s>[INST] How can I obtain an IVR Password [/INST] By Sending SMS request: Send an SMS 'PWD<space>last 4 digits of your card no.' to 97171 80808 from your registered mobile number. </s>"
  }
]
    """, language="json")
    st.markdown("""
    - ✅ Cleaned by removing missing values.
    - ✅ Formatted into `<s>[INST] question [/INST] answer </s>` for Mistral compatibility.
    - ✅ Saved as JSONL and uploaded to Hugging Face Hub.
    - ✅ Used training split with 1,764 samples for supervised fine-tuning.
    """)

    st.divider()

    st.markdown("### 🛠️ Step-by-Step Fine-Tuning Process")

    st.markdown("""
    #### 🔹 1. Dataset Preparation
    Loaded `BankFAQs.csv` with 1,764 Q&A pairs, cleaned missing values, and reformatted into Mistral’s instruction format. Converted to a Hugging Face `Dataset` and saved as JSONL.

    #### 🔹 2. Environment Setup
    Used Google Colab’s T4 GPU. Installed `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`, and `accelerate`. Configured memory optimization with `PYTORCH_CUDA_ALLOC_CONF`.

    #### 🔹 3. Load Base Model in 4-Bit
    Loaded `mistralai/Mistral-7B-Instruct-v0.3` with 4-bit quantization (`nf4`) using `BitsAndBytesConfig` to fit within Colab’s memory limits. Set `device_map="auto"` for GPU distribution.

    #### 🔹 4. Load Tokenizer
    Loaded Mistral’s tokenizer and set padding to the EOS token for training alignment.

    #### 🔹 5. Apply LoRA with PEFT
    Configured LoRA to target `q_proj` and `v_proj` layers with rank=8, alpha=16, and dropout=0.1. Applied using `peft` for parameter-efficient fine-tuning.

    #### 🔹 6. Train with SFTTrainer
    Used `SFTTrainer` from `trl` with:
    - Batch size: 8
    - Gradient accumulation: 4 (effective batch size 32)
    - Epochs: 1
    - Optimizer: AdamW with weight decay
    - Logging: Every 20 steps
    Loss dropped from ~2.3 to ~1.6, showing improved performance.

    #### 🔹 7. Save & Push to Hugging Face
    Saved LoRA adapter weights and tokenizer to `Mistral-FAQs-Lora`. Pushed to `Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned` on Hugging Face Hub.
    """)

    st.divider()

    st.markdown("### ⚙️ Inference Prompt Format")
    st.code("<s>[INST] your question here [/INST]", language="html")
    st.markdown("""
    - `[INST]` signals the start of the user’s question.
    - The model generates the answer, and the tokenizer adds `</s>` to close the response.
    - Ensures structured, FAQ-style responses for customer queries.
    """)

    st.divider()

    st.markdown("### 🚀 Try it Out on Colab (Live Demo)")
    st.markdown("""
    [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/your_colab_link_here)

    Test the model on Colab with:
    - ✅ One-click environment setup
    - ✅ 4-bit model loading
    - ✅ Interactive prompt-based testing
    - ✅ Option to input custom questions
    Replace `your_colab_link_here` with the actual notebook link.
    """)

    st.divider()

    st.markdown("### 🔬 Final Training Configuration & Results")
    st.markdown("""
    - **Tokenizer**: MistralTokenizer
    - **Batch Size**: 8
    - **Gradient Accumulation**: 4
    - **Epochs**: 1
    - **Optimizer**: AdamW with weight decay
    - **Quantization**: 4-bit `nf4` via `bitsandbytes`
    - **LoRA Parameters**: `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, targeting `q_proj`, `v_proj`
    - **Memory Optimization**: Gradient checkpointing, garbage collection
    - **Dataset**: 1,764 Q&A pairs in Mistral format

    📊 **Result**: The model delivers accurate, FAQ-style responses for customer support queries, with training loss reduced from ~2.3 to ~1.6. Lightweight LoRA adapters enable efficient deployment.
    """)

    st.success("📌 Run the model on Colab to test or extend it for your use case!")

# Function to Run FAQ Page
def faq_page():
    """Displays the FAQs from the dataset."""
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>📖 Frequently Asked Questions (FAQs)", unsafe_allow_html=True)
    faqs_data = get_faqs()
    if "error" in faqs_data:
        st.error(faqs_data["error"])
        st.stop()
    
    df_faqs = pd.DataFrame(faqs_data["data"])
    for index, row in df_faqs.iterrows():
        with st.expander(f"❓ {row['question']}"):
            st.write(f"**Answer:** {row['answer']}")

# Function to Run Analytics Dashboard
def analytics():
    """Displays the AI Analytics Dashboard without API Calls."""
    st.markdown("<h1 style='text-align: center; color: #FFA500;'>💹 AI Customer Support - Analytics Dashboard</h1>", unsafe_allow_html=True)

    # Fetch Data Directly from MongoDB
    chat_data = get_chat_history()
    if "error" in chat_data:
        st.error(chat_data["error"])
        st.stop()

    df = pd.DataFrame(chat_data["data"])

    # Convert Timestamp Column
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 📌 Sentiment Distribution - Enhanced Pie Chart
    st.subheader("🧠 Sentiment Distribution 📊")
    sentiment = df[df['sentiment'] != "Not a Review"]
    sentiment_counts = sentiment["sentiment"].value_counts()
    fig_sentiment_pie = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title="Customer Sentiment Distribution",
        color=sentiment_counts.index,
        color_discrete_map={"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#FFA500"},
        hole=0.3,
        labels={"labels": "Sentiments"}
    )
    fig_sentiment_pie.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0.05])
    with st.expander("View Chart", expanded=True):
        st.plotly_chart(fig_sentiment_pie, use_container_width=True)

    st.divider()
    
    # 📌 Sentiment Trends Over Time - Interactive Area Chart
    st.subheader("📈 Sentiment Trends Over Time ⏳")
    df_sentiment = df[df['sentiment'] != "Not a Review"]
    df_sentiment["timestamp"] = pd.to_datetime(df_sentiment["timestamp"])
    df_sentiment["year_month"] = df_sentiment["timestamp"].dt.to_period("M")
    df_sentiment_time = df_sentiment.groupby(["year_month", "sentiment"]).size().unstack(fill_value=0)

    fig_sentiment_trend = px.area(
        df_sentiment_time,
        x=df_sentiment_time.index.astype(str),
        y=df_sentiment_time.columns,
        title="Sentiment Trends Over Time",
        labels={"x": "Year-Month", "y": "Sentiment Count"},
        color_discrete_map={"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#FFA500"},
        markers=True,
    )
    with st.expander("View Chart", expanded=True):
        st.plotly_chart(fig_sentiment_trend, use_container_width=True)

    st.divider()
    
    # 📌 Sentiment Over Year - Interactive Line Chart
    st.subheader("📊 Sentiment Trends by Year 📅")
    sentiment_year = df[df['sentiment'] != "Not a Review"]
    sentiment_year["year"] = sentiment_year["timestamp"].dt.year
    selected_year = st.selectbox("Select Year", sorted(sentiment_year["year"].unique(), reverse=True))
    df_year = sentiment_year[sentiment_year["year"] == selected_year]
    df_year["month"] = df_year["timestamp"].dt.month_name()
    df_sentiment_yearly = df_year.groupby(["month", "sentiment"]).size().unstack(fill_value=0)
    df_sentiment_yearly = df_sentiment_yearly.reindex(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        fill_value=0
    )
    
    fig_sentiment_yearly = px.line(
        df_sentiment_yearly,
        x=df_sentiment_yearly.index,
        y=df_sentiment_yearly.columns,
        labels={"x": "Month", "y": "Sentiment Count"},
        title=f"📊 Sentiment Trends for {selected_year}",
        markers=True,
        color_discrete_map={"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#FFA500"}
    )
    with st.expander("View Chart", expanded=True):
        st.plotly_chart(fig_sentiment_yearly, use_container_width=True)


    st.divider()
    
    st.subheader("📈 Most Frequently Asked Question Categories")
    # 📌 Most Asked FAQs - Advanced Horizontal Bar Chart
    category_counts = df["category"].value_counts()
    fig_category = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={"x": "Category", "y": "Number of Questions"},
        title="Most Frequently Asked Question Categories",
        color=category_counts.values,
        color_continuous_scale=px.colors.sequential.Viridis
    )

    fig_category.update_traces(marker=dict(line=dict(width=2, color="black")))
    with st.expander("", expanded=True):
        st.plotly_chart(fig_category, use_container_width=True)
    
    st.divider()
    
    # ====== 📌 User Engagement Heatmap (Hourly) ======
    st.subheader("🔥 User Engagement by Time of Day")
    df["hour"] = df["timestamp"].dt.hour
    hourly_counts = df["hour"].value_counts().sort_index()

    fig_heatmap = go.Figure(go.Heatmap(
        z=hourly_counts.values.reshape(1, -1),
        x=hourly_counts.index,
        colorscale="reds"
    ))
    fig_heatmap.update_layout(
        title="User Engagement Across Different Hours",
        xaxis_title="Hour of Day",
        yaxis_title="Engagement Level",
        template="plotly_dark"
    )
    with st.expander("", expanded=True):
        st.plotly_chart(fig_heatmap, use_container_width=True)