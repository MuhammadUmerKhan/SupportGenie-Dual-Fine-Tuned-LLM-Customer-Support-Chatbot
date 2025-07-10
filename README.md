# 📌 **SupportGenie: AI Assistant for Customer Support** 🤖

![ai_chatbot.png](https://www.addevice.io/storage/ckeditor/uploads/images/64d0d72b8dcde_the.role.of.chatbots.and.humans.in.customer.support.1.png)

## 🚀 **Project Overview**
In today's fast-paced digital world, businesses need **efficient and scalable** customer support solutions. **SupportGenie** leverages **AI-powered chatbots**, **FAQ retrieval**, **sentiment analysis**, **fine-tuned LLMs**, and **interactive analytics dashboards** to deliver exceptional customer experiences. 🌟

💡 **What makes SupportGenie unique?**
- 💬 **AI-Powered Chatbot**: Combines FAQ retrieval with responses from fine-tuned **Mistral-7B-Instruct-v0.3** and **LLaMA-3-8B-Instruct** for accurate, banking-specific answers.
- 😊 **Sentiment Analysis**: Classifies interactions as **positive**, **negative**, or **neutral** to understand customer emotions.
- ⚡ **FAISS Vector Search**: Enables fast and accurate FAQ retrieval using semantic embeddings.
- 💾 **MongoDB Integration**: Stores chat history, feedback, and analytics for continuous learning.
- 📊 **Interactive Analytics Dashboard**: Offers real-time insights into customer interactions and trends.
- 🌐 **Streamlit UI**: Provides a user-friendly interface for chatbot and analytics.

---

## 📁 **Table of Contents**
- [📌 Problem Statement](#problem-statement)
- [🛠️ Solution Approach](#solution-approach)
- [🔥 Project Features](#project-features)
- [🤖 Chatbot & LLM Functionality](#chatbot--llm-functionality)
- [📊 Analytics Dashboard](#analytics-dashboard)
- [🧠 Fine-Tuned LLM Details](#fine-tuned-llm-details)
- [⚙️ Setup and Installation](#setup-and-installation)
- [🌐 Deployment](#deployment)
- [🚀 Future Improvements](#future-improvements)
- [📌 Conclusion](#conclusion)

---

## 📌 **Problem Statement**
Customer support teams face **high workloads** and **delays**, leading to **poor user experiences**. The challenge is:  
**"Can we automate responses to common queries while understanding customer sentiment and optimizing support?"** 🤔

We address this with:
- ⚡ A **fast, accurate chatbot** powered by FAQs and fine-tuned LLMs (Mistral and LLaMA).
- 😊 **Sentiment analysis** to categorize customer feedback.
- 📈 **Real-time analytics** to monitor trends and improve responses.

---

## 🛠️ **Solution Approach**
SupportGenie uses **AI, NLP, and analytics** to automate and enhance customer interactions: 🎉

1. **FAQ-Based Chatbot with Fine-Tuned LLMs** 🗣️  
   - Retrieves answers from a **[BankFAQs.csv dataset](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv)** using **FAISS vector search** for semantic matching.  
   - Generates responses with fine-tuned **Mistral-7B-Instruct-v0.3** or **LLaMA-3-8B-Instruct** for queries without FAQ matches, tailored for banking contexts.

2. **Sentiment Analysis & Feedback** 😊  
   - Detects emotions (Positive, Negative, Neutral) using NLP models.  
   - Stores interactions in **MongoDB** for trend analysis and learning.

3. **Real-Time Analytics Dashboard** 📊  
   - Tracks usage, sentiment trends, and engagement patterns.  
   - Provides actionable insights to optimize customer support.

---

## 🔥 **Project Features**
- 💬 **AI Chatbot**: Instant support with FAQ retrieval and LLM-generated responses.
- 🧠 **Fine-Tuned LLMs**: Context-specific answers for banking queries using Mistral and LLaMA.
- 😊 **Sentiment Analysis**: Understands customer emotions for better interactions.
- ⚡ **FAISS Vector Search**: Fast FAQ retrieval via semantic embeddings.
- 💾 **MongoDB Integration**: Stores chat history and analytics data.
- 🌐 **Streamlit UI**: Interactive interface for chatbot and analytics.

---

## 🤖 **Chatbot & LLM Functionality**
1. **Understanding Input** 🧠  
   - Detects **language** and **query intent** using NLP techniques.
2. **Classifying Questions** 📋  
   - Categorizes queries (e.g., Loans, Security, Payments) with classification models.
3. **Retrieving Answers** 🔍  
   - Searches **FAISS database** for FAQ matches using vector embeddings.  
   - Falls back to fine-tuned **Mistral-7B-Instruct-v0.3** or **LLaMA-3-8B-Instruct** for dynamic responses.
4. **Sentiment & Storage** 💾  
   - Predicts sentiment (Positive, Negative, Neutral) using TextBlob and transformers.  
   - Logs interactions in MongoDB for analytics and retraining.

---

## 📊 **Analytics Dashboard**
- 😊 **Sentiment Distribution**: Visualizes Positive, Negative, and Neutral interactions.
- 📈 **Trends Over Time**: Tracks chatbot usage and sentiment shifts.
- 🕒 **Engagement Heatmap**: Highlights peak usage hours.
- 🔥 **Top FAQs**: Identifies frequently asked questions for dataset refinement.
- 🔍 **LLM Performance**: Monitors fine-tuned model accuracy and relevance for both Mistral and LLaMA.

| Visualization Type | Insights |
|--------------------|----------|
| **Most Frequent Questions** | ![Feature Importance](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/most_fre_ques.png) |
| **Sentiments Over Time** | ![Sentiment Over Time](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sent_ovr_time.png) |
| **Sentiment Trend** | ![Sentiment Trend](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sent_trend.png) |
| **Sentiment Distribution** | ![Sentiment Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sentiment_distribution.png) |

---

## 🧠 **Fine-Tuned LLM Details**

### **Model Overview**
SupportGenie uses **Mistral-7B-Instruct-v0.3** (7.3 billion parameters) and **LLaMA-3-8B-Instruct** (8 billion parameters), both fine-tuned for banking-specific customer support. 🌟 Mistral’s efficiency (via Grouped-query Attention and Sliding Window Attention) and LLaMA’s conversational fluency make them ideal for generating accurate, context-aware responses. The fine-tuned models excel in handling banking queries with improved relevance and clarity. 🎉

### **Fine-Tuning Process**
The fine-tuning process adapts **Mistral-7B-Instruct-v0.3** and **LLaMA-3-8B-Instruct** for banking-specific tasks: 🚀

1. **Dataset Selection** 📊  
   - **Original Dataset**: **[FAQ_Dataset](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQ_Dataset)** (~1,000 question-answer pairs from [BankFAQs.csv](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv)), covering banking topics like loans, account management, and security.  
   - **Reformatted Dataset for Mistral**: **[FAQs-Mistral-7b-v03-17k](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k)** (17,000 instruction-response pairs), augmented with synthetic data using GPT-4 and self-reflection for diverse banking scenarios. Formatted as JSONL with `[INST]` and `[/INST]` tokens.  
   - **Reformatted Dataset for LLaMA**: **[FAQs-Meta-Llama-3-8B-Instruct](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Meta-Llama-3-8B-Instruct)** (1,764 instruction-response pairs), formatted with `<|begin_of_text|>`, `<|user|>`, `<|assistant|>`, and `<|end_of_text|>` tokens.

2. **Preprocessing** 🛠️  
   - Converted CSV to JSONL, cleaning duplicates and ensuring consistency for both models.  
   - Validated data to skip problematic samples, ensuring training stability.  
   - Uploaded to Hugging Face as **[FAQs-Mistral-7b-v03-17k](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k)** and **[FAQs-Meta-Llama-3-8B-Instruct](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Meta-Llama-3-8B-Instruct)**.

3. **Fine-Tuning Setup** ⚙️  
   - **Environment**: Google Colab T4 GPU with **4-bit quantization** (`BitsAndBytesConfig`) for memory efficiency.  
   - **Technique**: Parameter-Efficient Fine-Tuning (PEFT) with **LoRA** (Low-Rank Adaptation):  
     - **Mistral**: `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, targeting `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]`.  
     - **LLaMA**: `r=8`, `lora_alpha=16`, `lora_dropout=0.1`, targeting `["q_proj", "v_proj"]`.  
     - Task: `CAUSAL_LM`.  
   - **Parameters**: Learning rate `2e-4` for Mistral, adjusted for LLaMA, 1 epoch, batch size 4, gradient checkpointing, AdamW optimizer.  
   - **Tokenizers**: Mistral and LLaMA tokenizers with right padding and EOS as pad token.  
   - **Duration**: ~12 hours for Mistral, ~10 hours for LLaMA on T4 GPU.

4. **Training** 🚀  
   - Used **Supervised Fine-Tuning (SFT)** with `SFTTrainer` from `trl` library for both models.  
   - Monitored training loss and perplexity, with 5% validation split for generalization.  
   - Logged metrics via Weights & Biases (W&B) for transparency.  
   - Mistral loss: ~1.78; LLaMA loss: ~1.49.

5. **Merging & Validation** ✅  
   - Merged LoRA adapters with base models using `PeftModel`.  
   - Validated on test set, confirming improved accuracy and relevance for banking queries.

6. **Pushing to Hugging Face** 🌐  
   - Authenticated via `huggingface-cli login`.  
   - Created repositories:  
     - Mistral: [Muhammad-Umer-Khan/SupportGenie-Mistral-7B](https://huggingface.co/Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned).  
     - LLaMA: [Muhammad-Umer-Khan/SupportGenie-LLaMA-3-8B](https://huggingface.co/Muhammad-Umer-Khan/LLaMA-3-8B-FAQs-Finetuned).  
   - Pushed models and tokenizers to Hugging Face Hub.  
   - Uploaded datasets to respective repositories.

7. **Integration** 🔗  
   - Integrated into `chatbot_analytics.py` for the “Fine-Tuned Bot” Streamlit page, using same chat history and response logic as base chatbot.

### **Dataset Details**
- **Original Dataset**:  
  - Source: [BankFAQs.csv](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv).  
  - Size: ~1,000 pairs.  
  - Content: Banking topics (loans, security, payments).  
  - Format: CSV (question, answer).  
- **Reformatted Dataset for Mistral**:  
  - Size: 17,000 pairs.  
  - Augmentation: Synthetic data via GPT-4 and self-reflection.  
  - Format: JSONL with `prompt` and `response` (e.g., `{"messages": [{"role": "user", "content": "[INST] How can I check my account balance? [/INST]"}, {"role": "assistant", "content": "Log into your online banking portal or mobile app..."}]}`).  
- **Reformatted Dataset for LLaMA**:  
  - Size: 1,764 pairs.  
  - Format: JSONL with conversation pairs (e.g., `{"messages": [{"role": "user", "content": "How can I check my account balance?"}, {"role": "assistant", "content": "Log into your online banking portal or mobile app..."}]}`).  
  - Purpose: Instruction tuning for LLaMA compatibility.

### **Performance Improvements**
- ✅ **Accuracy**: Higher response relevance for banking queries from both models.  
- 🧠 **Contextual Understanding**: Improved handling of banking terminology.  
- 📝 **Response Quality**: Mistral offers concise answers; LLaMA provides fluent responses.  
- 📊 **Metrics**: Lower perplexity and better coherence (via W&B logs).

### **Challenges & Solutions**
- 🖥️ **Memory**: Used 4-bit quantization and LoRA for T4 GPU compatibility for both models.  
- 📋 **Data Errors**: Validated and skipped problematic samples.  
- ⚙️ **Loading Issues**: Added `"model_type": "mistral"` to `config.json` for Mistral; adjusted LLaMA config accordingly.

---

## ⚙️ **Setup and Installation**
1. **Clone Repository** 📂  
   ```bash
   git clone https://github.com/your-repo/AI-Powered-Customer-Support-System.git
   cd AI-Powered-Customer-Support-System
   ```

2. **Install Dependencies** 📦  
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up MongoDB Atlas** 💾  
   - Create a MongoDB Atlas cluster.  
   - Add connection string to `.env`:  
     ```ini
     MONGO_URI=mongodb+srv://your_user:your_password@your_cluster.mongodb.net/chatbotDB?retryWrites=true&w=majority&appName=Cluster0
     ```

4. **Set Up API Keys** 🔑  
   - Add Groq API key to `.env`:  
     ```ini
     GROQ_API_KEY=your_grok_api_key
     ```

5. **Load Fine-Tuned LLMs** 🧠  
   - Use **[Muhammad-Umer-Khan/SupportGenie-Mistral-7B](https://huggingface.co/Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned)**:  
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer
     model = AutoModelForCausalLM.from_pretrained("Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned")
     tokenizer = AutoTokenizer.from_pretrained("Muhammad-Umer-Khan/Mistral-7b-v03-FAQs-Finetuned")
     ```
   - Use **[Muhammad-Umer-Khan/SupportGenie-LLaMA-3-8B](https://huggingface.co/Muhammad-Umer-Khan/Llama-3-8B-FAQs-Finetuned)**:  
     ```python
     from transformers import AutoModelForCausalLM, AutoTokenizer
     model = AutoModelForCausalLM.from_pretrained("Muhammad-Umer-Khan/Llama-3-8B-FAQs-Finetuned")
     tokenizer = AutoTokenizer.from_pretrained("Muhammad-Umer-Khan/SupportGenie-LLaMA-3-8B")
     ```
   - Ensure `config.json` includes `"model_type": "mistral"` for Mistral and appropriate LLaMA settings.

6. **Run Streamlit App** 🚀  
   ```bash
   streamlit run main.py
   ```

---

1. **FastAPI Server** 🖥️  
   ```bash
   uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
   ```

---

## 🚀 **Future Improvements**
- 👤 **User Sessions**: Personalize interactions for returning users.  
- 🧠 **Advanced LLM Tuning**: Use larger datasets for better responses from both Mistral and LLaMA.  
- 🎙️ **Voice Interaction**: Add speech recognition for voice queries.  
- 📱 **Messaging Apps**: Integrate with WhatsApp and Telegram.  
- 😊 **Enhanced Sentiment Analysis**: Use transformers for deeper insights.  
- 🔮 **Proactive Suggestions**: Predict user needs from chat history.

---

## 🧐 **Test Fine Tuned LLMs**
 - Try at **[Notebook](https://colab.research.google.com/drive/1_XijXkhLfuTMmpfkluWJXmRa7oPsFfuH?usp=sharing)** 👈

---

## 📢 **Shoutout**
Big thanks to **[MrJay10](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv)** for the FAQ dataset! 🙌 🎉

---

## 🌟 **Live Demo**
Try **SupportGenie** at: [SupportGenie AI Chatbot](https://ai-powered-customer-support-and-analytics-system.streamlit.app/) 🚀

---

## 📌 **Conclusion**
**SupportGenie** revolutionizes customer support with **AI-driven chatbots**, **fine-tuned LLMs (Mistral and LLaMA)**, **sentiment analysis**, and **real-time analytics**. Its scalable design and insightful dashboards empower businesses to enhance customer engagement across industries. 🌍 🎊

🔹 **Want to contribute?** Fork the repo and submit a PR! 🎉
