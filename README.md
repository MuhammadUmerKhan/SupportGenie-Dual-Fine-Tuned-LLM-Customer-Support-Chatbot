# 📌 **SupportGenie: AI Assistant for Customer Support**

![ai_chatbot.png](https://www.addevice.io/storage/ckeditor/uploads/images/64d0d72b8dcde_the.role.of.chatbots.and.humans.in.customer.support.1.png)

## 🚀 **Project Overview**
In today's digital world, businesses need **efficient and scalable** customer support solutions. This project leverages **AI-powered chatbots, FAQ retrieval, sentiment analysis, fine-tuned LLMs, and analytics dashboards** to enhance customer experience.

💡 **What makes this project unique?**
- 👉 **AI-Powered Chatbot** → Retrieves responses from **FAQs** and generates answers using **fine-tuned LLMs**.
- 👉 **Multilingual Support** → Automatically detects **input language**, translates it into **English**, processes it, and responds in the original language.
- 👉 **Sentiment Analysis** → Understands customer emotions to classify interactions as **positive, negative, or neutral**.
- 👉 **FAISS Vector Search** → Stores and retrieves **FAQ embeddings** for **fast and accurate** responses.
- 👉 **MongoDB Integration** → Stores **chat history, feedback, and analytics**.
- 👉 **Interactive Analytics Dashboard** → Provides **data insights** on chatbot interactions and sentiment trends.
- 👉 **Streamlit UI** → Web-based **interactive chatbot and analytics dashboard**.
- 👉 **Fine-Tuned LLM** → Custom-trained **Mistral-7B-Instruct-v0.3** for accurate, context-specific responses tailored to banking queries.

---

## **📁 Table of Contents**
- [📌 Problem Statement](#-problem-statement)
- [🛠️ Solution Approach](#-solution-approach)
- [🔥 Project Features](#-project-features)
- [📊 AI-Powered Chatbot](#-ai-powered-chatbot)
- [📈 Analytics Dashboard](#-analytics-dashboard)
- [🧠 Fine-Tuned LLM Details](#-fine-tuned-llm-details)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🚀 Running the Chatbot & Analytics](#-running-the-chatbot--analytics)
- [🖥️ Deployment on Streamlit Cloud](#-deployment-on-streamlit-cloud)
- [🛠️ Future Improvements](#-future-improvements)
- [📌 Conclusion](#-conclusion)

---

## 📌 **Problem Statement**
Customer support teams face **high workloads and delays**, leading to **poor user experience**. The challenge is:
**"Can we automate responses to common queries while understanding customer sentiment and improving support?"**

To solve this, we need:
- 👉 A **fast & accurate chatbot** to **handle FAQs** automatically with enhanced LLM capabilities.
- 👉 **Sentiment analysis** to categorize **customer feedback**.
- 👉 **Real-time analytics** to monitor trends and **optimize responses**.

---

## 🛠️ **Solution Approach**
Our solution uses **AI chatbots, NLP, fine-tuned LLMs, and analytics** to **automate and improve customer interactions**.

### **1️⃣ FAQ-Based Chatbot**
- 🚀 **Retrieves relevant answers** from a pre-defined **[FAQ dataset](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv)**, a comprehensive CSV file containing banking-related questions and answers.
- 📡 **Uses FAISS for vector search** to fetch the most relevant FAQ based on semantic similarity.
- 🤖 **Generates responses** via a fine-tuned **Mistral-7B-Instruct-v0.3** when no FAQ matches, trained on a curated dataset to improve response accuracy and relevance for banking-specific queries.

### **2️⃣ Sentiment Analysis & Feedback Collection**
- 🧠 **Detects user sentiment** (Positive, Negative, Neutral) using advanced NLP models.
- 📊 **Stores insights in MongoDB for continuous learning**, including chat history, sentiment, and user feedback for trend analysis.

### **3️⃣ Real-Time Analytics Dashboard**
- 📈 **Tracks chatbot usage & sentiment trends over time** with interactive visualizations.
- 🎨 **Provides insights** into customer engagement, query categories, and peak interaction times.
- 🔄 **Helps optimize responses** by identifying areas for improvement using analytics data.

---

## 🔥 **Project Features**
- **AI-Powered Chatbot** for **instant support** with FAQ retrieval and LLM-generated responses.
- **Fine-Tuned LLM** for **context-specific and accurate responses** tailored to banking queries.
- **Sentiment Analysis & Feedback** to understand customer emotions and improve interactions.
- **FAISS Vector Search** for **fast FAQ retrieval** using semantic embeddings.
- **MongoDB Integration** for **chat storage** and analytics data management.
- **Interactive Streamlit UI** for **chatbot, fine-tuned bot, and analytics dashboard**.

---

## 📈 **Analytics Dashboard**
- ✅ **Sentiment Distribution** (Positive, Negative, Neutral) to gauge customer satisfaction.
- ✅ **Trends Over Time** – Tracks **chatbot usage patterns** and sentiment shifts.
- ✅ **Engagement Heatmap** – Shows **peak chatbot usage hours** for operational insights.
- ✅ **Top FAQs** – Identifies **most asked questions** to refine FAQ dataset.

---

## 🔁 **LLM Functionality**

### **1️⃣ Understanding User Input**
- Detects **language** and **query intent** using NLP techniques.

### **2️⃣ Classifying Questions**
- Determines **category (e.g., Loans, Security, Payments, etc.)** using classification models.

### **3️⃣ Retrieving Answers**
- Searches **FAISS database** for relevant FAQ answers based on vector embeddings.
- If no match is found, generates a response **using a fine-tuned Mistral-7B-Instruct-v0.3**, trained on a banking-specific dataset derived from the FAQ CSV and additional customer interaction logs.

### **4️⃣ Sentiment Analysis & Storage**
- Predicts **user sentiment** (Positive, Negative, Neutral) using TextBlob and transformer models.
- Stores **chat history, sentiment, and feedback** in MongoDB for analytics and model retraining.

---

## 🧠 **Fine-Tuned LLM Details**

### **Model Overview**
The project utilizes **Mistral-7B-Instruct-v0.3**, a 7.3 billion parameter language model developed by Mistral AI, fine-tuned for instruction-based tasks. This model was chosen for its high performance in natural language understanding and generation, as well as its efficiency with techniques like Grouped-query Attention (GQA) and Sliding Window Attention (SWA). The fine-tuned version enhances its ability to provide accurate, context-specific responses for banking-related customer support queries.[](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

### **Fine-Tuning Process**
The fine-tuning process was conducted to adapt **Mistral-7B-Instruct-v0.3** for banking-specific customer support tasks. Below is a detailed breakdown of the process:

1. **Dataset Selection**:
   - **Original Dataset**: The base dataset used was the **[FAQ_Dataset](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQ_Dataset)**, which contains banking-related question-answer pairs derived from the [BankFAQs.csv](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv). This dataset includes approximately 1,000 high-quality FAQ pairs covering topics like loans, account management, security, and payments.[](https://mistral.ai/news/announcing-mistral-7b)
   - **Reformatted Dataset**: To ensure compatibility with the fine-tuning pipeline, the original dataset was reformatted into a JSONL format suitable for Mistral’s training requirements. The reformatted dataset, **[FAQs-Mistral-7b-v03-17k](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k)**, contains 17,000 instruction-response pairs, augmented with synthetic data generated using GPT-4 and self-reflection techniques to enhance diversity and coverage of banking scenarios. The reformatting process involved structuring each entry with a `prompt` (user query) and `response` (answer), wrapped in `[INST]` and `[/INST]` tokens for instruction tuning.[](https://medium.com/%40ahmet_celebi/create-react-fine-tuning-dataset-for-mistral-7b-instruct-v0-3-d6556bab7c56)

2. **Preprocessing and Reformatting**:
   - The original dataset was processed using a custom script to convert CSV entries into a JSONL format compatible with the Hugging Face Transformers library. This involved cleaning the data to remove duplicates, handling missing values, and ensuring consistent formatting.
   - The reformatting script validated the data to avoid errors during training, as described in Mistral’s fine-tuning documentation. For example, problematic samples were skipped to ensure data integrity.[](https://docs.mistral.ai/guides/finetuning/)
   - The reformatted dataset was uploaded to Hugging Face as **[FAQs-Mistral-7b-v03-17k](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k)**, making it publicly accessible for reproducibility.

3. **Fine-Tuning Setup**:
   - **Environment**: The fine-tuning was performed on a Google Colab T4 GPU to leverage free computational resources, with the process optimized for memory efficiency using **4-bit quantization** via the `BitsAndBytesConfig` library.[](https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe)
   - **Technique**: Parameter-Efficient Fine-Tuning (PEFT) with **Low-Rank Adaptation (LoRA)** was used to reduce memory requirements and training time. The LoRA configuration included:
     - `r=32` (LoRA attention dimension)
     - `lora_alpha=64` (scaling factor)
     - `lora_dropout=0.05` (dropout probability)
     - Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]`
     - Task type: `CAUSAL_LM`[](https://huggingface.co/blog/nroggendorff/finetune-mistral)
   - **Training Parameters**:
     - Learning rate: `2e-4`
     - Number of epochs: 1
     - Batch size: 4 (per device)
     - Gradient accumulation steps: 1
     - Gradient checkpointing enabled to save memory
     - Optimizer: AdamW with weight decay[](https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe)
   - **Tokenizer**: The Mistral tokenizer (`mistralai/Mistral-7B-Instruct-v0.3`) was used, with padding set to the right and the EOS token as the pad token.[](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
   - **Training Duration**: The process took approximately 12 hours on a T4 GPU, handling the 17,000 samples efficiently due to quantization and LoRA.

4. **Model Training**:
   - The **Supervised Fine-Tuning (SFT)** approach was employed using the `SFTTrainer` from the `trl` library, which is optimized for instruction tuning. The trainer processed the reformatted dataset, aligning the model’s outputs with banking-specific responses.[](https://medium.com/%40incle/mistral-7b-fine-tuning-a-step-by-step-guide-52122cdbeca8)
   - The training monitored metrics like training loss and perplexity, with periodic evaluation on a validation split (5% of the dataset) to ensure generalization.[](https://github.com/mistralai/mistral-finetune)
   - Weights & Biases (W&B) was integrated to log training metrics, including loss curves and evaluation metrics, ensuring transparency and reproducibility.[](https://www.datacamp.com/tutorial/mistral-7b-tutorial)

5. **Merging and Validation**:
   - Post-training, the LoRA adapters were merged with the base model using the `PeftModel` class to create a single, fine-tuned model.[](https://huggingface.co/blog/nroggendorff/finetune-mistral)
   - The model was validated on a test set to ensure improved performance on banking queries, achieving higher accuracy and relevance compared to the base model.

6. **Pushing to Hugging Face**:
   - The fine-tuned model was pushed to the Hugging Face Hub using the `huggingface_hub` library. The process involved:
     - Authenticating with a Hugging Face access token via `huggingface-cli login`.
     - Creating a new repository (e.g., `Muhammad-Umer-Khan/SupportGenie-Mistral-7B`) on Hugging Face.
     - Pushing the model and tokenizer using:
       ```python
       model.push_to_hub("Muhammad-Umer-Khan/SupportGenie-Mistral-7B", use_temp_dir=False)
       tokenizer.push_to_hub("Muhammad-Umer-Khan/SupportGenie-Mistral-7B", use_temp_dir=False)
       ```
     - The dataset was also uploaded to `[FAQs-Mistral-7b-v03-17k](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k)` using a similar process:
     - The repository includes the reformatted dataset, model weights, and configuration files for reproducibility.[](https://docs.mistral.ai/guides/finetuning/)

7. **Integration into Project**:
   - The fine-tuned model was integrated into the chatbot pipeline via the `chatbot_analytics.py` file, accessible through the “Fine-Tuned Bot” page in the Streamlit UI. It uses the same chat history management and response generation logic as the base chatbot but leverages the fine-tuned model for enhanced performance.[](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

### **Dataset Details**
- **Original Dataset ([FAQ_Dataset](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQ_Dataset))**:
  - Source: Derived from [BankFAQs.csv](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv).
  - Size: ~1,000 question-answer pairs.
  - Content: Covers banking topics such as account management, loans, security, and payments.
  - Format: CSV with columns for questions and answers.
- **Reformatted Dataset [FAQs-Mistral-7b-v03-17k](https://huggingface.co/datasets/Muhammad-Umer-Khan/FAQs-Mistral-7b-v03-17k)**:
  - Size: 17,000 instruction-response pairs.
  - Augmentation: Synthetic data generated using GPT-4 and self-reflection techniques to expand the dataset, ensuring coverage of diverse banking scenarios.
  - Format: JSONL, with each entry containing a `prompt` (user query wrapped in `[INST]` and `[/INST]`) and a `response` (answer).
  - Example Entry:
    ```json
    {
      "messages": [
        {"role": "user", "content": "[INST] How can I check my account balance? [/INST]"},
        {"role": "assistant", "content": "You can check your account balance by logging into your online banking portal or mobile app, navigating to the account summary section, or visiting an ATM or bank branch."}
      ]
    }
    ```
  - Purpose: Structured for instruction tuning, ensuring compatibility with Mistral’s tokenizer and training pipeline.

### **Performance Improvements**
- The fine-tuned **Mistral-7B-Instruct-v0.3** outperforms the base model in:
  - **Accuracy**: Higher relevance in responses due to domain-specific training.
  - **Contextual Understanding**: Better handling of banking-specific terminology and scenarios.
  - **Response Quality**: More concise and user-friendly answers, reducing ambiguity.
- Validation metrics showed a decrease in perplexity and improved response coherence compared to the base model, as logged via Weights & Biases.[](https://github.com/mistralai/mistral-finetune)

### **Challenges and Solutions**
- **Memory Constraints**: Addressed using 4-bit quantization and LoRA to fit training on a T4 GPU.[](https://adithyask.medium.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model-0f39647b20fe)
- **Dataset Formatting Errors**: Resolved by validating and reformatting the dataset using a custom script, skipping problematic samples.[](https://docs.mistral.ai/guides/finetuning/)
- **Local Loading Issues**: Ensured compatibility by updating the `config.json` file with `"model_type": "mistral"` when loading the model locally in VS Code.[](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

---

## 🛠️ **Tech Stack**
| Technology | Usage |
|------------|-------|
| **Python** | Backend API, Chatbot, Data Processing |
| **Streamlit** | Frontend UI for chatbot, fine-tuned bot, and analytics dashboard |
| **MongoDB Atlas** | Stores chat history, FAQs, and analytics data |
| **FAISS** | Efficient **vector search** for FAQ retrieval |
| **Hugging Face Transformers** | Embedding model for vector similarity and fine-tuned LLM |
| **TextBlob** | Sentiment Analysis |
| **Plotly** | Visualization in **analytics dashboard** |
| **FastAPI** | API layer for chatbot |

---

## 📊 **Dashboard Analytics**

| Visualization Type      | Distribution/Insights |
|------------------------|----------------------|
| **Most Frequently Asked Questions** | ![Feature Importance](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/most_fre_ques.png) |
| **Sentiments Over Time**   | ![Confusion Matrix](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sent_ovr_time.png) |
| **Sentiment Trend** | ![Churn Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sent_trend.png) |
| **Sentiment Distribution**    | ![Customer Tenure Distribution](https://raw.githubusercontent.com/MuhammadUmerKhan/AI-Powered-Customer-Support-and-Analytics-System/main/imgs/sentiment_distribution.png) |

---

## ⚙️ **Setup and Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/AI-Powered-Customer-Support-System.git
cd AI-Powered-Customer-Support-System
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up MongoDB Atlas**
- Create a **MongoDB Atlas** cluster.
- Add your **connection string** to `.env` file:
```ini
MONGO_USER=your_mongodb_username  # (Find in MongoDB Atlas under Database Access)
MONGO_PASSWORD=your_mongodb_password  # (Set while creating the database user)
MONGO_CLUSTER=your_cluster.mongodb.net  # (Find in MongoDB Atlas under Cluster Overview)
MONGO_DB=chatbotDB  # (Set database name, default: chatbotDB)
MONGO_URI=mongodb+srv://your_user:your_password@your_cluster.mongodb.net/chatbotDB?retryWrites=true&w=majority&appName=Cluster0
```

### **4️⃣ Set Up API Keys**
- **Grok API Key** (For LLM-powered responses):
```ini
GROK_API_KEY=your_grok_api_key  # (Obtain from Groq API Dashboard)
```

### **5️⃣ Fine-Tuned LLM Setup**
- The fine-tuned **Mistral-7B-Instruct-v0.3** is hosted on Hugging Face at `[Muhammad-Umer-Khan/SupportGenie-Mistral-7B](https://huggingface.co/Muhammad-Umer-Khan/SupportGenie-Mistral-7B)`.
- Load the model in the project using:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("Muhammad-Umer-Khan/SupportGenie-Mistral-7B")
  tokenizer = AutoTokenizer.from_pretrained("Muhammad-Umer-Khan/SupportGenie-Mistral-7B")
  ```
- Ensure the `config.json` includes `"model_type": "mistral"` to avoid loading errors.[](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

Update `.env` with your **MongoDB credentials and API Key** before running the chatbot.

---

## **🖥️ Running the FastAPI Server**
Once the model is trained and registered, run **FastAPI** to serve real-time predictions:

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```
This starts the FastAPI server on **http://127.0.0.1:8000**.

---

## 📊 **AI-Powered Analytics Dashboard**
The **Analytics Dashboard** provides insights into chatbot interactions.
- 📈 **Sentiment Trends** → Tracks how users feel about responses.
- 🔥 **Most Asked Questions** → Identifies common customer concerns.
- 🕒 **User Engagement Heatmap** → Shows peak chat hours.
- ✅ **Feedback Ratings** → Measures helpful vs. unhelpful responses.
- 🔍 **Fine-Tuned LLM Performance** → Monitors the accuracy and relevance of LLM-generated responses.

---

## 🐳 **Dockerization & Deployment**

You can easily run this project using Docker and share or deploy it from Docker Hub.

### ✅ **Build the Docker Image**

Make sure your `Dockerfile` is correctly set up. Then run:

```bash
docker build -t muhammadumerkhan/customer-churn-predictor .
```

### 🚀 **Run the Docker Container**

Anyone can pull and run the app using:

```bash
docker pull muhammadumerkhan/customer-churn-predictor
docker run -p 8501:8501 muhammadumerkhan/customer-churn-predictor
```

---

## 🛠️ **Future Improvements**
- **User Sessions** → Recognize returning users for personalized interactions.
- **Advanced LLM Fine-Tuning** → Further refine the LLM with larger, diverse datasets.
- **Voice Interaction** → Convert text-based chatbot into a **voice assistant**.
- **Voice-Enabled Chatbot** – Integrate **speech recognition** for voice queries.
- **WhatsApp & Telegram Integration** – Expand support to messaging apps.
- **Advanced Sentiment Analysis** – Use transformer models for better predictions.
- **Proactive Support Suggestions** – Predict user needs based on chat history.

---

### 📢 **Shoutout to [MrJay10](https://github.com/MrJay10/banking-faq-bot/blob/master/BankFAQs.csv) for providing the FAQ**

---

## 🚀 **Want to see a live demo?**
- **Click here: [SupportGenie AI Chatbot](https://ai-powered-customer-support-and-analytics-system.streamlit.app/)**

---

## 📌 **Conclusion**
The **AI-Powered Customer Support System** provides **seamless, intelligent customer interactions** through **FAQ retrieval, fine-tuned LLM responses, sentiment analysis, and analytics**. With **scalable deployment** and **real-time insights**, this project can revolutionize **customer engagement** across multiple industries. 🚀
