Here's a **detailed README** for the **deployment folder**, explaining how to build, run, and deploy the project using Docker. 🚀  

---

### 📄 **`deployment/README.md`**
```markdown
# 🚀 AI Customer Support System - Deployment Guide

This guide explains how to **deploy** the **AI-Powered Customer Support System** using **Docker**.

---

## 🐳 **1️⃣ Docker Installation**
First, ensure **Docker is installed** on your machine. If not, install it:

🔗 **Download Docker:**  
[Docker Desktop](https://www.docker.com/get-started/)

To verify the installation, run:
```sh
docker --version
```

---

## 📦 **2️⃣ Build the Docker Image**
Run the following command **inside the project root folder**:

```sh
docker build -t ai-customer-support .
```
📌 This will:
- Install **Python & dependencies**
- Copy project files inside the container
- Set up the Streamlit app

---

## 🚀 **3️⃣ Run the Docker Container**
Once built, start the app:

```sh
docker run -p 8501:8501 ai-customer-support
```
📌 This will:
- Start the **Streamlit app** on `http://localhost:8501`
- Keep the app running inside Docker

---

## ⚙️ **4️⃣ Using Docker Compose (Optional)**
If you want to **run MongoDB & the chatbot together**, use **Docker Compose**:

```sh
docker-compose up -d
```

📌 **Ensure `docker-compose.yml` contains:**  
```yaml
version: '3.8'
services:
  chatbot:
    build: .
    ports:
      - "8501:8501"
    depends_on:
      - mongodb

  mongodb:
    image: mongo
    container_name: mongo_db
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

volumes:
  mongo_data:
```

---

## 🛑 **5️⃣ Stopping & Removing Containers**
To **stop the container**, run:
```sh
docker stop <container_id>
```

To **remove all containers**, run:
```sh
docker rm $(docker ps -aq)
```

To **remove the Docker image**, run:
```sh
docker rmi ai-customer-support
```

---

## ✅ **6️⃣ Deployment on Cloud**
You can deploy the container to:
1. **Streamlit Cloud** (Directly running `main.py`)
2. **AWS, GCP, or Azure** (Using Docker & Kubernetes)
3. **Vercel** (Using `vercel.json`)

---

## 🎯 **Summary**
✅ **Dockerfile** ensures the chatbot can run in a **containerized environment**  
✅ **Docker Compose** allows running **MongoDB + Chatbot together**  
✅ **Streamlit & MongoDB** can be deployed easily to **cloud platforms**  

---

## 🚀 Happy Deployment!
```

---

### **📌 Where to Save This?**
📌 **Path:** `deployment/README.md`

Would you like me to generate **a `start.sh` script** for automating the setup? 🤖🚀