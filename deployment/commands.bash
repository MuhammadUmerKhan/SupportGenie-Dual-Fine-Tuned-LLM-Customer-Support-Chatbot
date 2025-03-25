# Navigate to deployment folder (if needed)
cd deployment

# 1️⃣ Build the Docker Image
docker build -t ai-customer-support .

# 2️⃣ Run the Container
docker run -p 8501:8501 ai-customer-support