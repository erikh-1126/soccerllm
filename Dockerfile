
# 1. Base image: Python 3.11 slim
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements and install
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    llama-cpp-python==0.2.79

# 4. Copy application code
COPY src/ ./src
COPY models/ ./models

# 5. Environment variables
ENV MODEL_PATH="/app/models/Meta-Llama-3-8B.Q4_K_M.gguf"
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# 6. Expose port and set default command
EXPOSE 5000

CMD ["python", "src/api.py"]
