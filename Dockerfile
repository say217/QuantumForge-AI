FROM python:3.10-slim  # Your choice – perfect!

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir \
    torch==2.4.1+cpu torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN mkdir -p models reports
EXPOSE 8888
CMD ["python", "main.py"]
