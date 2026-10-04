FROM python:3.9-slim-bookworm

EXPOSE 8000

# 设置当前目录为工作目录
WORKDIR /app

# Copy only the necessary files for pip install
COPY requirements.txt /app

# Install shared libraries required by the default PaddleOCR CPU runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . /app

# CMD ["python3", "./main.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
