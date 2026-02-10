FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data vectorstore

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DATA_PATH=/app/data
ENV VECTORSTORE_PATH=/app/vectorstore

# Expose port for API
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
