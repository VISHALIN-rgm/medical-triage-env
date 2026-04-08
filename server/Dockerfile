FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-dotenv

# Copy application files
COPY server/ ./server/
COPY *.py .
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
