FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY server/requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-dotenv

# Copy all files
COPY server/ ./server/
COPY *.py .
COPY openenv.yaml .

# Create a simple test endpoint
RUN echo 'from server.app import app' > app.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]