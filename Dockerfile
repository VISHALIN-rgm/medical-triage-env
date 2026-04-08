FROM python:3.10-slim

WORKDIR /app

# Install minimal dependencies
RUN pip install fastapi uvicorn

# Copy only essential files
COPY server/app.py ./server/
COPY server/__init__.py ./server/
COPY models.py .
COPY openenv.yaml .

# Create a simple health endpoint for testing
RUN echo 'from server.app import app' > app.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]