dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements from server folder
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY server/ ./server/
COPY *.py .
COPY openenv.yaml .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
