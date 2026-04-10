# Rebuilt: 2026-04-10
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY server/ ./server/
COPY *.py .
COPY openenv.yaml .

EXPOSE 7860

# Run inference.py which has all OpenEnv endpoints + LLM proxy calls
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]