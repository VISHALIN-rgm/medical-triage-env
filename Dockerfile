FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/
COPY *.py .
COPY web_ui.html .

COPY openenv.yaml .


EXPOSE 7860

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "7860"]
