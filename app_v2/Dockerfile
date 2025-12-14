FROM python:3.12-slim

WORKDIR /app

# 시스템 의존성 (psycopg 빌드 이슈 예방용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_v2 ./app_v2

EXPOSE 8000
CMD ["uvicorn", "app_v2.main:app", "--host", "0.0.0.0", "--port", "8000"]
