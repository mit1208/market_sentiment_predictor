FROM python:3.11-slim

WORKDIR /app

COPY backend/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /app

CMD ["uvicorn", "backend.swarm.main:app", "--host", "0.0.0.0", "--port", "8081"]
