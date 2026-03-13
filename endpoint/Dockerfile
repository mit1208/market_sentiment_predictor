FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY endpoint/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import joblib, lightgbm"
COPY endpoint/main.py      ./main.py
COPY endpoint/predictor.py  ./predictor.py
COPY shared/ ./shared/
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
