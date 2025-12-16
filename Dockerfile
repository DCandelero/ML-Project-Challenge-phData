FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY ml/ ./ml/
COPY model/ ./model/
COPY data/zipcode_demographics.csv ./data/zipcode_demographics.csv

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
