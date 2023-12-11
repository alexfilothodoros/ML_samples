FROM python:3.10-slim

EXPOSE 8501

WORKDIR /app

COPY . .

# RUN apt-get update &&\
#    apt-get install --no-install-recommends --yes build-essential

# RUN pip install -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
