# frontent/Dockerfile

FROM python:3.8-slim-buster

COPY requirements.txt app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

RUN pip install ndg-httpsclient

RUN pip install pyopenssl

RUN pip install pyasn1

COPY . /app

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["app.py"]