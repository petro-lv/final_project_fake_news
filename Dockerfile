FROM python:3

COPY . /workspace

WORKDIR /workspace

RUN chown -R 42420:42420 /workspace

ENV HOME=/workspace

RUN pip install -r requirements.txt

CMD ["python3", "app/app.py"]
