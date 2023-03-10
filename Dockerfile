FROM python:3

COPY . /workspace

WORKDIR /workspace

RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["app/app.py"]
