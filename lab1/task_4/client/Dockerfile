FROM python

RUN pip install requests

ADD client.py .

ENTRYPOINT [ "python3", "client.py" ]