FROM python

ARG flask_version=1.1.2

RUN pip install flask==${flask_version}

ADD flask_api.py .

ENV ENV_VAR Kornel

ENTRYPOINT [ "python3", "flask_api.py" ]
CMD [ "--port", "5000" ]
