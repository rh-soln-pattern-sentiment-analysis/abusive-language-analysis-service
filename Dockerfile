FROM registry.access.redhat.com/ubi8/python-39:1-97

USER 0

RUN mkdir /app
RUN mkdir -p /app/cache

ENV TRANSFORMERS_CACHE=/app/cache/

COPY requirements.txt /app

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY abusive_language_analysis.py /app/app.py

RUN chown -R 1001:0 /app\
&&  chmod -R og+rwx /app \
&&  chmod -R +x /app

WORKDIR /app/

ENV PYTHONPATH=/app

USER 1001

CMD gunicorn -w 1 --threads $G_THREADS --timeout $G_TIMEOUT  -b 0.0.0.0:8080 'app:app'
