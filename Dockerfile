FROM python:3.12-slim

RUN useradd -m myuser

WORKDIR /app

# Comment this out if you do not have the model in your directory 
COPY /all-MiniLM-L6-v2 /home/myuser/.cache/chroma/onnx_models/all-MiniLM-L6-v2 


COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY /myapp .

COPY /persistent /app/persistent

RUN chown -R myuser:myuser /app

USER myuser

EXPOSE 8001

CMD ["gunicorn", "-b", "0.0.0.0:8001", "--workers=4", "app:app"]