FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ src/
COPY models/ models/
COPY scripts/ scripts/

ENV PYTHONPATH=/app

CMD ["python", "scripts/run_batch_scoring.py"]