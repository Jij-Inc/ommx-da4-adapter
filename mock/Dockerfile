FROM python:3.12-slim

COPY ./ ./app

WORKDIR /app
RUN pip install . --no-cache-dir

CMD ["python", "-m", "mock_da4_server"]
