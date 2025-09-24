FROM python:3.10-slim


WORKDIR /app

COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

COPY main.py .

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
