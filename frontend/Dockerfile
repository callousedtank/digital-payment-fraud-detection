FROM python:3.13-slim

WORKDIR /app

COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/streamlit_app.py .

EXPOSE 8501

CMD streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8501}
