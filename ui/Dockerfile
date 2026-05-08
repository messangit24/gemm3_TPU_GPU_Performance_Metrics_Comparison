FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install streamlit openai PyPDF2 google-cloud-monitoring google-cloud-bigquery db-dtypes requests pandas plotly matplotlib
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
