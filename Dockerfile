# WatershedShift private deployment: Streamlit app with interactive forecasting.
# Model artifacts and caches are mounted at run time; nothing private is baked into the image.
FROM python:3.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    WATERSHED_DEPLOYMENT=private \
    WATERSHED_CACHE_DIR=/data/cache \
    WATERSHED_RECORD_DIR=/data/records
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
RUN useradd --create-home app && mkdir -p /data && chown -R app /data /app
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
