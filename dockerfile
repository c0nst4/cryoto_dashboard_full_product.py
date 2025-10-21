# Basis-Image mit Python
FROM python:3.10-slim

# Arbeitsverzeichnis erstellen
WORKDIR /app

# requirements installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit App kopieren
COPY crypto_dashboard_full_product.py .

# Port für Streamlit
EXPOSE 7860

# Streamlit starten
CMD ["streamlit", "run", "crypto_dashboard_full_product.py", "--server.port=7860", "--server.address=0.0.0.0"]
