FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App modules
COPY app.py soc_lp_final.py chart_export.py ./

# Price data. all_countries.csv (155 MB) is deliberately excluded via
# .dockerignore — app.py never offers it as a region.
COPY european_wholesale_electricity_price_data_hourly/ ./european_wholesale_electricity_price_data_hourly/
COPY caiso_wholesale_electricity_price_data_hourly/    ./caiso_wholesale_electricity_price_data_hourly/
COPY aemo_wholesale_electricity_price_data_hourly/     ./aemo_wholesale_electricity_price_data_hourly/

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
