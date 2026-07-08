FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv
RUN uv pip install --system .
# RUN uv export --frozen --no-dev > requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

# CMD ["sh", "-c", "uv run streamlit run main.py"]

# For GCP
CMD ["sh", "-c", "uv run streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]

# CMD ["uv", "run", "streamlit", "run", "main.py"]
# CMD sh -c "streamlit run app/main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"