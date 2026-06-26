FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port and run FastAPI
EXPOSE 8000
CMD ["uvicorn", "src.football_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
