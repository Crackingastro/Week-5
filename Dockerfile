# 1. Base image
FROM python:3.10-slim

# 2. Set a working directory
WORKDIR /app

# 3. Install OS deps (if any) and Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 4. Copy in your source code
COPY src/ ./src

# 5. Expose the port Uvicorn will serve on
EXPOSE 8000

# 6. Run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
