# Use lightweight Python image
FROM python:3.11-slim
WORKDIR /app
# Copy requirements file (use --no-cache-dir to avoid pip cache)
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip
# Copy source code into the container
COPY src/ ./src

# Make sure Python can find src/
ENV PYTHONPATH="/app"

# Create a non-root user and switch to it for security
RUN addgroup --system appuser && \
    adduser --system --ingroup appuser appuser && \
    chown -R appuser:appuser /app
USER appuser
# Expose the port the app runs on
EXPOSE 8000
# Create a directory for endpoints
# Command to run the application
CMD ["uvicorn", "src.endpoint.main:app", "--host", "0.0.0", "--port", "8000"]