# Use official Python slim image
FROM python:3.11-slim

# Set environment
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create app directory
WORKDIR $APP_HOME

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Non-root user
RUN useradd -m -u 1000 covenant && chown -R covenant:covenant $APP_HOME
USER covenant

# Default command (adjust to your entrypoint)
CMD ["python", "src/covenant/main.py"]
