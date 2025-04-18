# Stage 1: Build Stage
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Install application dependencies to /install
RUN pip install --prefix=/install .

# Stage 2: Runtime Stage
FROM python:3.9-slim

# Set environment variables
ENV PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.9/site-packages"

# Set working directory
WORKDIR /app

# Copy installed packages and application code from builder stage
COPY --from=builder /install /install
COPY --from=builder /app /app

# Expose port if necessary (adjust if your application uses a different port)
EXPOSE 80

# Define environment variable
ENV NAME=SlackSentimentApp

# Command to run the application
CMD ["python", "bot.py"]

