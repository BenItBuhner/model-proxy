# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy dependency files first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the application code
COPY . .

# Create necessary directories for persistent storage
RUN mkdir -p /app/config/providers /app/config/models /app/config/templates

# Install the package in editable mode
RUN uv pip install -e .

# Expose the default port
EXPOSE 9876

# Specify the command to run on container startup
# The setup UI will be available at http://localhost:9876/setup/
CMD ["model-proxy", "start", "--host", "0.0.0.0", "--port", "9876"]
