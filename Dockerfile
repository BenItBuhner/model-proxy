# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy the dependency files to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Install dependencies using uv for better performance
RUN uv sync --frozen

# Copy the content of the local src directory to the working directory
COPY . .

# Make the CLI entry points available
RUN uv pip install -e .

# Specify the command to run on container startup using the model-proxy CLI
# These flags can be overridden at runtime, e.g.:
# docker run model-proxy start --port 8000
# or via docker-compose environment variables
CMD ["model-proxy", "start", "--host", "0.0.0.0", "--port", "9876"]
