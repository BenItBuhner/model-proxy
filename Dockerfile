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

# Specify the command to run on container startup using uv run
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9876"]
