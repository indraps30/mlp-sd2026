# Docker container will have Python version: 3.12-slim.
# Check the Python image in https://hub.docker.com/_/python
FROM python:3.12-slim

# Set working directory to execute our program.
WORKDIR /app

# Copy anything needed to Docker container.
COPY ./requirements_api.txt .

# Command to execute when building Docker images.
RUN \
apt-get update && \
apt-get install -y curl && \
pip install --upgrade pip && \
pip install --no-cache-dir -r requirements_api.txt

COPY src ./src
COPY models ./models
COPY config ./config

# Expose port the app runs on.
EXPOSE 8080

# Command to execute when running Docker container.
CMD ["python", "src/api.py"]


# To build and run the Docker container, use the following commands:
# docker build -t api_aq .
# docker run -p 8080:8080 api_aq