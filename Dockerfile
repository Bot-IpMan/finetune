FROM python:3.10-slim

# Install system dependencies for bitsandbytes and other libs
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default entrypoint runs training then starts the inference server.
# Training arguments can be supplied via `command` in docker-compose which will
# be forwarded to the script.
ENTRYPOINT ["/app/scripts/train_and_serve.sh"]
CMD []
