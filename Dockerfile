# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy current directory contents into container at /app
COPY . /app

# Install dependencies if you have requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make run.sh executable
RUN chmod +x run.sh

# Default command to run when container starts
ENTRYPOINT ["python3", "test.py", "$inputDataset/dataset.jsonl", "$outputDir"]