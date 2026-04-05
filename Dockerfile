# Use the official Python 3.10 image as required by the OpenEnv spec
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to cache the installations
COPY requirements.txt .

# Install the required packages safely
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your backend engine files into the container
COPY . .

# When the judges run the container, automatically trigger the baseline test
CMD ["python", "run_baseline.py"]