# Danny Clemens
#
# Dockerfile
#
# MLOPS File

FROM python:3.13

# Set the working directory
WORKDIR /app

# Required modules
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the project files
COPY . .

# Run training script
CMD ["python", "train.py"]
