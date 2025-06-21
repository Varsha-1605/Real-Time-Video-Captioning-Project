# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir ensures the image is smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Make port 5000 available to the world outside this container
# Your app runs on port 5000 according to your script
EXPOSE 5000

# Define environment variables if needed
ENV FLASK_APP=app.py

# Command to run your application using gunicorn for production
# This is more robust than `flask run`
# CMD ["gunicorn", "--workers", "1", "--threads", "4", "--bind", "0.0.0.0:5000", "--log-level", "info", "app:app"]

CMD ["python", "app.py"]