# Dockerfile

# Use an existing TensorFlow Docker image as the base
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Install Flask and other dependencies
RUN pip install Flask
RUN pip install -r requirements.txt

# Expose the Flask port
EXPOSE 5000

# Specify the command to run your application
CMD ["python", "your_tensorflow_application.py"]
