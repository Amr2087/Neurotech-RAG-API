# Use the official Python 3.12.4 slim image.
FROM python:3.12.4-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file first, so that dependency installation is cached.
COPY requirements.txt .

# Upgrade pip and install dependencies.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Expose the port that your Flask app listens on.
EXPOSE 5000

# Optionally, you can define a default value (or leave it unset) for OPENAI_API_KEY:
# ENV OPENAI_API_KEY=your_default_api_key_here

# Define the command to run your Flask application.
CMD ["python", "app.py"]
