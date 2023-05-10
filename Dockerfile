# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY  requirements.txt RL_train.py /app/

# Set the environment variable to disable output buffering
ENV PYTHONUNBUFFERED=1

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# Make port 6006 available to the world outside this container
EXPOSE 6006

# Run your script when the container launches
CMD ["python", "RL_train.py"]