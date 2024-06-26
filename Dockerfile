# Dockerfile

# Set the base image to Python 3.10
FROM python:3.10

# Set the working directory within the container
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy the requirements file and install the dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the model_deployment.py script into the container
COPY model.py ./

# Expose port 8888 for external access
EXPOSE 8888

# Define the command to run when the container starts
CMD [ "bash", "-c", "ray start --head --block --object-manager-port=8076 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8266"]
