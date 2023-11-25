# Use a Python base image
FROM python:3.8.10-slim-buster
# Set the working directory
WORKDIR /mini-project


# Copy the application code
COPY . .



# Install dependencies
RUN apt-get  update \
     && apt-get  install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        tesseract-ocr-ara
RUN pip install --no-cache-dir  -r requirements.txt

# Expose the application port
EXPOSE 8000

# Set the entrypoint
CMD ["python", "app.py"]
