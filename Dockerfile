FROM python:3.10.13 

# Set the working directory in the container
WORKDIR /app

# Copy the Streamlit application file into the container
COPY dashboard.py /app/

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir streamlit tensorflow pandas numpy

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
