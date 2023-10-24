# Use an official Streamlit image as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

#Install streamlit
RUN pip install streamlit
RUN pip install matplotlib
RUN pip install torch torchvision

# Copy your Streamlit app script into the container
COPY fetch_run.py ./
COPY Fetch_MLE_Assignment_Xiaoshu_Luo_model_training.ipynb ./

# Copy the data.csv and model.pth files into the container
COPY data_daily.csv ./
COPY fetch_LSTM_model.pth ./
COPY README ./

# Expose the port where Streamlit will run
EXPOSE 8501

# Define the command to run your Streamlit app
# CMD streamlit run fetch_run.py

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "fetch_run.py", "--server.port=8501", "--server.address=127.0.0.1"]