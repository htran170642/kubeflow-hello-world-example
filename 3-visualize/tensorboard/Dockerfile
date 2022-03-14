FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# ADD examples/v1beta1/trial-images/pytorch-mnist /opt/pytorch-mnist
WORKDIR /opt/pytorch-mnist

# Add folder for the logs.
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .