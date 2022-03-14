FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

# ADD examples/v1beta1/trial-images/pytorch-mnist /opt/pytorch-mnist
WORKDIR /opt/pytorch-mnist

# Add folder for the logs.
COPY *.py .

ENTRYPOINT ["python3", "/opt/pytorch-mnist/mnist.py"]