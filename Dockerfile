# if gcc needed change this to python:3.13
FROM python:3.13-slim

# Install uv
RUN pip install uv

# need this to check results
RUN apt install jq 


# Copy requirements
COPY requirements.txt ./

# ! will need to do the nvidia container toolkit bit here if it's going to work with local files

# Install torch first to get CUDA wheels right
uv pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# Copy rest of the project
COPY . .

# Default command
# should probably be run.sh
#CMD ["python", "main.py"]
