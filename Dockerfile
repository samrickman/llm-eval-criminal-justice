FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

# Sort environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive

# Install uv
RUN pip install uv

# Copy requirements
COPY requirements.txt ./

# ! will need to do the nvidia container toolkit bit here if it's going to work with local files

# Shouldn't need this anymore if the above FROM works
# Install torch first to get CUDA wheels right
# uv pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install -r requirements.txt

# Copy rest of the project
COPY . .

# Install R
RUN chmod +x ./docker/install_r.sh
RUN ./docker/install_r.sh
RUN Rscript ./docker/install_r_packages.R

# Default command
# should probably be run.sh
CMD ["./run.sh"]
