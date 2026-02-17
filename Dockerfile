FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

# -------------------------------------------------------
# 1. Dependências básicas do sistema
# -------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python2.7 python2.7-dev python-pip \
        build-essential \
        gcc-6 g++-6 \
        git \
        libhdf5-dev \
        libpng-dev \
        libtiff-dev \
        libjpeg-dev \
        libopenblas-dev \
        liblapack-dev \
        libxft-dev \
        libxml2-dev \
        libxslt1-dev \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

# Ajustar GCC padrão para garantir compatibilidade com libs antigas
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 60

# -------------------------------------------------------
# 2. Atualizar pip do Python 2.7 para a última versão suportada
# -------------------------------------------------------
# RUN python2 -m pip install --upgrade pip==20.3.4 setuptools==44.1.1 wheel==0.34.2
RUN python2 -m pip install --upgrade --no-cache-dir  pip==20.3.4 setuptools==40.6.3 wheel==0.32.3

# -------------------------------------------------------
# 3. Instalar bibliotecas EXATAS do ambiente do estudo
# -------------------------------------------------------
RUN pip install  --no-cache-dir numpy==1.16.6 \
    --no-cache-dir scipy==1.2.3 \
    h5py==2.10.0 \
    matplotlib==2.2.5 \
    SimpleITK==1.2.4 \
    numexpr==2.7.1 \
    --no-cache-dir tables==3.5.1 \
    tensorflow-gpu==1.14.0 \
    keras==2.3.1 \ 
    pyaml==19.12.0 \ 
    scikit-image==0.14.5

# -------------------------------------------------------
# 4. Criar diretório de trabalho
# -------------------------------------------------------
WORKDIR /workspace

#CMD ["/bin/bash"]
