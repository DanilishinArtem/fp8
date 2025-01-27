
class Config:
    def __init__(self):
        self.batch_size = 64
        self.learning_rate = 0.01
        self.num_epochs = 3
        self.split = 0.8
        self.pathToData = "/home/adanilishin/fp8/mnist_analisys/mnist_dataset"
        self.pathToLogs = "/home/adanilishin/fp8/mnist_analisys/Logs/test"



# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .



# conda create -n apex_env python=3.11
# conda activate apex_env
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# export CUDA_HOME=/usr/local/cuda-12.3
# pip install -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings="--build-option=--cpp_ext" --config-settings="--build-option=--cuda_ext" .