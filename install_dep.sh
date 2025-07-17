export CUDA_HOME=/usr/local/cuda-12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
echo "Switched to CUDA 12.9"

sudo apt-get update
sudo apt-get install swig
sudo apt-get install -y libcudnn8 libcudnn8-dev
sudo apt-get install libgl1-mesa-glx
sudo apt-get install libglib2.0

uv venv --python 3.12 --seed
source .venv/bin/activate
pip3 install -r requirements.txt
