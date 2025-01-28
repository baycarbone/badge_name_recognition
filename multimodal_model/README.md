# Readme

- clone repo
- create a python venv: `python3 -m venv .venv`
- activate the venv: `. .venv/bin/activate`
- install dependencies
```
pip install --no-cache-dir -r ./requirements.txt
```
Or individually (might be missing some here):
```
pip install transformers
pip install tokenizers
pip install datasets
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]
```
- kserve is used to expose the model as a prediction API when fed with a prompt and a URL to an image
- You can chose to run the model directly from python or using a container image using the provided Dockerfile

# Run as python app
```
python kserve_model.py
```

# Run as container
- There are 2 Dockerfile, one uses the a standard python base image while the other uses an nvidia pytorch base image. To use the nvidia image you will need to have access to https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch.
- Using the nvidia image is not mandatory but will save you time when building the image as it already contains a number of the required packages.

# Testing on AWS
- deploy a VM with GPU e.g. g5.2xlarge
- install necessary drivers (https://documentation.ubuntu.com/aws/en/latest/aws-how-to/instances/install-nvidia-drivers/)
```
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
nvidia-smi


Sun Oct 13 09:13:03 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.107.02             Driver Version: 550.107.02     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A10G                    Off |   00000000:00:1E.0 Off |                    0 |
|  0%   15C    P8              9W /  300W |       1MiB /  23028MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```